import copy
import multiprocessing
from multiprocessing import pool

import numpy as np
import itertools
from utils.camera_utils import project_to_image
from mesh_gnn.rollout import init_worker, rollout_worker
from utils.camera_utils import get_target_pos


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class Pool(pool.Pool):
    Process = NoDaemonProcess


class RandomShootingUVPickandPlacePlanner:

    def __init__(self, num_pick,
                 pull_step,  # stage_2_step,
                 wait_step,  # stage_3_step,
                 reward_model=None,
                 dynamics=None, num_worker=10,
                 move_distance_range=[0.05, 0.2], gpu_num=1,
                 image_size=None, normalize_info=None, delta_y_range=None,
                 matrix_world_to_camera=np.identity(4),
                 task='flatten',
                 pick_vis_only=False,
                 env=None
                 ):
        """
        cem with mpc
        move_distance_range is only used if move_distance is None
        delta_y_range is used if it is not None
        """

        self.normalize_info = normalize_info  # Used for robot experiments to denormalize before action clipping
        self.num_pick = num_pick
        self.delta_y_range = delta_y_range  # if not None, will override delta_y
        self.move_distance_low, self.move_distance_high = move_distance_range[0], move_distance_range[1]
        self.reward_model = reward_model
        self.pull_step, self.wait_step = pull_step, wait_step
        self.gpu_num = gpu_num
        self.pick_vis_only = pick_vis_only
        self.dynamics = dynamics
        assert num_worker > 0
        self.pool = Pool(processes=num_worker,
                         initializer=init_worker,
                         initargs=(dynamics.args, dynamics.get_main_model(), None, None, 0, env))
        self.num_worker = num_worker
        self.matrix_world_to_camera = matrix_world_to_camera
        self.image_size = image_size

        print("self.num_pick is: ", self.num_pick)
        self.task = task

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def project_3d(self, pos):
        return project_to_image(self.matrix_world_to_camera, pos, self.image_size[0], self.image_size[1])

    def get_action(self,
                   init_data,
                   dyn_args,
                   gpu_id=0,
                   robot_exp=False,
                   depth=None,
                   m_name='vsbl',
                   ):
        """
        init_data should be a list that include:
            ['positions', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params', 'observable_particle_indices]
            note: require position, velocity to be already downsampled

        """
        dyn_args.name = m_name

        data = init_data.copy()
        data['picked_points'] = [-1, -1]

        pull_step, wait_step = self.pull_step, self.wait_step

        # add a no-op action
        pick_try_num = self.num_pick + 1
        actions = np.zeros((pick_try_num, pull_step + wait_step, 8))

        verts = data["verts"]
        verts_vis = data["verts_vis"]
        bb_margin = 30

        us, vs = self.project_3d(verts)
        delta_y = self.delta_y_range
        params = [
            (us, vs, self.image_size, verts, verts_vis, pull_step,
             delta_y, bb_margin, self.matrix_world_to_camera,
             self.move_distance_low, self.move_distance_high, depth, self.pick_vis_only)
            for _ in range(self.num_pick)
        ]
        results = self.pool.map(parallel_generate_actions, params)

        delta_moves = [x[0] for x in results]
        start_poses = [x[1] for x in results]
        after_poses = [x[2] for x in results]

        start_poses.append(after_poses[0])
        after_poses.append(after_poses[0])

        # omit stage 1 in simulation
        actions[:-1, :pull_step, :3] = np.vstack(delta_moves)[:, None, :]
        actions[:-1, :pull_step, 3] = 1
        actions[:, :, 4:] = 0

        all_returns = []

        data['pointcloud'] = verts
        data_cpy = copy.deepcopy(data)
        params = []
        # job_each_gpu = pick_try_num // self.gpu_num

        for i in range(pick_try_num):
            data_cpy['picked_points'] = [-1, -1]
            data_cpy['picker_position'][0, :] = start_poses[i]

            params.append(
                dict(
                    model_input_data=copy.deepcopy(data_cpy),
                    actions=actions[i],
                    m_name=m_name,
                    reward_model=self.reward_model,
                    cuda_idx=gpu_id,
                    robot_exp=robot_exp,
                    args=dyn_args,
                    task=self.task,
                )
            )

        N = len(params)
        batch_size = int(np.ceil(N / self.num_worker))
        # for i in range(self.num_worker):
        #     rollout_worker(params[i * batch_size: min(len(params), (i + 1) * batch_size)])
        results = self.pool.map(rollout_worker, [params[i * batch_size:
                                                        min(len(params), (i + 1) * batch_size)]
                                                 for i in range(self.num_worker)])
        results = list(itertools.chain(*results))
        # Log return of different trajectories
        returns = [x['final_ret'] for x in results]
        all_returns.append(returns)
        infos = [x['info'] for x in results]

        ret_info = {}
        highest_return_idx = np.argmax(returns)
        ret_info['highest_return_idx'] = highest_return_idx
        action_seq = actions[highest_return_idx]
        ret_info['start_pos'] = start_poses[highest_return_idx]
        ret_info['after_pos'] = after_poses[highest_return_idx]
        model_predict_particle_positions = results[highest_return_idx]['model_positions']
        model_predict_shape_positions = results[highest_return_idx]['shape_positions']
        canon_tgt = results[highest_return_idx]['canon_tgt']
        if dyn_args.reward_model:
            ret_info['gt_rewards'] = np.array([info['gt_rewards'] for info in infos])
            ret_info['model_rewards'] = np.array([info['model_rewards'] for info in infos])
        planning_results = {
            'action_seq': action_seq,
            'model_predict_particle_positions': model_predict_particle_positions,
            'model_predict_shape_positions': model_predict_shape_positions,
            'model_canon_tgt': canon_tgt,
            'ret_info': ret_info
        }
        return planning_results

    def close_pool(self):
        self.pool.close()
        self.pool.terminate()
        self.pool = None


def pos_in_image(after_pos, matrix_world_to_camera, image_size):
    euv = project_to_image(matrix_world_to_camera, after_pos.reshape((1, 3)), image_size[0],
                           image_size[1])
    u, v = euv[0][0], euv[1][0]
    if 0 <= u < image_size[1] and v >= 0 and v < image_size[0]:
        return True
    else:
        return False


def parallel_generate_actions(args):
    us, vs, image_size, verts, verts_vis, pull_step, \
        delta_y, bb_margin, matrix_world_to_camera, move_distance_low, move_distance_high, \
        depth, pick_vis_only = args

    lb_u, lb_v, ub_u, ub_v = int(np.min(us)), int(np.min(vs)), int(np.max(us)), int(np.max(vs))
    u = np.random.randint(max(lb_u - bb_margin, 0), min(ub_u + bb_margin, image_size[1]))
    v = np.random.randint(max(lb_v - bb_margin, 0), min(ub_v + bb_margin, image_size[0]))
    target_pos = get_target_pos(verts, u, v, image_size, matrix_world_to_camera, depth, verts_vis,
                                pick_vis_only)

    # second stage: choose a random (x, z) direction, move towards that direction for 30 steps.
    while True:
        move_direction = np.random.rand(3) - 0.5
        move_direction[1] = np.random.uniform(delta_y[0], delta_y[1])

        move_direction = move_direction / np.linalg.norm(move_direction)
        move_distance = np.random.uniform(move_distance_low, move_distance_high)
        delta_move = move_distance / pull_step * move_direction

        after_pos = target_pos + move_distance * move_direction
        if pos_in_image(after_pos, matrix_world_to_camera, image_size):
            break

    # return target_pos, delta_move
    return delta_move, target_pos, after_pos
