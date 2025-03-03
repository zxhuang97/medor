import numpy as np
from multiprocessing import pool
import copy
from VCD.utils.camera_utils import project_to_image, get_target_pos


class RandomShootingUVPickandPlacePlanner():

    def __init__(self, num_pick, delta_y, pull_step, wait_step,
                    dynamics, reward_model, num_worker=10,
                    move_distance_range=[0.05, 0.2], gpu_num=1,
                    image_size=None, normalize_info=None, delta_y_range=None,
                    matrix_world_to_camera=np.identity(4), task='flatten',
                    use_pred_rwd=False):
        """
        Random Shooting planner.
        """

        self.normalize_info = normalize_info  # Used for robot experiments to denormalize before action clipping
        self.num_pick = num_pick
        self.delta_y = delta_y # for real world experiment, delta_y (pick_height) is fixed
        self.delta_y_range = delta_y_range  # for simulation, delta_y is randomlized
        self.move_distance_low, self.move_distance_high = move_distance_range[0], move_distance_range[1]
        self.reward_model = reward_model
        self.dynamics = dynamics
        self.pull_step, self.wait_step = pull_step, wait_step
        self.gpu_num = gpu_num
        self.use_pred_rwd = use_pred_rwd

        if num_worker > 0:
            self.pool = pool.Pool(processes=num_worker)
        self.num_worker = num_worker
        self.matrix_world_to_camera = matrix_world_to_camera
        self.image_size = image_size
        self.task = task

    def project_3d(self, pos):
        return project_to_image(self.matrix_world_to_camera, pos, self.image_size[0], self.image_size[1])

    def get_action(self, init_data, robot_exp=False, cloth_mask=None, check_mask=None, m_name='vsbl'):
        """
        check_mask: Used to filter out place points that are on the cloth.
        init_data should be a list that include:
            ['pointcloud', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params', 'observable_particle_indices]
            note: require position, velocity to be already downsampled

        """
        args = self.dynamics.args
        data = init_data.copy()
        data['picked_points'] = [-1, -1]

        pull_step, wait_step = self.pull_step, self.wait_step

        # add a no-op action
        pick_try_num = self.num_pick + 1 if self.task == 'flatten' else self.num_pick
        actions = np.zeros((pick_try_num, pull_step + wait_step, 8))
        pointcloud = copy.deepcopy(data['pointcloud'])

        picker_pos = data['picker_position'][0][:3] if data['picker_position'] is not None else None
        bb_margin = 30


        us, vs = self.project_3d(pointcloud)
        params = [
            (us, vs, self.image_size, pointcloud, pull_step,
            self.delta_y_range, bb_margin, self.matrix_world_to_camera,
            self.move_distance_low, self.move_distance_high, cloth_mask, self.task)
            for i in range(self.num_pick)
        ]
        results = self.pool.map(parallel_generate_actions, params)
        delta_moves, start_poses, after_poses = [x[0] for x in results], [x[1] for x in results], [x[2] for x in results]
        if self.task == 'flatten': # add a no-op action
            start_poses.append(data['picker_position'][0, :])
            after_poses.append(data['picker_position'][0, :])

        actions[:-1, :pull_step, :3] = np.vstack(delta_moves)[:, None, :]
        actions[:-1, :pull_step, 3] = 1
        actions[:, :, 4:] = 0
        move_vec = None

        # parallely rollout the dynamics model with the sampled action seqeunces
        data_cpy = copy.deepcopy(data)
        if self.num_worker > 0:
            job_each_gpu = pick_try_num // self.gpu_num
            params = []
            for i in range(pick_try_num):
                if robot_exp:
                    data_cpy['picked_points'] = [picked_particles[i], -1]
                else:
                    data_cpy['picked_points'] = [-1, -1]
                    data_cpy['picker_position'][0, :] = start_poses[i]

                gpu_id = i // job_each_gpu if i < self.gpu_num * job_each_gpu else i % self.gpu_num
                params.append(
                    dict(
                        model_input_data=copy.deepcopy(data_cpy), actions=actions[i], m_name=m_name,
                        reward_model=self.reward_model, cuda_idx=gpu_id, robot_exp=robot_exp,
                    )
                )
            results = self.pool.map(self.dynamics.rollout, params, chunksize=max(1, pick_try_num // self.num_worker))
            returns = [x['final_ret'] for x in results]
        else: # sequentially rollout each sampled action trajectory
            returns, results = [], []
            for i in range(pick_try_num):
                res = self.dynamics.rollout(
                    dict(
                        model_input_data=copy.deepcopy(data_cpy), actions=actions[i], m_name=m_name,
                        reward_model=self.reward_model, cuda_idx=0, robot_exp=robot_exp,
                    )
                )
                results.append(res), returns.append(res['final_ret'])

        ret_info = {}
        highest_return_idx = np.argmax(returns)

        ret_info['highest_return_idx'] = highest_return_idx
        action_seq = actions[highest_return_idx]
        if robot_exp:
            ret_info['waypoints'] = np.array(waypoints[highest_return_idx]).copy()
            ret_info['all_candidate'] = np.array(waypoints[:-1])
            ret_info['all_candidate_rewards'] = np.array(returns[:-1])
        else:
            ret_info['start_pos'] = start_poses[highest_return_idx]
            ret_info['after_pos'] = after_poses[highest_return_idx]

        model_predict_particle_positions = results[highest_return_idx]['model_positions']
        model_predict_shape_positions = results[highest_return_idx]['shape_positions']
        predicted_edges = results[highest_return_idx]['mesh_edges']


        return action_seq, model_predict_particle_positions, model_predict_shape_positions, ret_info, predicted_edges


def pos_in_image(after_pos, matrix_world_to_camera, image_size):
    euv = project_to_image(matrix_world_to_camera, after_pos.reshape((1, 3)), image_size[0], image_size[1])
    u, v = euv[0][0], euv[1][0]
    if u >= 0 and u < image_size[1] and v >= 0 and v < image_size[0]:
        return True
    else:
        return False


def parallel_generate_actions(args):
    us, vs, image_size, pointcloud, pull_step, delta_y_range, bb_margin, matrix_world_to_camera, move_distance_low, move_distance_high, cloth_mask, task = args

    # choosing a pick location
    lb_u, lb_v, ub_u, ub_v = int(np.min(us)), int(np.min(vs)), int(np.max(us)), int(np.max(vs))
    u = np.random.randint(max(lb_u - bb_margin, 0), min(ub_u + bb_margin, image_size[1]))
    v = np.random.randint(max(lb_v - bb_margin, 0), min(ub_v + bb_margin, image_size[0]))
    target_pos = get_target_pos(pointcloud, u, v, image_size, matrix_world_to_camera, cloth_mask)

    # second stage: choose a random (x, y, z) direction, move towards that direction to determine the pick point
    while True:
        move_direction = np.random.rand(3) - 0.5
        if task == 'flatten':
            move_direction[1] = np.random.uniform(delta_y_range[0], delta_y_range[1])
        else: # for fold, just generate horizontal move
            move_direction[1] = 0

        move_direction = move_direction / np.linalg.norm(move_direction)
        move_distance = np.random.uniform(move_distance_low, move_distance_high)
        delta_move = move_distance / pull_step * move_direction

        after_pos = target_pos + move_distance * move_direction
        if pos_in_image(after_pos, matrix_world_to_camera, image_size):
            break

    return delta_move, target_pos, after_pos