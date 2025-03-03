import copy
import os
import pdb
import time

import numpy as np
import os.path as osp
import torch
import pyflex
# from VCD.models_graph_res import Dynamics
from mesh_gnn.mesh_dyn_dataset import ClothDynDataset
from mesh_gnn.vc_edge import VCConnection
from mesh_gnn.mesh_dyn import MeshDynamics
from softgym.envs.any_cloth_flatten import AnyClothFlattenEnv
from visualization.plot import plot_mesh_face

dyn = None
vcd_edge = None
dataset = None
vcd_dyn = None


def init_worker(dyn_args, dyn_cpu, ve_args=None,
                ve_state=None, gpu_id=0, env=None):
    global dyn
    global vcd_edge
    global dataset
    dyn_cpu.to(gpu_id)
    dyn = dyn_cpu
    dyn.device = torch.device(f'cuda:{gpu_id}')
    print(f'copy a model to gpu {gpu_id}')
    if dataset is None:
        dataset = ClothDynDataset(dyn_args, [dyn_args.train_mode],  'train', env)


def init_worker_hybrid(dyn_args, dyn_cpu, vcd_dyn_cpu, ve_args=None,
                       ve_state=None, gpu_id=0, env=None):
    global dyn
    global vcd_edge
    global dataset
    global vcd_dyn

    dyn_cpu.to(gpu_id)
    dyn = dyn_cpu
    vcd_dyn_cpu.to(gpu_id)
    vcd_dyn = vcd_dyn_cpu
    if ve_args[0] is not None:
        edge_model_args, edge_path = ve_args
        vcd_edge = VCConnection(edge_model_args, env=env)
        vcd_edge.load_model(edge_path)
        vcd_edge.to(gpu_id)
        print('build vcd edge')
    dyn.device = torch.device(f'cuda:{gpu_id}')
    vcd_dyn.device = torch.device(f'cuda:{gpu_id}')
    print(f'copy a model to gpu {gpu_id}')
    if dataset is None:
        dataset = ClothDynDataset([dyn_args.train_mode], dyn_args.log_dir, dyn_args, 'train', env)


def set_worker_gpu(cuda_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)


def rollout_worker(inputs):
    """
    args need to contain the following contents:
        model_input_data: current point cloud, velocity history, picked point, picker position, etc
        actions: rollout actions
        reward_model: reward function
        cuda_idx (optional): default 0
        robot_exp (optional): default False

    return a dict:
        final_ret: final reward of the rollout
        model_positions: model predicted point cloud positions
        shape_positions:
        info:
        mesh_edges: predicted mesh edge
        time_cost: time cost for different parts of the rollout function
    TODO: Do everything on GPU
    """
    global dyn
    global dataset
    global vcd_edge
    global vcd_dyn

    results = []
    for input in inputs:
        model_input_data = input['model_input_data']
        if vcd_dyn is None or model_input_data.get('dyn_mode', 'MEDOR') == 'MEDOR':
            cur_dyn = dyn
            # print('Roll out with MEDOR')
        else:
            cur_dyn = vcd_dyn
            # print('Roll out with vcd')
        actions = input['actions']  # NOTE: sequence of actions to rollout
        reward_model = input['reward_model']
        m_name = input['m_name']
        planning_horizon = len(actions)
        cuda_idx = input.get('cuda_idx', 0)
        robot_exp = input.get('robot_exp', False)
        model_idx = input.get('model_idx', 0)
        input['reward_mode'] = input.get('reward_mode', 'full')
        task = input['task']
        # print('wuwuwu', dyn)
        cur_dyn.set_mode('eval')
        pred_time_interval = cur_dyn.args.pred_time_interval
        dt = cur_dyn.args.dt

        # if self.vcd_edge is not None:
        #     self.vcd_edge.to(self.device)
        # self.vcd_edge.to(self.device )

        particle_pos = model_input_data['pointcloud']
        velocity_his = model_input_data['vel_his']
        picker_pos = model_input_data['picker_position']
        picked_particles = model_input_data['picked_points']
        observable_particle_index = model_input_data['mapped_particle_indices']
        rest_dist = model_input_data.get('rest_dist', None)
        mesh_edges = model_input_data.get('mesh_edges', None)
        model_canon_pos = model_input_data.get('model_canon_pos', None)
        assert rest_dist is None  # The rest_dist will be computed from the initial_particle_pos?

        # record model predicted point cloud positions
        model_positions = np.zeros((planning_horizon, len(particle_pos), 3))
        shape_positions = np.zeros((planning_horizon, 2, 3))
        initial_particle_pos = particle_pos.copy()
        # print('vcd_edge is working ', vcd_edge)
        # get the mesh edge prediction using the mesh edge model
        if (vcd_edge is not None and mesh_edges is None) or m_name == 'vsbl':
            print("inferring mesh_edge in rollout function")
            model_input_data['cuda_idx'] = cuda_idx
            mesh_edges = vcd_edge.infer_mesh_edges(model_input_data)

        # for ablation that uses first-time step collision edges as the mesh edges
        final_ret = 0
        gt_rewards = []

        # some time statistics
        graph_prepare_time = []
        model_foward_time = []
        update_graph_time = []
        put_data_on_gpu_time = []

        # # Find particle whose height is above a certain threshold
        # if robot_exp:
        #     filtered_initial_particle = particle_pos[dyn.args.filter_idx].copy()

        for t in range(planning_horizon):
            action = actions[t]  # use the passed in action

            # set all velocities to zero
            if cur_dyn.args.__dict__.get('zero_vel_his', False):
                velocity_his = np.zeros_like(velocity_his)

            beg = time.time()
            data = {
                'particles': particle_pos,
                'vel_his': velocity_his,
                'picker_position': picker_pos,
                'action': action,
                'picked_points': picked_particles,
                'mapped_particle_indices': observable_particle_index,
                'mesh_edges': mesh_edges,
                'rest_dist': rest_dist,
                'initial_particle_pos': initial_particle_pos
            }

            data_dict = dataset.build_graph(data, input_type=m_name)



            node_attr, neighbors, edge_attr, picked_particles, picked_status = data_dict['node_attr'], \
                                                                               data_dict['neighbors'], \
                                                                               data_dict['edge_attr'], \
                                                                               data_dict['picked_particles'], \
                                                                               data_dict['picked_status']

            graph_prepare_time.append(time.time() - beg)

            model_positions[t] = particle_pos
            shape_positions[t] = picker_pos

            picked_particles = [int(x) for x in picked_particles]

            beg = time.time()
            inputs = {'x': node_attr.to(cur_dyn.device),
                      'edge_attr': edge_attr.to(cur_dyn.device),
                      'edge_index': neighbors.to(cur_dyn.device),
                      'x_batch': torch.zeros(node_attr.size(0), dtype=torch.long, device=cur_dyn.device),
                      'u': torch.zeros([1, cur_dyn.args.global_size], device=cur_dyn.device)}

            put_data_on_gpu_time.append(time.time() - beg)

            # obtain model predictions
            beg = time.time()
            with torch.no_grad():
                pred = cur_dyn(inputs)
                pred_accel = pred['accel']
            pred_accel = pred_accel.cpu().numpy()
            model_foward_time.append(time.time() - beg)

            # update graph
            beg = time.time()
            particle_pos, velocity_his, picker_pos = MeshDynamics.update_graph(pred_accel, particle_pos, velocity_his,
                                                                             picked_status, picked_particles,
                                                                             pred_time_interval, dt)
            update_graph_time.append(time.time() - beg)

            # get reward of the new position
            reward_pos = particle_pos

            # heuristic reward

            reward = reward_model(reward_pos) if reward_model is not None else 0

            if t == planning_horizon - 1:
                # print(model_canon_pos.mean())
                if task == 'flatten':
                    final_ret = reward
                    canon_tgt = None
                elif task == 'canon':
                    final_ret, canon_tgt = AnyClothFlattenEnv.compute_canon_dis_ambiguity_agnostic(
                        model_canon_pos.copy(), reward_pos,
                        find_rigid=False)
                    final_ret = -final_ret
                elif task == 'canon_rigid':
                    final_ret, canon_tgt = AnyClothFlattenEnv.compute_canon_dis_ambiguity_agnostic(
                        model_canon_pos.copy(), reward_pos,
                        find_rigid=True)
                    final_ret = -final_ret
                else:
                    raise NotImplementedError()
            info = {}

        time_cost = [np.sum(graph_prepare_time), np.sum(put_data_on_gpu_time), np.sum(model_foward_time),
                     np.sum(update_graph_time)]
        results.append(dict(
            final_ret=final_ret,
            model_positions=model_positions,
            shape_positions=shape_positions,
            canon_tgt=canon_tgt,
            info=info,
            mesh_edges=mesh_edges,
            time_cost=time_cost,
        ))
    return results


cem_env = None


def rollout_sim(args):
    env_class, env_kwargs, curr_configs, initial_state, batch_target_pos, batch_actions, extra, env, record = args
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["EGL_GPU"] = str(cuda_id)
    # print(cuda_id)
    global cem_env
    # pdb.set_trace()
    if cem_env is None:
        if env is not None:
            cem_env = env
        else:
            env_kwargs['render'] = False
            env_kwargs['observation_mode'] = 'point_cloud'
            cem_env = env_class(**env_kwargs)

    if batch_actions is None:  # Just for creating env
        return
    results = []
    canon_pos = extra.get('canon_pos', None)
    init_pos = extra.get('init_pos', None)
    # mean_len = np.linalg.norm(
    #     curr_config['v'][curr_config['stretch_e'][:, 0]] - curr_config['v'][curr_config['stretch_e'][:, 1]])
    # plot_mesh_face(curr_config['v'], curr_config['f']).show()
    # print('mean_len ', mean_len)
    if record:
        cem_env.start_record()
    for j, (target_pos, actions) in enumerate(zip(batch_target_pos, batch_actions)):
        # print('nmd')
        curr_config = curr_configs[min(len(curr_configs), j)]
        ds_id = extra.get('ds_id', np.ones(curr_config['v'].shape[0], dtype=np.bool))
        cem_env.reset(config=curr_config, canon_pos=canon_pos, init_pos=init_pos)
        cem_env.set_state(initial_state)

        # cem_env.start_record()
        shape_states = pyflex.get_shape_states().reshape((-1, 14))
        # shape_states[1, :3] = -1
        # shape_states[1, 3:6] = -1

        shape_states[0, :3] = target_pos
        shape_states[0, 3:6] = target_pos

        pyflex.set_shape_states(shape_states)

        pyflex.step()
        # time.sleep(0.05)
        planning_horizon = len(actions)
        pos_sample = pyflex.get_positions().reshape(-1, 4)
        model_positions = np.zeros((planning_horizon, len(pos_sample), 3))
        shape_positions = np.zeros((planning_horizon, 2, 3))

        info = None
        for i in range(len(actions)):
            is_nan = np.isnan(pyflex.get_positions()).any()
            # if is_nan:
            #     print('fucked ', i)
            #     pdb.set_trace()
            _, _, _, info = cem_env.step(actions[i], get_info=(i == len(actions) - 1))
            # time.sleep(0.05)
            model_positions[i] = pyflex.get_positions().reshape(-1, 4)[:, :3]
            shape_positions[i] = pyflex.get_shape_states().reshape(-1, 14)[:, :3]

        if extra['task'] == 'flatten':
            final_ret = info['coverage']
            canon_tgt = None
        elif extra['task'] == 'canon':
            final_ret = info['normalized_canon_improvement']
            canon_tgt = info['canon_tgt'][ds_id]
        elif extra['task'] == 'canon_rigid':
            final_ret = info['normalized_canon_rigid_improvement']
            canon_tgt = info['canon_rigid_tgt'][ds_id]

        results.append({
            'final_ret': final_ret,
            'canon_tgt': canon_tgt,
            'model_positions': model_positions[:, ds_id],
            'shape_positions': shape_positions,
            'info': {}
        })

    if record:
        results[-1]['vid'] = cem_env.end_record()

    return results


def free_drop(model_input_data, env=None, mode='sim'):
    actions = np.zeros((100, 8))
    start_poses = np.zeros(3) - 10
    curr_state = env.get_state()
    curr_config = env.get_current_config()
    ori_canon_pos, ori_init_pos, init_covered_area = env.canon_poses.copy(), env.init_pos.copy(), env.init_covered_area
    ori_state, ori_config = copy.deepcopy(curr_state), copy.deepcopy(curr_config)
    num_particles = model_input_data['pointcloud'].shape[0]
    particle_pos = np.concatenate([model_input_data['pointcloud'],
                                   np.ones((num_particles, 1)) * curr_state['particle_pos'][3]],
                                  axis=1)

    # update config and state to predicted particle
    curr_state.update({'particle_pos': particle_pos,
                       'particle_vel': np.zeros((num_particles, 3)),
                       })
    curr_config.update({
        'v': model_input_data['pointcloud'],
        'f': model_input_data['model_face'],
        'stretch_e': model_input_data['mesh_edges'],
        'bend_e': np.array([]),
        'shear_e': np.array([]),
        'stiff': [1, 0, 0],
        'gravity': -9.8 * 10,  # super gravity
        'radius': 0.005
    })
    extra = {'task': 'flatten',
             'canon_pos': model_input_data['pointcloud'][None],
             }
    # Dirty fix. Otherwise step will raise error when computing canon dis
    # Simulate the dropping effect by simulator and roll back to original state
    result = rollout_sim((None, None, [curr_config], curr_state, np.expand_dims(start_poses, 0),
                          np.expand_dims(actions, 0), extra, env, False))
    env.reset(config=ori_config,
              initial_state=ori_state,
              canon_pos=ori_canon_pos,
              init_pos=ori_init_pos,
              init_covered_area=init_covered_area)
    flatten_pos_model = result[0]['model_positions'][-1]
    return flatten_pos_model

