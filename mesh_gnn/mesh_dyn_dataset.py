import pdb
import numpy as np

import torch
import os

from scipy import spatial
import time
import pyflex
import scipy

import utils.pyflex_utils
from utils.cloth3d.DataReader.IO import readOBJ
from utils.data_utils import read_h5_dict, store_h5_dict

from utils.camera_utils import (
    get_observable_particle_index, get_mapping_from_pointcloud_to_partile_nearest_neighbor,
    project_to_image,
    get_matrix_world_to_camera, get_pointcloud
)
from utils.misc_utils import voxelize_pointcloud, remove_suffix
from garmentnets.common.potpourri3d_util import geodesic_matrix
from softgym.utils.misc import vectorized_range, vectorized_meshgrid, quads2tris
from softgym.utils.visualization import save_numpy_as_gif
# from menpo.shape.mesh.base import TriMesh, PointCloud
# from menpo3d.barycentric import barycentric_coordinates_of_pointcloud
from utils.data_utils import PrivilData
from visualization.plot import plot_pointclouds


class ClothDynDataset(torch.utils.data.Dataset):
    def __init__(self, args, input_types, phase, env=None):
        self.all_trajs = []
        self.args = args
        self.input_types = input_types
        self.n_rollout = self.args.n_rollout

        # self.online_n_rollout = 0
        self.phase = phase
        self.env = env

        self.running_workers = list()

        self.time_step = args.time_step
        self.num_workers = args.num_workers

        self.env_name = args.env_name
        self.dt = args.dt

        ratio = self.args.train_valid_ratio
        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        if self.args.dataf is not None:
            self.data_dir = os.path.join(self.args.dataf, phase)
            self.stat_path = os.path.join(self.args.dataf, 'stat.h5')
            self.data_dirs = [os.path.join(self.data_dir, str(i)) for i in range(self.n_rollout)]
            os.system('mkdir -p ' + self.data_dir)
            # self.data_dirs = sorted(glob.glob(os.path.join(self.data_dir, '*')))[:self.offline_amount]
        else:
            self.data_dir = None
            self.stat_path = None

        self.data_names = ['positions',
                           'picker_position',
                           'action',
                           # 'picked_points',
                           'downsample_id',
                           'observable_idx',
                           'pointcloud',
                           'mesh_edges',
                           'cloth_id',
                           "config_id"
                           ]
        # self.data_names.append('downsample_indices')
        # self.data_names.append('scene_params')

        self.vcd_edge = None
        self.max_dist = -1

    def __len__(self):
        return self.n_rollout * (self.args.time_step - self.args.n_his) - 10

        # return 10 * (self.args.time_step - self.args.n_his) - 10

    def len(self):  # required by torch_geometric.data.dataset
        return self.n_rollout * (self.args.time_step - self.args.n_his) - 10
        # return 10 * (self.args.time_step - self.args.n_his) - 10

    def gt_reward_model(self, pos, cloth_particle_radius=0.00625, downsample_scale=3):
        cloth_particle_radius *= downsample_scale
        pos = np.reshape(pos, [-1, 3])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)
        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = vectorized_range(slotted_x_low, slotted_x_high)
        listy = vectorized_range(slotted_y_low, slotted_y_high)
        listxx, listyy = vectorized_meshgrid(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1

        res = np.sum(grid) * span[0] * span[1]
        return res

    def load_rollout_data(self, idx_rollout, idx_timestep):
        data_cur_path = os.path.join(self.data_dirs[idx_rollout], str(idx_timestep) + '.h5')
        data_cur = read_h5_dict(data_cur_path, self.data_names)
        # data_nxt = read_h5_dict(data_nxt_path, load_names)
        # data_cur = load_data(self.data_dir, idx_rollout, idx_timestep, self.data_names)
        # accumulate action if we need to predict multiple steps
        action = data_cur['action']
        for t in range(1, self.args.pred_time_interval):
            data_cur_path = os.path.join(self.data_dirs[idx_rollout], str(idx_timestep+t) + '.h5')
            t_action = read_h5_dict(data_cur_path, ['action'])['action']
            # TODO: pass it if picker drops in the middle
            action[:3] += t_action[:3]
        data_cur['action'] = action
        return data_cur

    def prepare_transition(self, idx, eval=False):
        """
        Return the raw input for both full cloth and partial point cloud.
        Noise augmentation is only supported when fd_input = True
        Two modes for input and two modes for output:
            self.args.fd_input = True:
                Calculate vel his by 5-step finite differences
            else:
                Retrieve vel from dataset, which is obtained by 1-step finite differences.
            self.args.fd_output = True:
                Calculate vel_nxt by 5-step finite differences
            else:
                Calculate vel_nxt by retrieving one-step vel at 5 timesteps later.

        """
        # # TODO: more concise load names
        # load_names = ['positions',
        #               'picker_position',
        #               'action',
        #               # 'picked_points',
        #               'downsample_id',
        #               'observable_idx',
        #               'pointcloud',
        #               'mesh_edges',
        #               'cloth_id',
        #               'config_id',
        #               ]
        pred_time_interval = self.args.pred_time_interval
        idx_rollout = idx // (self.args.time_step - self.args.n_his)
        idx_timestep = max((self.args.n_his - pred_time_interval) + idx % (self.args.time_step - self.args.n_his), 0)
        data_path = os.path.join(self.data_dirs[idx_rollout], str(idx_timestep) + '.h5')
        data_nxt_path = os.path.join(self.data_dirs[idx_rollout],
                                     str(idx_timestep + pred_time_interval) + '.h5')
        # self.data_names.pop('downsample_id')

        data_cur = read_h5_dict(data_path, self.data_names)
        data_nxt = read_h5_dict(data_nxt_path, self.data_names)
        # data_cur['downsample_id'] = data_cur['downsample_indices']
        # data_cur['config_id'] = int(data_cur['scene_params'][-1])
        # data_nxt['downsample_id'] = data_nxt['downsample_indices']
        # data_nxt['config_id'] = int(data_nxt['scene_params'][-1])
        all_pos_cur = data_cur["positions"]
        downsample_id_cur = data_cur["downsample_id"]

        if 'vsbl' in self.input_types:
            pointcloud = data_cur["pointcloud"].astype(np.float32)
            pointcloud_cur = voxelize_pointcloud(pointcloud, self.args.voxel_size)
            partial_pc_mapped_idx = get_mapping_from_pointcloud_to_partile_nearest_neighbor(
                pointcloud_cur, all_pos_cur)
        else:  # mapping mode is nearest neighbor
            pointcloud_cur = np.zeros((1, 3))
            partial_pc_mapped_idx = np.zeros(1, np.int32)

        # accumulate action if we need to predict multiple steps
        action = data_cur["action"]
        for t in range(1, pred_time_interval):
            tmp_path = os.path.join(self.data_dirs[idx_rollout], str(idx_timestep + t) + '.h5')
            tmp_data = read_h5_dict(tmp_path, ['action'])
            action[:3] += tmp_data['action'][:3]
        full_pos_cur = data_cur["positions"][downsample_id_cur]
        full_pos_nxt = data_nxt["positions"][downsample_id_cur]
        all_pos_list = [full_pos_nxt, full_pos_cur]

        # for n_his = n, we need n+1 velocities(including target), and n+2 position
        seq_len = self.args.n_his + 1
        # shape_pos = [data_cur["shape_positions"]]
        for i in range(pred_time_interval, seq_len * pred_time_interval, pred_time_interval):
            path = os.path.join(self.data_dirs[idx_rollout],
                                str(max(0, idx_timestep - i)) + '.h5')  # max just in case
            data_his = read_h5_dict(path, ['positions'])
            all_pos_list.append(data_his["positions"][downsample_id_cur])
            # shape_pos.append(data_his["shape_positions"])
        all_pos_list.reverse()  # from past to future

        all_vel_list = []
        for i in range(seq_len):
            all_vel_list.append((all_pos_list[i + 1] - all_pos_list[i]) / (self.args.dt * pred_time_interval))

        # Get velocity history, remove target velocity(last one)
        partial_vel_his_list, full_vel_his_list = [], all_vel_list[:-1]
        full_vel_his_list.reverse()  # inverse order, from current to past
        for vel in full_vel_his_list:
            partial_vel_his_list.append(vel[partial_pc_mapped_idx])

        # 3 items for next step
        partial_vel_his = np.concatenate(partial_vel_his_list, axis=1)
        full_vel_his = np.concatenate(full_vel_his_list, axis=1)

        full_vel_cur = all_vel_list[-2]
        full_vel_nxt = all_vel_list[-1]

        full_gt_accel = torch.FloatTensor((full_vel_nxt - full_vel_cur) / (self.args.dt * pred_time_interval))
        partial_gt_accel = full_gt_accel[partial_pc_mapped_idx]

        # todo: do it in data collection
        gt_reward_crt = torch.FloatTensor([self.gt_reward_model(full_pos_cur)])
        gt_reward_nxt = torch.FloatTensor([self.gt_reward_model(full_pos_nxt)])

        data = {
            'particles_vsbl': pointcloud_cur,
            'vel_his_vsbl': partial_vel_his,
            'gt_accel_vsbl': partial_gt_accel,
            'picked_points_vsbl': [-1, -1],

            'particles_full': full_pos_cur,
            'vel_his_full': full_vel_his,
            'gt_accel_full': full_gt_accel,
            'picked_points_full': [-1, -1],

            'gt_reward_crt': gt_reward_crt,
            'gt_reward_nxt': gt_reward_nxt,
            'idx_rollout': idx_rollout,
            'picker_position': data_cur["picker_position"],
            'action': action,
            'mapped_particle_indices': partial_pc_mapped_idx,
            'all_pos_cur': all_pos_cur,
            'downsample_id': downsample_id_cur,
        }

        if self.args.train_mode == 'vsbl':
            data['quad'] = readOBJ(os.path.join('dataset/cloth3d/train',
                                                self.args.cloth_type, f'{data_cur["cloth_id"]:04d}.obj'))[1]
        if 'full' in self.input_types:
            mesh_edges = data_cur['mesh_edges'].T
            # if self.args.__dict__.get('vsbl_full', False):
            #     data['pointcloud_full'] = data['pointcloud_full'][observable_idx_cur]
            #     data['vel_his_full'] = data['vel_his_full'][observable_idx_cur]
            #     data['gt_accel_full'] = data['gt_accel_full'][observable_idx_cur]
            #     sender_valid = np.in1d(mesh_edges[0], observable_idx_cur)
            #     recie_valid = np.in1d(mesh_edges[1], observable_idx_cur)
            #     valid_edges = np.logical_and(sender_valid, recie_valid)
            #     mesh_edges = mesh_edges[:, valid_edges]
            #     full2vsbl_map = np.zeros(len(full_pos_cur))
            #     full2vsbl_map[observable_idx_cur] = np.arange(len(observable_idx_cur))
            #     mesh_edges = full2vsbl_map[mesh_edges]
            data['mesh_edges'] = mesh_edges.astype(np.int)
        else:
            data['mesh_edges'] = data_cur['mesh_edges_pc'].T

        # if self.vcd_edge is not None:
        #     # TODO: support vcd
        #     self.vcd_edge.set_mode('eval')
        #     model_input_data = dict(
        #         scene_params=data_cur[5],
        #         pointcloud=pointcloud,
        #         cuda_idx=-1,
        #     )
        #     mesh_edges = self.vcd_edge.infer_mesh_edges(model_input_data)
        #     data['mesh_edges'] = mesh_edges
        #
        #     if self.args.__dict__.get('use_rest_distance', False):
        #         print("computing rest distance", flush=True)
        #         if self.args.use_cache:
        #             data_init = self._load_data_from_cache(load_names, idx_rollout, 0)
        #         else:
        #             data_path = os.path.join(self.data_dirs[idx_rollout], '0.h5')
        #             data_init = self._load_data_file(load_names, data_path)
        #         pc_pos_init = data_init[0][partial_pc_mapped_idx].astype(np.float32)
        #
        #         # pc_pos_init = self.get_interpolated_pc(bc, pc_tri_idx, particle_pos, tri_mesh).astype(np.float32)
        #         rest_dist = np.linalg.norm(pc_pos_init[mesh_edges[0, :]] - pc_pos_init[mesh_edges[1, :]], axis=-1)
        #         data['rest_dist'] = rest_dist

        if not eval:
            return data
        else:  # TODO: do we need this?
            data['downsample_id'] = data_cur["downsample_id"]
            data['observable_idx'] = data_cur["observable_idx"]
            data['pc_to_mesh_mapping'] = partial_pc_mapped_idx
            data['config_id'] = data_cur['config_id']
            data['cloth_id'] = data_cur['cloth_id']
            return data

    def _find_and_update_picked_point(self, data):
        """
        Directly change the position and velocity of the picked point so that the dynamics model understand the action
        Note: 1. vox_pc, vel_his is modified in-place
        """
        picked_pos = []  # Position of the picked particle
        picked_velocity = []  # Velocity of the picked particle

        action = data['action'].reshape([-1, 4])  # scale to the real action

        particles, picker_pos, velocity_his = data['particles'], data['picker_position'], data['vel_his']

        picked_particles = [-1 for _ in picker_pos]
        pick_flag = action[:, 3] > 0.5
        new_picker_pos = picker_pos.copy()

        for i in range(self.env.action_tool.num_picker):
            new_picker_pos[i, :] = picker_pos[i, :] + action[i, :3]
            if pick_flag[i]:
                if picked_particles[i] == -1:
                    # No particle is currently picked and thus need to select a particle to pick
                    dists = scipy.spatial.distance.cdist(picker_pos[i].reshape((-1, 3)),
                                                         particles[:, :3].reshape((-1, 3)))
                    picked_particles[i] = np.argmin(dists)

                old_pos = particles[picked_particles[i]]
                new_pos = particles[picked_particles[i]] + new_picker_pos[i, :] - picker_pos[i, :]
                new_vel = (new_pos - old_pos) / (self.dt * self.args.pred_time_interval)

                tmp_vel_history = velocity_his[picked_particles[i]][:-3].copy()
                velocity_his[picked_particles[i], 3:] = tmp_vel_history
                velocity_his[picked_particles[i], :3] = new_vel

                particles[picked_particles[i]] = new_pos

                picked_velocity.append(velocity_his[picked_particles[i]])
                picked_pos.append(new_pos)
            else:
                picked_particles[i] = int(-1)
        picked_status = (picked_velocity, picked_pos, new_picker_pos)
        return picked_particles, picked_status

    def _compute_node_attr(self, vox_pc, picked_points, velocity_his):
        # picked particle [0, 1]
        # normal particle [1, 0]
        node_one_hot = np.zeros((len(vox_pc), 2), dtype=np.float32)
        node_one_hot[:, 0] = 1
        for picked in picked_points:
            if picked != -1:
                node_one_hot[picked, 0] = 0
                node_one_hot[picked, 1] = 1
        distance_to_ground = torch.from_numpy(vox_pc[:, 1]).view((-1, 1))
        node_one_hot = torch.from_numpy(node_one_hot)
        node_attr = torch.from_numpy(velocity_his)
        node_attr = torch.cat([node_attr, distance_to_ground, node_one_hot], dim=1)
        return node_attr

    def _compute_edge_attr(self, input_type, data):
        ##### add env specific graph components
        ## Edge attributes:
        # [1, 0] Distance based neighbor
        # [0, 1] Mesh edges
        # Calculate undirected edge list and corresponding relative edge attributes (distance vector + magnitude)
        particles_pos = data['particles']
        mesh_edges = data.get('mesh_edges', None)
        pc_to_mesh_mapping_cur = data['mapped_particle_indices']
        all_pos_cur = data.get('all_pos_cur', None)
        # _, cloth_xdim, cloth_ydim, _ = data['scene_params']
        rest_dist = data.get('rest_dist', None)

        point_tree = scipy.spatial.cKDTree(particles_pos)
        undirected_neighbors = np.array(list(point_tree.query_pairs(self.args.neighbor_radius, p=2))).T

        if len(undirected_neighbors) > 0:
            dist_vec = particles_pos[undirected_neighbors[0, :]] - particles_pos[undirected_neighbors[1, :]]
            dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
            edge_attr = np.concatenate([dist_vec, dist], axis=1)
            edge_attr_reverse = np.concatenate([-dist_vec, dist], axis=1)

            # Generate directed edge list and corresponding edge attributes
            edges = np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1)
            edge_attr = np.concatenate([edge_attr, edge_attr_reverse])
            num_distance_edges = edges.shape[1]
        else:
            num_distance_edges = 0

        if mesh_edges is not None and 'full' in self.input_types:  # Note: input mesh edge is unidirectional
            mesh_edges = np.concatenate([mesh_edges, mesh_edges[::-1]], axis=1)

        if mesh_edges is None:
            tri = quads2tris(data['quad'])
            tri = np.array(tri, dtype=np.int32)
            mapped_edges = pc_to_mesh_mapping_cur[edges.T]
            distinct_id = sorted(list(set(mapped_edges.flatten().tolist())))
            ds_v = all_pos_cur  # [data['downsample_ indices']]
            mat = geodesic_matrix(ds_v, faces=tri,
                                  # vert_idxs=np.arange(ds_v.shape[0]),
                                  vert_idxs=distinct_id,
                                  )
            distinct_map = np.zeros(max(distinct_id) + 1, dtype=np.int)
            distinct_map[distinct_id] = np.arange(len(distinct_id))

            geo_dis = mat[distinct_map[mapped_edges[:, 0]], distinct_map[mapped_edges[:, 1]]]
            gt_edge_connection = geo_dis < 0.035
            mesh_edges = edges[:, gt_edge_connection]
        else:
            mesh_edges = mesh_edges

        mesh_dist_vec = particles_pos[mesh_edges[0, :]] - particles_pos[mesh_edges[1, :]]  # Nx3
        mesh_dist = np.linalg.norm(mesh_dist_vec, axis=1, keepdims=True)
        mesh_edge_attr = np.concatenate([mesh_dist_vec, mesh_dist], axis=1)
        num_mesh_edges = mesh_edges.shape[1]

        if rest_dist is None:
            if data.get('idx_rollout',
                        None) is not None:  # training case, without using an edge model to get the mesh edges
                idx_rollout = data['idx_rollout']

                data_path = os.path.join(self.data_dirs[idx_rollout], '0.h5')
                init_pos = read_h5_dict(data_path, ['positions'])['positions']
                downsample_id = data['downsample_id']
                if input_type == 'vsbl':
                    pc_pos_init = init_pos[downsample_id][data['mapped_particle_indices']].astype(np.float32)
                else:
                    pc_pos_init = init_pos[downsample_id].astype(np.float32)
            else:  # rollout during training
                assert 'initial_particle_pos' in data
                pc_pos_init = data['initial_particle_pos']
            rest_dist = np.linalg.norm(pc_pos_init[mesh_edges[0, :]] - pc_pos_init[mesh_edges[1, :]], axis=-1)

        # rollout during test case, rest_dist should already be computed outwards.
        rest_dist = rest_dist.reshape((-1, 1))
        displacement = mesh_dist.reshape((-1, 1)) - rest_dist
        mesh_edge_attr = np.concatenate([mesh_edge_attr, displacement.reshape(-1, 1)], axis=1)
        if num_distance_edges > 0:
            edge_attr = np.concatenate([edge_attr, np.zeros((edge_attr.shape[0], 1), dtype=np.float32)], axis=1)

        # concatenate all edge attributes
        edge_attr = np.concatenate([edge_attr, mesh_edge_attr],
                                   axis=0) if num_distance_edges > 0 else mesh_edge_attr
        edge_attr, mesh_edges = torch.from_numpy(edge_attr), torch.from_numpy(mesh_edges)

        # Concatenate edge types
        edge_types = np.zeros((num_mesh_edges + num_distance_edges, 2), dtype=np.float32)
        edge_types[:num_distance_edges, 0] = 1.
        edge_types[num_distance_edges:, 1] = 1.
        edge_types = torch.from_numpy(edge_types)
        edge_attr = torch.cat([edge_attr, edge_types], dim=1)

        if num_distance_edges > 0:
            edges = torch.from_numpy(edges)
            edges = torch.cat([edges, mesh_edges], dim=1)
        else:
            edges = mesh_edges

        return edges, edge_attr

    def build_graph(self, data, input_type):
        """
        data: positions, vel_history, picked_points, picked_point_positions, scene_params
        downsample: whether to downsample the graph
        test: if False, we are in the training mode, where we know exactly the picked point and its movement
            if True, we are in the test mode, we have to infer the picked point in the (downsampled graph) and compute
                its movement.

        return:
        node_attr: N x (vel_history x 3)
        edges: 2 x E, the edges
        edge_attr: E x edge_feature_dim
        global_feat: fixed, not used for now
        """

        vox_pc, velocity_his = data['particles'], data['vel_his']
        picked_points, picked_status = self._find_and_update_picked_point(data)  # Return index of the picked point
        node_attr = self._compute_node_attr(vox_pc, picked_points, velocity_his)
        edges, edge_attr = self._compute_edge_attr(input_type, data)

        return {'node_attr': node_attr,
                'neighbors': edges,
                'edge_attr': edge_attr,
                'picked_particles': picked_points,
                'picked_status': picked_status}

    def __getitem__(self, idx):
        all_input = {}
        ori_data = self.prepare_transition(idx, eval=self.phase == 'valid')
        for input_type in self.input_types:
            suffix = '_' + input_type
            data = remove_suffix(ori_data, input_type)
            gt_accel, gt_reward_crt, gt_reward_nxt = data['gt_accel'], data['gt_reward_crt'], data['gt_reward_nxt']
            d = self.build_graph(data, input_type=input_type)
            node_attr, neighbors, edge_attr = d['node_attr'], d['neighbors'], d['edge_attr']
            # print('node ', node_attr.shape[0], 'edge ', neighbors.shape)
            all_input.update({
                'x' + suffix: node_attr,
                'edge_index' + suffix: neighbors,
                'edge_attr' + suffix: edge_attr,
                'gt_accel' + suffix: gt_accel,
            })
        data = PrivilData.from_dict(all_input)
        return data

    def get(self, idx):
        return self.__getitem__(idx)

    def _generate_policy_info(self, matrix_world_to_camera):
        # NOTE: data collection policy

        print("preparing policy! ")
        """ Doing something after env reset but before collecting any data"""
        picker_position = self.env.action_tool.get_picker_pos()

        utils.pyflex_utils.set_picker_pos(picker_position)
        while 1:
            # randomly select a move direction and a move distance
            move_direction = np.random.rand(3) - 0.5
            move_direction[1] = np.random.uniform(0, 0.5)

            policy_info = {}
            policy_info['move_direction'] = move_direction / np.linalg.norm(move_direction)
            policy_info['move_distance'] = np.random.uniform(
                self.args.collect_data_delta_move_min,
                self.args.collect_data_delta_move_max)
            policy_info['move_steps'] = 50
            policy_info['delta_move'] = policy_info['move_distance'] / policy_info['move_steps']
            # return
            end_pos = picker_position[0] + policy_info['move_direction'] * policy_info['move_distance']
            u, v = project_to_image(matrix_world_to_camera, world_coordinate=end_pos[None], height=720,
                                    width=720)
            # actions = np.zeros((self.time_step, 8))
            # actions[:move_step, :3] = self.policy_info['delta_move'] * self.policy_info['move_direction']
            # actions[:move_step, 3] = 1
            # actions[:, 4:] = 0
            # self.policy_info['actions'] = actions
            if 0 < u < self.env.camera_height and 0 < v < self.env.camera_width:
                print("policy info [move direction]: ", policy_info['move_direction'])
                return policy_info

    def _collect_policy(self, timestep, policy_info):
        """ Policy for collecting data"""
        if timestep <= policy_info['move_steps']:
            delta_move = policy_info['delta_move']
            action = np.zeros_like(self.env.action_space.sample())
            action[3] = 1
            action[:3] = delta_move * policy_info['move_direction']
        else:
            action = np.zeros_like(self.env.action_space.sample())
            # self.env.action_tool.set_picker_pos([100, 100, 100]) # Set picker away
        return action

    def get_curr_env_data(self, base_data, matrix_world_to_camera):
        # Env info that does not change within one episode
        position = pyflex.get_positions().reshape(-1, 4)[:, :3]
        picker_position = self.env.action_tool.get_picker_pos()

        # Cloth and picker information
        # Get partially observed particle index
        rgbd = self.env.get_rgbd(show_picker=False)
        rgb, depth = rgbd[:, :, :3], rgbd[:, :, 3]
        pointcloud = get_pointcloud(depth, matrix_world_to_camera)

        observable_idx = get_observable_particle_index(pointcloud, position[base_data['downsample_id']])

        base_data.update({'positions': position.astype(np.float32),
                          'picker_position': picker_position,
                          'observable_idx': observable_idx,
                          'pointcloud': pointcloud.astype(np.float32),
                          'config_id': self.env.current_config_id,
                          })
        if self.args.gen_gif:
            base_data['rgb'], base_data['depth'] = rgb, depth
        return base_data

    def generate_dataset(self):
        """ Write data collection function for each task. Use random actions by default"""
        np.random.seed(0)
        rollout_idx = 0

        env = self.env
        camera_param = env.camera_params[env.camera_name]
        matrix_world_to_camera = get_matrix_world_to_camera(cam_pos=camera_param['pos'],
                                                            cam_angle=camera_param['angle'])

        while rollout_idx < self.n_rollout:
            print("{} / {}".format(rollout_idx, self.n_rollout))
            rollout_dir = os.path.join(self.data_dir, str(rollout_idx))
            os.system('mkdir -p ' + rollout_dir)

            self.env.reset()

            config = env.get_current_config()
            cloth_path = f'{self.args.cloth3d_dir}/nocs/{env.cloth_type}/{config["cloth_id"]:04d}_info.h5'
            cloth_info = read_h5_dict(cloth_path, ["downsample_id", "mesh_edges"])  # M, Nx2
            cloth_info['cloth_id'] = config["cloth_id"]

            curr_data = self.get_curr_env_data(cloth_info, matrix_world_to_camera)
            observable_idx, picker_position = curr_data['observable_idx'], curr_data['picker_position']
            downsample_id = curr_data['downsample_id']

            pp = np.random.randint(len(observable_idx))
            picker_position[0] = curr_data['positions'][downsample_id][observable_idx[pp]] + np.array(
                [0., 0.01, 0.])  # Pick above the particle
            # plot_pointclouds([curr_data['positions'][downsample_id][observable_idx],
            #                     curr_data['pointcloud']]
            #                  ).show()
            # plot_pointclouds([curr_data['positions'][downsample_id],
            #                   # curr_data['pointcloud'],
            #                   picker_position[0:1].reshape(1,3)]).show()
            picker_position[1] = np.array([100, 100, 100])
            utils.pyflex_utils.set_picker_pos(picker_position)
            prev_data = curr_data
            policy_info = self._generate_policy_info(matrix_world_to_camera)


            if self.args.gen_gif:
                frames_rgb, frames_depth = [prev_data['rgb']], [prev_data['depth']]
            print('start rolling out')
            for j in range(1, self.args.time_step):
                action = self._collect_policy(j, policy_info)
                self.env.step(action, get_info=False)
                curr_data = self.get_curr_env_data(cloth_info, matrix_world_to_camera)

                prev_data['action'] = action
                store_h5_dict(os.path.join(rollout_dir, str(j - 1) + '.h5'), prev_data)
                prev_data = curr_data
                if self.args.gen_gif:
                    frames_rgb.append(prev_data['rgb'])
                    frames_depth.append(prev_data['depth'])
            if j < self.args.time_step - 1:
                continue
            if self.args.gen_gif:
                save_numpy_as_gif(np.array(np.array(frames_rgb) * 255).clip(0., 255.),
                                  os.path.join(rollout_dir, 'rgb.gif'))
                save_numpy_as_gif(np.array(frames_depth) * 255., os.path.join(rollout_dir, 'depth.gif'))
            # pdb.set_trace()
            prev_data['action'] = 0
            store_h5_dict(os.path.join(rollout_dir, str(self.args.time_step - 1) + '.h5'), prev_data)
            rollout_idx += 1

        print('Done !!!!!')
