import copy
import json
import multiprocessing as mp
import os
import os.path as osp
import pdb
import pickle
import random
import time
from multiprocessing.pool import Pool
import numpy as np
import pyflex
import wandb
from torch_geometric.data import Batch

import torch
import torchvision
from omegaconf import OmegaConf
from functools import partial

from mesh_gnn.mesh_dyn import MeshDynamics
from garmentnets.networks.conv_implicit_wnf import ConvImplicitWNFPipeline
from garmentnets.networks.hrnet_nocs import HRNet2NOCS
from garmentnets.networks.pointnet2_nocs import PointNet2NOCS
import plotly.express as px

from softgym.registered_env import SOFTGYM_ENVS
from softgym.registered_env import env_arg_dict
from softgym.utils.visualization import save_numpy_as_gif

from utils.camera_utils import get_matrix_world_to_camera, get_visible
from utils.async_utils import init_io_worker, async_vis_io
from utils.diff_render_utils import get_visibility_by_rendering
from utils.geometry_utils import get_world_coords
from utils.misc_utils import transform_info, draw_planned_actions, voxelize_pointcloud
from utils.pyflex_utils import coverage_reward
from utils.data_utils import update_config, find_best_checkpoint, read_h5_dict, process_any_cloth
from chester import logger
from planning.rs_planner import RandomShootingUVPickandPlacePlanner


render_env = None


class VArgs(object):
    def __init__(self, vv):
        for key, val in vv.items():
            setattr(self, key, val)


def vv_to_args(vv):
    args = VArgs(vv)
    return args


def set_picker_pos(pos):
    shape_states = pyflex.get_shape_states().reshape((-1, 14))
    shape_states[1, :3] = -1
    shape_states[1, 3:6] = -1
    shape_states[0, :3] = pos
    shape_states[0, 3:6] = pos

    pyflex.set_shape_states(shape_states)
    pyflex.step()


def prepare_policy(env):
    print("preparing policy! ", flush=True)

    # move one of the picker to be under ground
    shape_states = pyflex.get_shape_states().reshape(-1, 14)
    shape_states[1, :3] = -1
    shape_states[1, 3:6] = -1

    # move another picker to be above the cloth
    pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
    pp = np.random.randint(len(pos))
    shape_states[0, :3] = pos[pp] + [0., 0.06, 0.]
    shape_states[0, 3:6] = pos[pp] + [0., 0.06, 0.]
    pyflex.set_shape_states(shape_states.flatten())


def cem_make_gif(all_frames, save_dir, save_name):
    # Convert to T x index x C x H x W for pytorch
    all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
    grid_imgs = [
        torchvision.utils.make_grid(torch.from_numpy(frame), nrow=5).permute(1, 2, 0).data.cpu().numpy()
        for
        frame in all_frames]
    save_numpy_as_gif(np.array(grid_imgs), osp.join(save_dir, save_name))


def run_task(vv, log_dir, exp_name):
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    mp.set_start_method('spawn', force=True)
    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    print("in run_cem_partial.py run_task: logger.get_dir() is: ", logdir, flush=True)

    # Configure torch
    seed = vv['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda:0')
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    vv['m_name'] = 'full'
    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    # create env
    env_args = copy.deepcopy(env_arg_dict[vv['env_name']])
    env_args['render_mode'] = 'both'
    env_args['observation_mode'] = 'cam_rgb'
    env_args['render'] = True
    env_args['camera_height'] = 720
    env_args['camera_width'] = 720
    env_args['camera_name'] = vv['camera_name']
    env_args['headless'] = True
    env_args['action_repeat'] = vv['action_repeat']
    env_args['picker_radius'] = 0.01
    env_args['picker_threshold'] = 0.00625
    env_args['cached_states_path'] = vv['cached_states_path']
    env_args['num_variations'] = vv.get('num_variations', 40)
    env_args['cloth_type'] = vv['cloth_type']
    env_args['particle_radius'] = 0.005
    env_args['cloth3d_dir'] = "dataset/cloth3d"
    env_args['ambiguity_agnostic'] = vv.get('ambiguity_agnostic', False)
    env = SOFTGYM_ENVS[vv['env_name']](**env_args)
    # pdb.set_trace()
    render_env_kwargs = copy.deepcopy(env_args)
    render_env_kwargs['render_mode'] = 'particle'
    render_env_kwargs['particle_radius'] = 0.01
    env_args['load_flat'] = False

    finetune_cfg = {'opt_mesh_density': vv['opt_mesh_density'],
                    'opt_mesh_init': vv['opt_mesh_init'],
                    'opt_iter_total': vv['opt_iter_total'],
                    'chamfer_mode': 'scipy', 'chamfer3d_w': vv['chamfer3d_w'],
                    'laplacian_w': vv['laplacian_w'], 'normal_w': vv['normal_w'],
                    'edge_w': vv['edge_w'], 'rest_edge_len': 0.,
                    'depth_w': vv['depth_w'], 'silhouette_w': vv['silhouette_w'],
                    'obs_consist_w': vv['obs_consist_w'], 'consist_iter': vv['consist_iter'],
                    'table_w': vv['table_w'],
                    'lr': vv['opt_lr'], 'opt_model': vv['opt_model']}
    finetune_cfg = OmegaConf.create(finetune_cfg)

    print(os.getcwd())
    model_vv_dir = osp.dirname(vv["full_dyn_path"])
    model_vv = json.load(open(osp.join(model_vv_dir, 'variant.json')))
    # model_vv = json.load(open(osp.join(vv['resume1'], 'variant.json')))
    if model_vv.get('particle_radius') is None:
        model_vv['particle_radius'] = 0.00625
    if model_vv.get('partial_observable') is None:
        model_vv['partial_observable'] = True
    if model_vv.get("normalize") is None:
        model_vv['normalize'] = False

    model_vv['reward_model'] = model_vv.get('reward_model', False)
    model_vv['fix_collision_edge'] = model_vv.get('fix_collision_edge', vv['fix_collision_edge'])
    model_vv['use_collision_as_mesh_edge'] = model_vv.get('use_collision_as_mesh_edge',
                                                          vv['use_collision_as_mesh_edge'])
    model_vv['add_occluded_reward'] = model_vv.get('add_occluded_reward', vv['add_occluded_reward'])
    model_vv['debug'] = model_vv.get('debug', vv['debug'])
    model_vv['resume1'] = vv['resume1']
    model_vv['full_dyn_path'] = vv['full_dyn_path']
    model_vv['resume_epoch'] = vv['model_epoch']
    model_vv['train_mode'] = model_vv.get('train_mode', 'vsbl')
    model_vv['use_reward'] = model_vv.get('use_reward', False)
    model_vv['copy'] = False
    model_vv['resume2'] = None
    model_vv['finite_diff'] = True
    model_vv['use_wandb'] = False
    model_vv['action_repeat'] = 1
    model_vv['eval'] = 1
    model_vv['n_epoch'] = vv['model_epoch'] + 1
    model_vv['load_optim'] = False
    model_vv['onpolicy'] = False
    model_vv['cuda_idx'] = "cpu"
    # model_vv['cuda_idx'] = -1
    model_vv['offline_amount'] = 500
    model_vv['onpolicy'] = False
    # model_vv['nstep_eval_rollout'] = 20
    model_vv['proc_layer'] = model_vv.get('proc_layer', 10)
    model_vv['rm_rep_col'] = model_vv.get('rm_rep_col', False)
    model_vv['no_edge_type'] = model_vv.get('no_edge_type', False)
    # model_vv['dataf'] = './datasets/combine-2000-12-16-and-0316/'
    model_vv['dataf'] = './dataset/uniform_c1000_s47_e10_2500'
    model_vv['use_cache'] = False
    model_vv['data_worker'] = model_vv.get('data_worker', 10)
    # vv['pred_time_interval'] = model_vv['pred_time_interval']
    model_vv['pred_time_interval'] = vv['pred_time_interval']
    print(model_vv)
    args = vv_to_args(model_vv)
    device = torch.device('cuda:0')

    reward_func = partial(coverage_reward,
                          cloth_particle_radius=render_env_kwargs['particle_radius'],
                          downsample_scale=model_vv['down_sample_scale'])
    camera_pos, camera_angle = env.get_camera_params()
    matrix_world_to_camera = get_matrix_world_to_camera(cam_angle=camera_angle, cam_pos=camera_pos)

    if vv['pos_mode'] == 'flow':
        vv['model_path'] = vv['model_path'][vv['cloth_type']]
        cfg = OmegaConf.load(vv['model_path'] + '/config.yaml')
        cfg = update_config(cfg, vv)
        pred_cfg = OmegaConf.load('garmentnets/config/predict_default.yaml')
        cfg['prediction'] = pred_cfg.prediction
        if vv['sample_mode'] == 'best_opt':
            cfg['canon_infer'] = 'all'
        batch_size = 1
        print(cfg.canon_checkpoint)
        if cfg.input_type == 'pc':
            pointnet2_model = PointNet2NOCS.load_from_checkpoint(
                find_best_checkpoint(cfg.canon_checkpoint))
        else:
            pointnet2_model = HRNet2NOCS.load_from_checkpoint(
                find_best_checkpoint(cfg.canon_checkpoint))
        pointnet2_params = dict(pointnet2_model.hparams)
        pipeline_model = ConvImplicitWNFPipeline(cfg,
                                                 pointnet2_params=pointnet2_params,
                                                 batch_size=batch_size, **cfg.conv_implicit_model)
        pipeline_model.pointnet2_nocs = pointnet2_model
        pipeline_model.batch_size = batch_size
        model_path = find_best_checkpoint(vv['model_path'])

        pipeline_state_dict = torch.load(model_path)['state_dict']
        pipeline_model.load_state_dict(pipeline_state_dict, strict=False)
        pipeline_model = pipeline_model.cuda()
        pipeline_model.eval()
        if vv['use_wandb']:
            wandb.init(
                project="Occluded cloth",
                name=exp_name,
                group=vv['exp_prefix']
            )
            wandb.config.update(cfg, allow_val_change=True)

    # gdloader = GarmentnetsDataloader(vv['cloth_type'])
    # vcdynamics = VCDynamics(args, env=env)
    vcdynamics = MeshDynamics(args, env=env)

    rs_policy = RandomShootingUVPickandPlacePlanner(vv['cem_num_pick'],
                                                    vv['pull_step'],
                                                    vv['wait_step'],
                                                    dynamics=vcdynamics,
                                                    reward_model=reward_func,
                                                    num_worker=vv['num_worker'],
                                                    move_distance_range=vv['move_distance_range'],
                                                    gpu_num=vv['gpu_num'],
                                                    delta_y_range=vv['delta_y_range'],
                                                    image_size=(env.camera_height, env.camera_width),
                                                    matrix_world_to_camera=matrix_world_to_camera,
                                                    task=vv['task'],
                                                    sample_vis=vv['sample_vis'],
                                                    pos_mode=vv['pos_mode'],
                                                    env=env,
                                                    )

    print("cem policy built done")
    parallel = None
    io_pool = Pool(1, initializer=init_io_worker,
                   initargs=(SOFTGYM_ENVS[vv['env_name']], render_env_kwargs))
    initial_states, action_trajs, configs, all_infos = [], [], [], []
    overall_cem_planning_time = []
    mesh_time = []
    all_normalized_performance = []
    # if vv['debug']:
    #     vv['test_episodes'] = [0]
    #     vv['pick_and_place_num'] = 1
    #     vv['cem_num_pick'] = 2
    for episode_idx in vv['test_episodes']:
        # print(episode_idx)
        if episode_idx > env.num_variations:
            break
        # setup environment, ensure the same initial configuration
        env.reset(config_id=episode_idx)

        # move one picker below the ground, set another picker randomly to a picked point / above the cloth
        prepare_policy(env)

        config = env.get_current_config()
        config_id = env.current_config_id
        # prepare environment and do downsample

        initial_state = env.get_state()
        initial_states.append(initial_state)
        configs.append(config)

        ret = 0
        action_traj = []
        infos = []
        frames = []

        gt_positions = []
        gt_canon_particle_poses = []
        gt_shape_positions = []
        model_pred_particle_poses = []
        model_canon_particle_poses = []
        edges_all = []
        edges_gt = []

        real_pick_num = 0
        cem_planning_time = []

        flex_states = [env.get_state()]
        start_poses = []
        after_poses = []
        obs = env.get_image(env.camera_width, env.camera_height)
        obses = [obs]

        # info for this trajectory
        cloth_id = config['cloth_id']

        ds_path = os.path.join('dataset', vv['cached_states_path'][:-4])
        info3d_path = os.path.join(ds_path, 'data', f'{episode_idx:05d}_3d.h5')
        ds_info_path = os.path.join("dataset/cloth3d",
                                    "nocs",  vv["cloth_type"],
                                    f'{cloth_id:04d}_info.h5')
        ds_data = read_h5_dict(ds_info_path)
        gt_ds_id = ds_data['downsample_id']
        gt_ds_mesh = ds_data['mesh_edges'].T
        gt_face = ds_data['triangles']
        scene_params = [0.025, -1, -1, config_id]
        for pick_try_idx in range(vv['pick_and_place_num']):

            # todo use  get_all_obs
            rgbd = env.get_rgbd(show_picker=False)
            rgb = rgbd[:, :, :3]
            depth = rgbd[:, :, 3]
            unflattened_depth = depth.copy()

            positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
            if vv['pos_mode'] == 'flow':
                pipeline_model.eval()

                mesh_t1 = time.time()
                input_dict = process_any_cloth(rgb,
                                               depth,
                                               matrix_world_to_camera,
                                               input_type=cfg.input_type,
                                               coords=positions,
                                               cloth_id=cloth_id,
                                               info_path=info3d_path,
                                               real_world=False)
                batch_input = Batch.from_data_list([input_dict],
                                                   follow_batch=['cloth_tri', 'cloth_nocs_verts'])

                batch_input = batch_input.to(device=device)
                results_list = pipeline_model.predict_mesh(batch_input,
                                                           voxel_size=0.025,
                                                           finetune_cfg=finetune_cfg,
                                                           parallel=parallel,
                                                           env=env,
                                                           make_gif=False,
                                                           # get_flat_canon_pose= "canon" in vv['task'],
                                                           get_flat_canon_pose=True,
                                                           )
                results = None
                min_loss = 1e10
                if vv['sample_mode'] == 'best_pred':
                    loss_name = 'loss_init'
                else:
                    loss_name = 'loss_end'
                for r in results_list:
                    if r[loss_name] < min_loss:
                        results = r
                        min_loss = r[loss_name]

                # pipeline_model.eval_metrics(batch_input, results, episode_idx, pick_try_idx, log_dir=log_dir)
                model_pos, model_mesh = results['warp_field_ds'], results['mesh_edges_ds'].T
                model_face = results['faces_ds']
                model_canon_pos = results['flat_verts_ds']
                dense_pos = results['warp_field']

                if cfg.finetune:  # 'opt_warp_field_ds' in results:
                    print('use finetuned ')
                    model_pos = results['opt_warp_field_ds']
                    dense_pos = results['opt_warp_field']
                # model_vis = get_visible(camera_param, model_pos, depth)
                model_vis, _ = get_visibility_by_rendering(torch.tensor(dense_pos).cuda(),
                                                           torch.tensor(results['faces']).cuda().long())
                model_vis = model_vis[results['downsample_id']]
                # from visualization.plot import seg_3d_figure
                # f = seg_3d_figure(model_pos, labels=model_vis.numpy())
                # print(torch.sum(model_vis==1), model_vis.shape)
                # f.show()
                # f = plot_pointcloud(results['warp_field'])
                # f.show()
                # plt.imshow(depth.cpu().numpy()[0])
                # plt.show()
                print("creating predicted mesh takes ", time.time() - mesh_t1)
                mesh_time.append(time.time() - mesh_t1)
            elif vv['pos_mode'] == 'pc':
                world_coordinates = get_world_coords(rgb, depth, env=env)
                depth = depth.flatten()
                world_coords = world_coordinates[:, :, :3].reshape((-1, 3))
                positions = world_coords[depth > 0].astype(np.float32)
                model_pos = voxelize_pointcloud(positions, vv['pc_voxel_size'])
                observable_particle_indices = np.zeros(len(positions), dtype=np.int32)
                model_vis = np.ones(model_pos.shape[0])
                model_mesh = np.array([[0], [1]])
                model_face = None
                # NOTE: VCD use gt canon pose for planning
                model_canon_pos = env.canon_poses.copy()
            else:
                # Use ground-truth downsampled mesh for rollout
                model_pos, model_mesh, model_face = positions[gt_ds_id], gt_ds_mesh, gt_face
                model_vis = get_visible(camera_param, positions, depth).squeeze()
                model_canon_pos = env.canon_poses.copy()

                model_vis = model_vis[gt_ds_id]
                model_canon_pos = [x[[gt_ds_id]] for x in model_canon_pos]


            print('Predicted mesh contains {} nodes  {} edges'.format(model_pos.shape[0],
                                                                      model_mesh.shape[1]))

            edges_all.append(model_mesh)
            edges_gt.append(gt_ds_mesh)

            sample_pos = model_pos

            vel_history = np.zeros((model_pos.shape[0], args.n_his * 3), dtype=np.float32)

            picker_position = env.action_tool._get_pos()[0]
            picked_points = [-1, -1]
            # data = [positions, vel_history, picker_position, env.action_space.sample(), picked_points, scene_params, observable_particle_indices]
            data = {
                'sample_pos': sample_pos,
                'model_pos': model_pos,
                'model_vis': model_vis,
                'vel_his': vel_history,
                'picker_position': picker_position,
                'action': env.action_space.sample(),
                'picked_points': picked_points,
                'mesh_edges': model_mesh,  # 2 x n
                'model_face': model_face,
                'scene_params': scene_params,
                'mapped_particle_indices': None,
                'ds_id': gt_ds_id,
                'model_canon_pos': model_canon_pos,
                'init_pos': env.init_pos
            }
            # pdb.set_trace()
            # for k,v in data.items():
            #     if isinstance(v, list):
            #         print(k, len(v))
            #     else:
            #         print(k, v.shape)
            # pdb.set_trace()
            beg = time.time()
            env_info = {
                'env_class': SOFTGYM_ENVS[vv['env_name']],
                'env_args': env_args
            }
            planning_results = rs_policy.get_action(data,
                                                    vcdynamics.args,
                                                    gpu_id=0,
                                                    depth=unflattened_depth,
                                                    m_name=vv['m_name'],
                                                    )

            # pdb.set_trace()
            action_sequence = planning_results['action_seq']
            model_pred_particle_pos = planning_results['model_predict_particle_positions']
            model_canon_tgt = planning_results['model_canon_tgt']

            cem_info = planning_results['ret_info']
            cur_plan_time = time.time() - beg
            cem_planning_time.append(cur_plan_time)
            overall_cem_planning_time.append(cur_plan_time)
            print(
                "config {} pick idx {} cem get action cost time: {}".format(config_id, pick_try_idx,
                                                                            cur_plan_time),
                flush=True)

            # first set picker to target pos
            start_pos = cem_info['start_pos']
            after_pos = cem_info['after_pos']
            start_poses.append(start_pos)
            after_poses.append(after_pos)

            model_pred_particle_poses.append(model_pred_particle_pos)
            model_canon_particle_poses.append(model_canon_tgt)
            # predicted_edges_all.append(predicted_edges)

            # set to pick location directly
            set_picker_pos(start_pos)

            if vv.get('pred_time_interval', 1) >= 2 and vv.get('slow_move', False):
                action_sequence = np.zeros((50 + 30, 8))
                action_sequence[:50, 3] = 1  # first 50 steps pick the cloth
                action_sequence[:50, :3] = (after_pos - start_pos) / 50  # delta move

            gt_positions.append(np.zeros((len(action_sequence), len(gt_ds_id), 3)))
            gt_shape_positions.append(np.zeros((len(action_sequence), 2, 3)))
            ds = vcdynamics.datasets['train']

            for t_idx, ac in enumerate(action_sequence[:-1]):

                picker_position = env.action_tool._get_pos()[0]
                obs, reward, done, info = env.step(ac, record_continuous_video=True, img_size=360)

                imgs = info['flex_env_recorded_frames']
                info['planning_time'] = cur_plan_time
                frames.extend(imgs)
                info.pop("flex_env_recorded_frames")

                ret += reward
                action_traj.append(ac)

                gt_positions[pick_try_idx][t_idx] = pyflex.get_positions().reshape(-1, 4)[gt_ds_id, :3]
                shape_pos = pyflex.get_shape_states().reshape(-1, 14)
                for k in range(2):
                    gt_shape_positions[pick_try_idx][t_idx][k] = shape_pos[k][:3]

            real_pick_num += 1

            if vv['task'] == 'canon':
                gt_canon_particle_poses.append(info['canon_tgt'][gt_ds_id])
            elif vv['task'] == 'canon_rigid':
                gt_canon_particle_poses.append(info['canon_rigid_tgt'][gt_ds_id])
            else:
                gt_canon_particle_poses.append(None)

            if 'canon_tgt' in info:
                info.pop('canon_tgt')
                info.pop('canon_rigid_tgt')
            infos.append(info)
            obs = env.get_image(env.camera_width, env.camera_height)
            obses.append(obs)
            flex_states.append(env.get_state())

            metric_map = {'flatten': 'normalized_coverage_improvement',
                          'canon': 'normalized_canon_improvement',
                          'canon_rigid': 'normalized_canon_rigid_improvement'}
            metric = metric_map[vv['task']]
            print("Iter {} {} is {}".format(pick_try_idx, metric, info[metric]))
            if info[metric] > 0.95:
                break

        scores_gt = [info[metric] for info in infos]

        # dump the data for drawing the planning actions & draw the planning actions
        draw_data = [episode_idx, flex_states, start_poses, after_poses, obses, config]
        # draw the planned actions
        draw_planned_actions(episode_idx, obses, start_poses, after_poses, matrix_world_to_camera,
                             log_dir)
        with open(osp.join(log_dir, '{}_draw_planned_traj.pkl'.format(episode_idx)), 'wb') as f:
            pickle.dump(draw_data, f)
        interval = 1 if vv['pick_and_place_num'] < 100 else 10
        for pick_try_idx in range(0, real_pick_num, interval):
            if vv['pred_time_interval'] >= 2 and vv['slow_move']:
                # in this case the real rollout is longer, have to subsample it
                factor = 80 / len(model_pred_particle_poses[pick_try_idx])
                subsampled_gt_pos = []
                subsampled_shape_pos = []
                max_idx = 80
                for t in range(len(model_pred_particle_poses[pick_try_idx])):
                    subsampled_gt_pos.append(
                        gt_positions[pick_try_idx][min(int(t * factor), max_idx - 1)])
                    subsampled_shape_pos.append(
                        gt_shape_positions[pick_try_idx][min(max_idx - 1, int(t * factor))])
            else:
                subsampled_gt_pos = gt_positions[pick_try_idx]
                subsampled_shape_pos = gt_shape_positions[pick_try_idx]

            kwargs = {
                'particle_pos_model': model_pred_particle_poses[pick_try_idx],
                'particle_pos_gt': subsampled_gt_pos,
                'shape_pos': subsampled_shape_pos,
                'sample_idx': range(model_pred_particle_poses[pick_try_idx].shape[1]),
                'edges_model': edges_all[pick_try_idx],
                'edges_gt': edges_gt[pick_try_idx],
                'goal_model': model_canon_particle_poses[pick_try_idx],
                'goal_gt': gt_canon_particle_poses[pick_try_idx],
                'score': scores_gt[pick_try_idx],
                'gt_ds_id': gt_ds_id,
                'logdir': logdir,
                "episode_idx": episode_idx,
                'pick_try_idx': pick_try_idx,
                'draw_goal_flow': True
            }
            # Doing costly IO asynchronously
            # f.apply_async(async_vis_io, kwds=kwargs)
            io_pool.apply(async_vis_io, kwds=kwargs)

        normalized_performance_traj = [info['normalized_coverage'] for info in infos]
        with open(osp.join(log_dir, 'normalized_performance_traj_{}.pkl'.format(episode_idx)),
                  'wb') as f:
            pickle.dump(normalized_performance_traj, f)

        all_normalized_performance.append(normalized_performance_traj)

        transformed_info = transform_info([infos])
        with open(osp.join(log_dir, 'transformed_info_traj_{}.pkl'.format(episode_idx)), 'wb') as f:
            pickle.dump(transformed_info, f)

        for info_name in transformed_info:
            logger.record_tabular(info_name, transformed_info[info_name][0, -1])

        logger.record_tabular('average_planning_time', np.mean(cem_planning_time))
        logger.dump_tabular()

        cem_make_gif([frames], logger.get_dir(),
                     vv['env_name'] + '{}.gif'.format(episode_idx))

        action_trajs.append(action_traj)
        all_infos.append(infos)
    # rs_policy.pool.join()
    io_pool.close()
    io_pool.join()
    print('\n ---------------------------------------------')
    print("cem plan one action average time: ", np.mean(overall_cem_planning_time), flush=True)
    print('mesh prediction average time', np.mean(mesh_time))
    # print('Avg canoncal pose error', np.mean(chamfer_canon_to_gts))
    with open(osp.join(log_dir, 'all_normalized_performance.pkl'), 'wb') as f:
        pickle.dump(all_normalized_performance, f)

    traj_dict = {
        'initial_states': initial_states,
        'action_trajs': action_trajs,
        'configs': configs,
        'cem_planning_time': overall_cem_planning_time
    }
    with open(osp.join(log_dir, 'cem_traj.pkl'), 'wb') as f:
        pickle.dump(traj_dict, f)
