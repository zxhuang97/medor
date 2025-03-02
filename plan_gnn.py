import copy
import json
import multiprocessing as mp
import os
import os.path as osp
import pickle
import time
from multiprocessing.pool import Pool
import numpy as np
import pyflex
import wandb
from torch_geometric.data import Batch
import argparse
import pdb
import torch
import torchvision
from omegaconf import OmegaConf
from functools import partial
import plotly.express as px

from mesh_gnn.mesh_dyn import MeshDynamics
from garmentnets.networks.conv_implicit_wnf import ConvImplicitWNFPipeline
from garmentnets.networks.hrnet_nocs import HRNet2NOCS
from garmentnets.networks.pointnet2_nocs import PointNet2NOCS

from softgym.registered_env import SOFTGYM_ENVS
from softgym.utils.visualization import save_numpy_as_gif

from utils.camera_utils import get_matrix_world_to_camera, get_visible
from utils.async_utils import init_io_worker, async_vis_io
from utils.diff_render_utils import get_visibility_by_rendering
from utils.geometry_utils import get_world_coords
from utils.misc_utils import set_resource, transform_info, draw_planned_actions, voxelize_pointcloud
from utils.pyflex_utils import coverage_reward
from utils.data_utils import update_config, find_best_checkpoint, read_h5_dict, process_any_cloth
from planning.rs_planner import RandomShootingUVPickandPlacePlanner

from chester import logger
import pytorch_lightning.utilities.seed as seed_utils

render_env = None


def set_picker_pos(pos):
    shape_states = pyflex.get_shape_states().reshape((-1, 14))
    shape_states[1, :3] = -1
    shape_states[1, 3:6] = -1
    shape_states[0, :3] = pos
    shape_states[0, 3:6] = pos

    pyflex.set_shape_states(shape_states)
    pyflex.step()


def prepare_policy(env):
    print("preparing policy! ")

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


def create_env(vv):
    # create env
    env_args = OmegaConf.load("configs/env_cfg.yaml")
    env_args['camera_name'] = vv['camera_name']
    env_args['cached_states_path'] = vv['cached_states_path']
    env_args['cloth_type'] = vv['cloth_type']
    env_args['ambiguity_agnostic'] = vv.get('ambiguity_agnostic', False)

    env = SOFTGYM_ENVS[vv['env_name']](**env_args)

    render_env_kwargs = copy.deepcopy(env_args)
    render_env_kwargs['render_mode'] = 'particle'
    render_env_kwargs['particle_radius'] = vv['radius_r']

    io_pool = Pool(1, initializer=init_io_worker,
                   initargs=(SOFTGYM_ENVS[vv['env_name']], render_env_kwargs))

    return env, io_pool


def create_dynamics(vv, env):
    model_vv_dir = osp.dirname(vv["full_dyn_path"])
    model_vv = json.load(open(osp.join(model_vv_dir, 'variant.json')))
    model_vv['full_dyn_path'] = vv['full_dyn_path']
    model_vv['eval'] = True
    model_vv['load_optim'] = False
    vv['pred_time_interval'] = model_vv['pred_time_interval']
    dyn_args = OmegaConf.create(model_vv)
    return MeshDynamics(dyn_args, env=env)


def create_medor_model(vv):
    vv['model_path'] = vv['model_path'][vv['cloth_type']]
    cfg = OmegaConf.load(vv['model_path'] + '/config.yaml')
    cfg = update_config(cfg, vv)
    pred_cfg = OmegaConf.load('garmentnets/config/predict_default.yaml')
    cfg['prediction'] = pred_cfg.prediction
    batch_size = 1
    if cfg.input_type == 'pc':
        pointnet2_model = PointNet2NOCS.load_from_checkpoint(
            find_best_checkpoint(cfg.canon_checkpoint))
    else:
        pointnet2_model = HRNet2NOCS.load_from_checkpoint(
            find_best_checkpoint(cfg.canon_checkpoint))
    pointnet2_params = dict(pointnet2_model.hparams)
    cloth_meta = OmegaConf.load("configs/cloth_nocs_aabb.yaml")
    cloth_nocs_aabb = np.array(cloth_meta[cfg.cloth_type], dtype=np.float32)
    pipeline_model = ConvImplicitWNFPipeline(cfg,
                                             pointnet2_params=pointnet2_params,
                                             batch_size=batch_size,
                                             cloth_nocs_aabb=cloth_nocs_aabb,
                                             **cfg.conv_implicit_model)
    pipeline_model.pointnet2_nocs = pointnet2_model
    pipeline_model.batch_size = batch_size
    model_path = find_best_checkpoint(vv['model_path'])

    pipeline_state_dict = torch.load(model_path)['state_dict']
    pipeline_model.load_state_dict(pipeline_state_dict, strict=False)
    pipeline_model = pipeline_model.cuda()
    pipeline_model.eval()
    return pipeline_model

def compute_cloth_mesh_for_planning(vv, medor_model, matrix_world_to_camera, finetune_cfg, env,
                                    gt_ds_id, gt_ds_edges, gt_face):
    rgbd = env.get_rgbd(show_picker=False)
    rgb = rgbd[:, :, :3]
    depth = rgbd[:, :, 3]
    # unflattened_depth = depth.copy()
    medor_cfg = medor_model.cfg
    positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
    if vv['pos_mode'] == 'medor':
        input_dict = process_any_cloth(rgb,
                                        depth,
                                        matrix_world_to_camera,
                                        input_type=medor_cfg.input_type,
                                        coords=positions,
                                        cloth_id=env.get_current_config()['cloth_id'],
                                        cloth_type=vv['cloth_type'])
        batch_input = Batch.from_data_list([input_dict],
                                            follow_batch=['cloth_tri', 'cloth_nocs_verts'])

        batch_input = batch_input.to(device=medor_model.device)
        results = medor_model.predict_mesh(batch_input,
                                            finetune_cfg=finetune_cfg,
                                            env=env,
                                            get_flat_canon_pose=True,
                                            )[0]

        verts, edges = results['warp_field_ds'], results['mesh_edges_ds'].T
        faces = results['faces_ds']
        canon_verts = results['flat_verts_ds']
        dense_verts = results['warp_field']

        if vv["tt_finetune"]:
            print('use finetuned ')
            verts = results['opt_warp_field_ds']
            dense_verts = results['opt_warp_field']
        verts_vis, _ = get_visibility_by_rendering(torch.tensor(dense_verts).cuda(),
                                                    torch.tensor(results['faces']).cuda().long())
        verts_vis = verts_vis[results['downsample_id']]

    elif vv['pos_mode'] == 'gt':
        verts, edges, faces = positions[gt_ds_id], gt_ds_edges, gt_face
        verts_vis = get_visible(matrix_world_to_camera, positions, depth).squeeze()
        canon_verts = env.canon_poses.copy()

        verts_vis = verts_vis[gt_ds_id]
        canon_verts = [x[[gt_ds_id]] for x in canon_verts]

    print('Predicted mesh contains {} nodes  {} edges'.format(verts.shape[0], edges.shape[1]))

    picker_position, picked_points = env.action_tool._get_pos()[0], [-1, -1]
    data = {
        'verts': verts,
        'verts_vis': verts_vis,
        'picker_position': picker_position,
        'picked_points': picked_points,
        'mesh_edges': edges,
        'model_face': faces,
        'mapped_particle_indices': None,
        'ds_id': gt_ds_id,
        'model_canon_pos': canon_verts,
        'init_pos': env.init_pos,
        "depth": depth
    }

    return data


def run_task(plan_cfg, log_dir, exp_name):
    set_resource()
    mp.set_start_method('spawn', force=True)
    # Configure logger 
    logger.configure(dir=log_dir, exp_name=exp_name)
    os.makedirs(log_dir, exist_ok=True)
    seed = plan_cfg.seed

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(OmegaConf.to_container(plan_cfg), f, indent=2, sort_keys=True)

    env, io_pool = create_env(plan_cfg)

    finetune_cfg = {'opt_mesh_density': plan_cfg.opt_mesh_density,
                    'opt_mesh_init': plan_cfg.opt_mesh_init,
                    'opt_iter_total': plan_cfg.opt_iter_total,
                    'chamfer_mode': 'scipy', 'chamfer3d_w': plan_cfg.chamfer3d_w,
                    'laplacian_w': plan_cfg.laplacian_w, 'normal_w': plan_cfg.normal_w,
                    'edge_w': plan_cfg.edge_w, 'rest_edge_len': 0.,
                    'depth_w': plan_cfg.depth_w, 'silhouette_w': plan_cfg.silhouette_w,
                    'obs_consist_w': plan_cfg.obs_consist_w, 'consist_iter': plan_cfg.consist_iter,
                    'table_w': plan_cfg.table_w,
                    'lr': plan_cfg.opt_lr, 'opt_model': plan_cfg.opt_model}
    finetune_cfg = OmegaConf.create(finetune_cfg)

    print(os.getcwd())

    mesh_dyn = create_dynamics(plan_cfg, env)
    dyn_args = mesh_dyn.args

    reward_func = partial(coverage_reward,
                          cloth_particle_radius=plan_cfg.radius_r)
    camera_pos, camera_angle = env.get_camera_params()
    matrix_world_to_camera = get_matrix_world_to_camera(cam_pos=camera_pos, cam_angle=camera_angle)

    if plan_cfg.pos_mode == 'medor':
        medor_model = create_medor_model(plan_cfg)
        medor_cfg = medor_model.cfg
        if plan_cfg.use_wandb:
            wandb.init(
                project="Occluded cloth",
                name=exp_name,
                group=plan_cfg.exp_prefix
            )
            wandb.config.update(medor_cfg, allow_val_change=True)

    rs_policy = RandomShootingUVPickandPlacePlanner(plan_cfg.cem_num_pick,
                                                    plan_cfg.pull_step,
                                                    plan_cfg.wait_step,
                                                    dynamics=mesh_dyn,
                                                    reward_model=reward_func,
                                                    num_worker=plan_cfg.num_worker,
                                                    move_distance_range=plan_cfg.move_distance_range,
                                                    gpu_num=plan_cfg.gpu_num,
                                                    delta_y_range=plan_cfg.delta_y_range,
                                                    image_size=(env.camera_height, env.camera_width),
                                                    matrix_world_to_camera=matrix_world_to_camera,
                                                    task=plan_cfg.task,
                                                    pick_vis_only=plan_cfg.pick_vis_only,
                                                    env=env,
                                                    )

    print("cem policy built done")
    metric_map = {'flatten': 'normalized_coverage_improvement',
                    'canon': 'normalized_canon_improvement',
                    'canon_rigid': 'normalized_canon_rigid_improvement'}
    metric_name = metric_map[plan_cfg.task]
    # K x dict , K x N x 16, K , K x N x 80 
    initial_states, action_trajs, configs, all_infos = [], [], [], []
    overall_cem_planning_time = []
    mesh_recon_time = []
    all_normalized_performance = []

    seed_utils.seed_everything(seed)
    for episode_idx in plan_cfg.test_episodes:
        # setup environment, ensure the same initial configuration
        env.reset(config_id=episode_idx)

        # move one picker below the ground, set another picker randomly to a picked point / above the cloth
        prepare_policy(env)
        config = env.get_current_config()
        # prepare environment and do downsample
        initial_states.append(env.get_state())
        configs.append(config)

        action_traj = []
        infos = []
        frames = []

        gt_positions, gt_canon_positions, gt_shape_positions, model_pred_particle_poses, model_canon_positions= [], [], [], [], []
        pred_edges, gt_edges = [], []
        actual_pick_num = 0
        cem_planning_time = []

        flex_states = [env.get_state()]
        start_poses, after_poses = [], []
        obs = env.get_image(env.camera_width, env.camera_height)
        obses = [obs]

        # info for this trajectory
        ds_info_path = os.path.join("dataset/cloth3d/nocs", plan_cfg.cloth_type, f'{config["cloth_id"]:04d}_info.h5')
        ds_data = read_h5_dict(ds_info_path)
        gt_ds_id = ds_data['downsample_id']
        gt_ds_edges = ds_data['mesh_edges'].T
        gt_face = ds_data['triangles']
        for pick_try_idx in range(plan_cfg.pick_and_place_num):

            # compute cloth mesh for planning
            data = compute_cloth_mesh_for_planning(plan_cfg, medor_model, matrix_world_to_camera, finetune_cfg, env,
                                                          gt_ds_id, gt_ds_edges, gt_face)
            data["vel_his"] = np.zeros((data["verts"].shape[0], dyn_args.n_his * 3), dtype=np.float32)
            pred_edges.append(data["mesh_edges"])
            gt_edges.append(gt_ds_edges)


            # planning using mesh_dyn to get action sequence
            beg = time.time()
            action_seq, model_pred_positions, canon_tgt, cem_info = rs_policy.get_action(data, mesh_dyn.args, m_name=plan_cfg.m_name)

            cur_plan_time = time.time() - beg
            cem_planning_time.append(cur_plan_time)
            print("Episode {} pick idx {} cem get action cost time: {}".format(episode_idx, pick_try_idx, cur_plan_time))

            # first set picker to target pos
            start_pos, after_pos = cem_info['start_pos'], cem_info['after_pos']
            start_poses.append(start_pos), after_poses.append(after_pos)
            set_picker_pos(start_pos)

            model_pred_particle_poses.append(model_pred_positions)
            model_canon_positions.append(canon_tgt)

            if plan_cfg.pred_time_interval >= 2:
                action_seq = np.zeros((50 + 30, 8))
                action_seq[:50, 3] = 1  # first 50 steps pick the cloth
                action_seq[:50, :3] = (after_pos - start_pos) / 50  # delta move

            gt_positions.append(np.zeros((len(action_seq), len(gt_ds_id), 3)))
            gt_shape_positions.append(np.zeros((len(action_seq), 2, 3)))

            for t_idx, ac in enumerate(action_seq):
                obs, reward, done, info = env.step(ac, record_continuous_video=True, img_size=360)

                info['planning_time'] = cur_plan_time
                frames.extend(info['flex_env_recorded_frames'])
                action_traj.append(ac)
                gt_positions[pick_try_idx][t_idx] = pyflex.get_positions().reshape(-1, 4)[gt_ds_id, :3]
                shape_pos = pyflex.get_shape_states().reshape(-1, 14)
                for k in range(2):
                    gt_shape_positions[pick_try_idx][t_idx][k] = shape_pos[k][:3]

            actual_pick_num += 1
            info.pop("flex_env_recorded_frames")
            info.pop('canon_tgt', None)
            info.pop('canon_rigid_tgt', None)
            if plan_cfg.task == 'canon':
                gt_canon_positions.append(canon_tgt[gt_ds_id])
            elif plan_cfg.task == 'canon_rigid':
                gt_canon_positions.append(info['canon_rigid_tgt'][gt_ds_id])
            else:
                gt_canon_positions.append(None)


            infos.append(info)
            obs = env.get_image(env.camera_width, env.camera_height)
            obses.append(obs)
            flex_states.append(env.get_state())


            print("Iter {} {} is {}".format(pick_try_idx, metric_name, info[metric_name]))
            if info[metric_name] > 0.95:
                break
        
        overall_cem_planning_time.extend(cem_planning_time)
        normalized_performance_traj = [info[metric_name] for info in infos]

        # dump the data for drawing the planning actions & draw the planning actions
        draw_data = [episode_idx, flex_states, start_poses, after_poses, obses, config]
        # draw the planned actions
        draw_planned_actions(episode_idx, obses, start_poses, after_poses, matrix_world_to_camera, log_dir)
        with open(osp.join(log_dir, '{}_draw_planned_traj.pkl'.format(episode_idx)), 'wb') as f:
            pickle.dump(draw_data, f)

        for pick_try_idx in range(actual_pick_num):
            # Subsample ground truth trajectory to match model prediction length
            model_pred_len = len(model_pred_particle_poses[pick_try_idx])
            max_idx = 80
            factor = max_idx / model_pred_len
            
            subsampled_gt_pos = [
                gt_positions[pick_try_idx][min(int(t * factor), max_idx - 1)]
                for t in range(model_pred_len)
            ]
            subsampled_shape_pos = [
                gt_shape_positions[pick_try_idx][min(int(t * factor), max_idx - 1)]
                for t in range(model_pred_len)
            ]

            # Prepare visualization arguments
            vis_kwargs = {
                'particle_pos_model': model_pred_particle_poses[pick_try_idx],
                'particle_pos_gt': subsampled_gt_pos,
                'shape_pos': subsampled_shape_pos,
                'sample_idx': range(model_pred_particle_poses[pick_try_idx].shape[1]),
                'edges_model': pred_edges[pick_try_idx],
                'edges_gt': gt_edges[pick_try_idx], 
                'goal_model': model_canon_positions[pick_try_idx],
                'goal_gt': gt_canon_positions[pick_try_idx],
                'score': normalized_performance_traj[pick_try_idx],
                'gt_ds_id': gt_ds_id,
                'logdir': log_dir,
                'episode_idx': episode_idx,
                'pick_try_idx': pick_try_idx,
                'draw_goal_flow': True
            }

            # Asynchronously generate visualization
            io_pool.apply_async(async_vis_io, kwds=vis_kwargs)

        with open(osp.join(log_dir, 'normalized_performance_traj_{}.pkl'.format(episode_idx)), 'wb') as f:
            pickle.dump(normalized_performance_traj, f)

        all_normalized_performance.append(normalized_performance_traj)
        final_perf = [x[-1] for x in all_normalized_performance]
        print('Avg final performance', np.mean(final_perf))
        print('Median final performance', np.median(final_perf))
        transformed_info = transform_info([infos])
        with open(osp.join(log_dir, 'transformed_info_traj_{}.pkl'.format(episode_idx)), 'wb') as f:
            pickle.dump(transformed_info, f)

        for info_name in transformed_info:
            logger.record_tabular(info_name, transformed_info[info_name][0, -1])

        logger.record_tabular('average_planning_time', np.mean(cem_planning_time))
        logger.dump_tabular()

        cem_make_gif([frames], logger.get_dir(),
                     plan_cfg.env_name + '{}.gif'.format(episode_idx))

        action_trajs.append(action_traj)
        all_infos.append(infos)
    # rs_policy.pool.join()
    io_pool.close()
    io_pool.join()
    print('\n ---------------------------------------------')
    print("cem plan one action average time: ", np.mean(overall_cem_planning_time))
    print('mesh prediction average time', np.mean(mesh_recon_time))
    final_perf = [x[-1] for x in all_normalized_performance]
    print('Avg final performance', np.mean(final_perf))
    print('Median final performance', np.median(final_perf))
    
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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--cloth_type', type=str, default='Trousers')
    arg_parser.add_argument('--exp_name', type=str, default='')
    args = arg_parser.parse_args()
    plan_cfg = OmegaConf.load("configs/plan.yaml")
    plan_cfg.cloth_type = args.cloth_type
    if args.exp_name == '':
        exp_name = args.cloth_type
    else:
        exp_name = args.exp_name
    log_dir = os.path.join("data/plan", exp_name)

    plan_cfg.mc_thres = {'Trousers': 0.1, 'Skirt': 0.05, 'Tshirt': 0.1, 'Dress': 0.1, 'Jumpsuit': 0.1}[plan_cfg.cloth_type]
    plan_cfg.cached_states_path = f"{plan_cfg.cloth_type}_hard_v4.pkl"
    plan_cfg.test_episodes = np.arange(40).tolist()
    run_task(plan_cfg, log_dir, exp_name)
