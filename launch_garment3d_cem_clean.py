import os.path
import socket
import time

import click
import numpy as np
import torch

from chester.run_exp import run_experiment_lite, VariantGenerator
from garmentnets.run_cem_gm_clean import run_task

import datetime
import dateutil.tz


def get_cem_num_pick(uv_sample_method, collect_data):
    if not collect_data:
        if uv_sample_method == 'bounding_box':
            return 500
            # return 100
        elif uv_sample_method == 'bias_towards_edge':
            return 100
        elif uv_sample_method == 'cloth_mask':
            return 500
    else:
        return 100


def get_stage_2(pull_up, pred_time_interval):  # TODO: check proper stage length
    if not pull_up:
        if pred_time_interval == 1:
            return 50  # 35
        elif pred_time_interval == 5:
            return 10
    else:
        return 50  # 65


def get_stage_3(pull_up, pred_time_interval):
    if not pull_up:
        if pred_time_interval == 1:
            return 30  # 20
        elif pred_time_interval == 5:
            return 6
    else:
        return 30  # 35


def get_move_range(task, env_name):
    return (0.05, 0.2)


def get_testset(ctype, version):
    test_set = {
        'Trousers': f'Trousers_hard_v{version}.pkl',
        'Tshirt': f'Tshirt_hard_v{version}.pkl',
        'Dress': f'Dress_hard_v{version}.pkl',
        'Skirt': f'Skirt_hard_v{version}.pkl',
        'Jumpsuit': f'Jumpsuit_hard_v{version}.pkl'
    }
    return test_set[ctype]


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    base_path = '/home/zixuanh/occlusion_reasoning/data/train/'

    vg = VariantGenerator()
    ############################
    #### PLANNING PARAMETERS ###
    ############################
    vg.add('debug', [False])
    # vg.add('algorithm', ['CEM'])

    vg.add('camera_name', ['top_down_camera'])
    vg.add("tt_finetune", [True])
    vg.add('normalize', [False])
    vg.add('use_reward_model', [False])

    vg.add('action_repeat', [1])
    vg.add('mode', ['pick-and-place-uv'])
    vg.add('uv_sample_method', ['bounding_box'])
    vg.add('gpu_num', [1])
    vg.add('pointcloud', [True])
    vg.add('pc_voxel_size', [0.0216])
    vg.add('add_occluded_particles', [False])
    vg.add('add_occluded_reward', [False])
    # vg.add('random', [False])  # if random sample an action in CEM.
    vg.add('seed', [100])

    vg.add('delta_y', [None])  # 0.1
    vg.add('delta_y_range', [(0, 0.5)])  # 0.1
    vg.add('move_distance', [None])  # 0.15
    vg.add('move_distance_range', lambda task, env_name: [get_move_range(task, env_name)])
    # vg.add('cem_stage_1_step', [20])  # not used anymore
    vg.add('pull_step',
           lambda use_pull_up, pred_time_interval: [get_stage_2(use_pull_up, pred_time_interval)])  # 60
    vg.add('wait_step',
           lambda use_pull_up, pred_time_interval: [get_stage_3(use_pull_up, pred_time_interval)])  # 40

    # vg.add('collect_data', [False])

    # vg.add('fix_collision_edge', [False])
    # ablation: trained with edge GNN, but no edge GNN at test time
    # vg.add('use_collision_as_mesh_edge', [False])
    # vg.add('policy', ['heuristic_pull_edge'])
    vg.add('policy', ['vcd'])
    # vg.add('task', ['fold'])
    vg.add('env_name', ['AnyClothFlatten'])
    # vg.add('reward_mode', ['coverage'])  # canon
    vg.add('use_pred_rwd', [False])
    vg.add('is_train', [False])

    vg.add('slow_move', [True])
    vg.add('planning_method', ['random_shooting'])  # "random_shooting", "hierarchical_random_shooting", "cem_iterative"
    vg.add('use_pull_up', [False])  # if first pick up then move horizontally.
    vg.add('collect_data', [False])  # if to collect data during cem rollout
    vg.add('dataf', ['./datasets/0515-collect-cem-plan-no-pull-up-data/'])  # if first pick up then move horizontally.

    # vg.add('model', ['/home/zixuanhu/occlusion_reasoning/data/train/vcd-5step/final'])

    vg.add('sample_space', ['mesh'])  # pointcloud mesh
    vg.add('sample_vis', [True])
    vg.add('pred_mode', ['oneshot'])
    # Pipeline model for each category
    vg.add('model_path', [
        {
            # 'Trousers': '1112-conv2-pipeline/1112-conv2-pipeline',
            # 'Trousers': '0105-im-Trousers_v2/None',
            # 'Tshirt': '1218-pipeline-Tshirt_v2/1218-pipeline-Tshirt_v2_1',
            # 'Dress': '1218-pipeline-Dress_v2/1218-pipeline-Dress_v2_1',
            # 'Skirt': '1222-pipeline-Skirt_v2/1222-pipeline-Skirt_v2_1',
            # 'Skirt': '0110_im_flow_norelu_nocs_init_Skirt_v2/0110_im_flow_norelu_nocs_init_Skirt_v2_1',
            # 'Skirt': '0110_im_flow_mix_chamfer_norelu_nocs_init_Skirt_v2/0110_im_flow_mix_chamfer_norelu_nocs_init_Skirt_v2_1',
            # 'Skirt': '0110_im_mixture_norelu_Skirt_v2/0110_im_mixture_norelu_Skirt_v2_mean_pair_w_0.1_2',
            # 'Skirt': '0110_im_mixture_norelu_Skirt_v2/0110_im_mixture_norelu_Skirt_v2_chamfer_whole_w_1_1',
            # 'Skirt': '0110_im_mixture_norelu_Skirt_v2/0110_im_mixture_norelu_Skirt_v2_mean_pair_w_0_1',
            # 'Skirt': '0110_flow_normal_noise_Skirt_v2/None','
            # 'Skirt': '0112-noise-flow-mixSkirt_v2/chamferK5',
            # 'Skirt': '0112-noise-flowSkirt_v2/0112-noise-flowSkirt_v2_loss_type_huber_2',
            # nosie flow:
            # 'Trousers': '0112-noise-flowTrousers_v2/None',
            # 'Tshirt': '0114-noise-flowTshirt_v2/0114-noise-flowTshirt_v2_1',
            # 'Dress': '0114-noise-flowDress_v2/0114-noise-flowDress_v2_1',
            # 'Skirt': '0112-noise-flowSkirt_v2/0112-noise-flowSkirt_v2_loss_type_l2_1',
            # 'Jumpsuit': '0118_noise_flow/Jumpsuit_v2',
            # noise base
            # 'Trousers': '0116_direct_base/Trousers_v2',
            # 'Tshirt': '0116_direct_base/Tshirt_v2',
            # 'Dress': '0116_direct_base/Dress_v2',
            # 'Skirt': '0116_direct_base/Skirt_v2',
            # 'Jumpsuit': '0116_direct_base/Jumpsuit_v2',
            # pointcloud
            # 'Trousers': '0125_pointnet_pipeline/Trousers_v2',
            # 'Tshirt': '0125_pointnet_pipeline/Tshirt_v2',
            # 'Dress': '0125_pointnet_pipeline/Dress_v2',
            # 'Skirt': '0125_pointnet_pipeline/Skirt_v2',
            # 'Jumpsuit': '0125_pointnet_pipeline/Jumpsuit_v2',

            # 'Tshirt': '0312_scale_volume_pipeline/Tshirt_v2',
            # 'Tshirt': '0316_norm_canon_pipe/Tshirt_v2',

            # reproduce_v1
            'Trousers': 'data/train/0112-noise-flowTrousers_v2/None',
            # "Trousers": "gn_test_trousers_pipe/trousers_pipe"
            # "Tshirt": "data/release/tshirt_release/pipeline/",
            # 'Tshirt': '0114-noise-flowTshirt_v2/0114-noise-flowTshirt_v2_1',

        }
    ])
    opt_model = False
    vg.add('pred_time_interval', [5])
    # vg.add('resume1', ['data/train/trousers_gnn_old/1119-trousers-5000-full_1'])
    vg.add("full_dyn_path", ["data/train/trousers_gnn_old/1119-trousers-5000-full_1/full_dyn_75.pth"])
    # vg.add("full_dyn_path", ["data/train/good_old_dyn/1119-trousers-5000-full_1/best.pth"])
    # vg.add('model_epoch', [75])

    # vg.add('edge_model_dir', [{
    #     'Trousers': 'data/train/0120-vcd-edge/Trousers',
    #     'Skirt': 'data/train/0120-vcd-edge/Skirt',
    #     'Tshirt': 'data/train/0120-vcd-edge/Tshirt',
    #     'Dress': 'data/train/0120-vcd-edge/Dress',
    #     'Jumpsuit': 'data/train/0120-vcd-edge/Jumpsuit',
    # }])
    #
    # vg.add('edge_model_name', [
    #     {
    #         'Trousers': 'vcd_edge_0.pth',
    #         'Skirt': 'vcd_edge_6.pth',
    #         'Tshirt': 'vcd_edge_1.pth',
    #         'Dress': 'vcd_edge_0.pth',
    #         'Jumpsuit': 'vcd_edge_1.pth'}
    # ])
    vg.add('finetune', [True])
    vg.add('opt_model', [opt_model])
    if opt_model:
        vg.add('chamfer3d_w', [1.])
        vg.add('laplacian_w', [0.])
        vg.add('normal_w', [0.])
        vg.add('edge_w', [0.])
    else:
        vg.add('opt_mesh_density', ['dense'])  # dense, sparse
        vg.add('opt_mesh_init', ['task'])  # task, canon
        vg.add('opt_lr', [1e-3])
        vg.add('opt_iter_total', [100])

        vg.add('chamfer3d_w', [1.])
        vg.add('laplacian_w', [0.0])
        vg.add('normal_w', [0.])
        vg.add('edge_w', [0.0])
        vg.add('depth_w', [0])
        vg.add('silhouette_w', [0])
        vg.add('obs_consist_w', [10.])
        vg.add('consist_iter', [50])
        vg.add('table_w', [10])

    # test_dir = '0118_noise_base'
    # test_dir = '0125_noise_flow_20picks'
    # test_dir = '0125_gt_dyn_test'
    # test_dir = '0127_pointnet_test'
    # test_dir = '0126_noise_flow_noft_20picks'
    # test_dir = '0125_vcd_test'
    # test_dir = '0127_noise_flow_inv'
    test_dir = 'release_'
    # exp_prefix = 'gn_reproduce'
    exp_prefix = 'test'
    vg.add('reward_mode', ['full'])   # full vis
    vg.add('reload_path', [None])
    vg.add('cached_states_path', lambda cloth_type: [get_testset(cloth_type, 4)])
    # vg.add('cloth_type', ['Tshirt'])
    # vg.add('cloth_type', ['Jumpsuit'])
    # vg.add('cloth_type', ['Tshirt', 'Dress'])
    # vg.add('cloth_type', ['Dress', 'Skirt', 'Jumpsuit'])
    vg.add('cloth_type', ['Trousers'])
    vg.add('pick_and_place_num', [10])
    vg.add('ambiguity_agnostic', [True])
    vg.add('task', ["flatten"])  # flatten, canon, canon_rigid
    vg.add('canon_model', ['sim'])  # linear, sim, model
    vg.add('pos_mode', ['flow'])  # flow   gt   pc
    vg.add('mesh_mode', ['mc'])
    vg.add('radius_s', [0.005])  # 0.005 original mesh, 0.025 ds mesh
    vg.add('radius_r', [0.01])  # 0.005 original mesh, 0.0
    vg.add('num_sample', [0])

    def get_mc_thres(cloth_type):
        thres = {'Trousers': 0.1, 'Skirt': 0.05, 'Tshirt': 0.1, 'Dress': 0.1, 'Jumpsuit': 0.1}
        return thres[cloth_type]

    vg.add('mc_thres', lambda cloth_type: [get_mc_thres(cloth_type)])  # 0.1  0.05
    vg.add('sample_mode', ['best_opt'])  # uniform best_pred best_opt best_gt best_chamf
    # vg.add('resume1', ['data/train/0915-newdata1000-full/0915-newdata1000-full_rm_rep_col_1'])
    # vg.add('resume1', [None])
    vg.add('basepath', ['/home/zixuanhu/occlusion_reasoning/'])

    def devide_epi(num, n, start=0):
        s = int(np.ceil(num / n))
        all_epi = [_ + start for _ in range(num)]
        return [all_epi[i:i + s] for i in range(0, len(all_epi), s)]

    test_episodes = devide_epi(40, 4, start=0)

    # test_episodes = [[i for i in range(31, 40)]]
    def get_num_worker(pos_mode):
        if pos_mode == 'flow' and mode not in ['autobot', 'local2']:
            return 4
        return 4

    if not debug:
        vg.add('test_episodes', test_episodes)
        # vg.add('test_episodes', [[_ for _ in range(0, 7)], [_ for _ in range(7, 14)], [_ for _ in range(14, 20)]])

        # pick-and-place
        vg.add('cem_num_pick',
               lambda uv_sample_method, collect_data: [get_cem_num_pick(uv_sample_method, collect_data)])
        vg.add('num_worker', lambda pos_mode: [get_num_worker(pos_mode)])

        # iterative cem
        vg.add('pop_size', [100])
        vg.add('num_iter', [6])
        vg.add('num_elite', [20])

        vg.add('hierarchical_pop_size', [[80, 40]])
        vg.add('hierarchical_num_iter', [[5, 4]])
        vg.add('hierarchical_num_elite', [[20, 10]])
    else:
        vg.add('test_episodes', test_episodes)

        vg.add('cem_num_pick', [5])
        vg.add('pop_size', [20])
        vg.add('num_iter', [2])
        vg.add('num_elite', [10])
        vg.add('num_worker', [1])

        vg.add('hierarchical_pop_size', [[10, 10]])
        vg.add('hierarchical_num_iter', [[1, 1]])
        vg.add('hierarchical_num_elite', [[2, 2]])

    variations = []

    if debug:
        exp_prefix += '_debug'
    vg.add('use_wandb', [False])

    print('Number of configurations: ', len(vg.variants()))
    print("exp_prefix: ", exp_prefix)
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    gpu_num = len(gpus)
    sub_process_popens = []
    variations = variations + vg.variations()
    for idx, vv in enumerate(vg.variants()):
        # dirs_parts = vv['model_path'].split('/')
        # log_dir_base = '/'.join(dirs_parts[:4])
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        if mode == 'local2':
            suffix = now.strftime('%m_%d_%H_%M%S%f')
        else:
            suffix = now.strftime('%m_%d_%H_%M%S')
        # if vv['finetune']:
        #     suffix = 'finetune_' + suffix
        if vv['opt_model']:
            suffix = 'opt_model' + suffix
        if vv['task'] == 'flatten':
            exp_name = "_".join([vv['cloth_type'], vv['task'], exp_prefix, suffix])
        else:
            exp_name = "_".join([vv['cloth_type'], vv['task'], exp_prefix, suffix])
        log_dir = os.path.join(base_path, test_dir, exp_name)
        log_dir = log_dir.replace('train', 'release_plan')
        # vv['model_path'] = {k: base_path + v for k, v in vv['model_path'].items()}
        vv['exp_prefix'] = exp_prefix
        # log_dir += suffix
        while len(sub_process_popens) >= 10:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        if mode in ['seuss', 'autobot', 'autobot2']:
            if idx == 0:
                # compile_script = 'compile.sh'  # For the first experiment, compile the current softgym
                compile_script = None  # For the first experiment, compile the current softgym
                wait_compile = 0
            else:
                compile_script = None
                wait_compile = 0  # Wait 30 seconds for the compilation to finish
        elif mode == 'ec2':
            compile_script = 'compile_1.0.sh'
            wait_compile = None
        else:
            compile_script = wait_compile = None
        if mode.startswith('local2'):
            env_var = {'CUDA_VISIBLE_DEVICES': str(gpus[idx % gpu_num])}
        else:
            env_var = None

        # folder = 'debug' if debug else 'test'
        # run_task(vv, f'data/{folder}/{exp_name}/{exp_name}', exp_name)
        if mode == 'local' or debug:
            folder = 'debug' if debug else 'release_plan'
            run_task(vv, f'data/{folder}/{exp_name}/{exp_name}', exp_name)
            # run_task(vv, log_dir, exp_name)
            break
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            variations=variations,
            exp_name=exp_name,
            log_dir=log_dir,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var,
            use_singularity=True
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
