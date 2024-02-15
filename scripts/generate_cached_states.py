import argparse
import glob
import os.path
import pdb

from softgym.registered_env import env_arg_dict
from softgym.utils.visualization import save_numpy_as_gif
import numpy as np
import os.path as osp
import torchvision
import torch
from softgym.registered_env import env_arg_dict
from softgym.registered_env import SOFTGYM_ENVS


def generate_env_state(env_name, num_variations=40, is_headless=True, is_render=True, ds_name="test",
                       cloth_type="Tshirt", eval_flag=False, cloth3d_dir='dataset/cloth3d'):
    env_args = env_arg_dict[env_name]
    env_args['headless'] = is_headless
    env_args['picker_radius'] = 0.01
    env_args['picker_threshold'] = 0.00625
    env_args['camera_name'] = 'top_down_camera'  # 'top_down_camera'  'default_camera
    env_args['camera_height'] = 720
    env_args['camera_width'] = 720
    env_args['num_variations'] = num_variations
    env_args['save_cached_states'] = True
    env_args['collect_mode'] = 'random_drop'  # random_drop    vcd_failure  vcd_decay
    env_args['render_mode'] = 'cloth'  # TODO(zixuan): figure out the difference
    env_args['observation_mode'] = 'cam_rgb'
    env_args['particle_radius'] = 0.005
    env_args['render'] = is_render
    env_args['cloth3d_dir'] = cloth3d_dir
    env_args['load_flat'] = True

    env_args['eval_flag'] = eval_flag
    env_args['tol'] = 0.9
    env_args['use_cached_states'] = True

    ds_name = f"{cloth_type}_{ds_name}"
    # ds_name = "Trousers_2500_v1"
    env_args['cached_states_path'] = ds_name + '.pkl'
    env_args['cloth_type'] = cloth_type
    env = SOFTGYM_ENVS[env_name](**env_args)
    env.close()


if __name__ == '__main__':
    # create a args parser to take in the arguments of generated_env_state
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='AnyClothFlatten')
    parser.add_argument('--num_variations', type=int, default=40)
    parser.add_argument('--is_headless', type=bool, default=True)
    parser.add_argument('--is_render', type=bool, default=True)
    parser.add_argument('--ds_name', type=str, default='test')
    parser.add_argument('--cloth3d_dir', type=str, default='dataset/cloth3d')
    parser.add_argument('--cloth_type', type=str, default='Trousers', help='[Trouser, Tshirt, Jumpsuit, Dress, Skirt, all]')
    parser.add_argument('--eval_flag', type=bool, default=False, help="generate train set or eval set")

    args = parser.parse_args()
    if args.cloth_type == 'all':
        cloth_types = ['Trousers', 'Tshirt', 'Shirt', 'Dress', 'Skirt']
    else:
        cloth_types = [args.cloth_type]
    print("Generate cached states from the following cloth_types: ", cloth_types)
    for cloth_type in cloth_types:
        generate_env_state(args.env_name, args.num_variations, args.is_headless, args.is_render,
                           args.ds_name, cloth_type, args.eval_flag)
