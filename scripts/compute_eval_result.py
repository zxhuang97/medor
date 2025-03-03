import pdb

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import glob
import pickle



# from omegaconf import OmegaConf

def read_info(p):
    with open(p, 'rb') as f:
        x = pickle.load(f)
    return x


def read_all_infos(pattern):
    # return a list of info_traj
    traj_infos = []
    dirs = glob.glob(pattern)
    # print(dirs)
    for d in dirs:
        trajs = glob.glob(os.path.join(d, 'transformed_info_traj_*.pkl'))
        for traj in trajs:
            traj_infos.append(read_info(traj))
    print(len(traj_infos))
    return traj_infos


all_runs = {}
# all_runs['Ours'] = {
#     'flatten': {
#         'Trousers': 'data/test/0115_noise_flow0.1/Trousers_flatten_inv_chamfer_0*',
#         'Skirt': 'data/test/0114_noise_flow_Skirt/Skirt_flatten_inv_chamfer_0*',
#         'Tshirt': 'data/test/0115_noise_flow0.1/Tshirt_flatten_inv_chamfer_0*',
#         'Dress': 'data/test/0115_noise_flow0.1/Dress_flatten_inv_chamfer_0*',
#         # 'Jumpsuit': 'data/test/0121_noise_flow/Jumpsuit_canon_rigid_inv_01*',
#     },
#     'canon_rigid': {
#         'Trousers': 'data/test/0115_noise_flow0.1/Trousers_canon_rigid_inv_chamfer_0*',
#         'Skirt': 'data/test/0114_noise_flow_Skirt/Skirt_canon_rigid_inv_chamfer_0*',
#         'Tshirt': 'data/test/0115_noise_flow0.1/Tshirt_canon_rigid_inv_chamfer_0*',
#         'Dress': 'data/test/0115_noise_flow0.1/Dress_canon_rigid_inv_chamfer_0*',
#         # 'Jumpsuit': 'data/test/0121_noise_flow/Jumpsuit_canon_rigid_inv_01*'
#     }
# }
all_runs["release"] = {
    "flatten":{
        "Trousers": "data/release_plan_test/0226/"
    }

}
num_runs = len(all_runs.keys())
categories = ['Trousers']
tasks = ['flatten']
# tasks = ['canon_rigid']
num_categories = len(categories)
mode = 'perc'  # 'perc'
tgt_picks = [10]
# task = 'flatten'  # canon  canon_rigid  flatten
# all_keys = {'flatten': {'normalized_performance': 'normalized_improvement'},
#             'canon': {'normalized_canon_improvement': 'normalized_canon_improvement'},
#             'canon_rigid': {'normalized_canon_rigid_improvement': 'normalized_canon_rigid_improvement'}}
all_keys = {'flatten': {'normalized_coverage_improvement': 'normalized_coverage_improvement'},
            'canon': {'normalized_canon_improvement': 'normalized_canon_improvement'},
            'canon_rigid': {'normalized_canon_rigid_improvement': 'normalized_canon_rigid_improvement'}}
all_keys_results = {}


# read the log of all experiments
# for task in ['flatten', 'canon_rigid', 'canon']:
for task in tasks:
    keys = all_keys[task]
    print(task)
    for key_old, key_new in keys.items():
        all_results = {}
        for run_name, run_info in all_runs.items():
            if task not in run_info:
                continue
            run_info = run_info[task]
            print(f'---------------------{run_name}------------------------')
            run_results = {}
            for cloth, cloth_pattern in run_info.items():
                print(f'##############  {cloth}  ################')
                traj_infos = read_all_infos(cloth_pattern)
                if key_old not in traj_infos[0]:
                    continue
                # print(f'##############  {key_new}  #################')
                cloth_results = [[], [], []]
                for num_p in tgt_picks:
                    scores = []
                    if key_old == 'planning_time':
                        for info in traj_infos:
                            scores.extend(info[key_old][0])
                    else:
                        for info in traj_infos:
                            result = info[key_old][0]
                            real_pick = min(len(result), num_p)
                            score = min(result[real_pick - 1], 1.0)
                            # score = result[real_pick-1]
                            scores.append(score)
                    # print(scores)
                    if mode == 'mean':
                        cloth_results[0].append(np.mean(scores) - np.std(scores))
                        cloth_results[1].append(np.mean(scores))
                        cloth_results[2].append(np.mean(scores) + np.std(scores))
                        # print(cloth_results[0][-1], cloth_results[1][-1], cloth_results[2][-1])
                    else:
                        cloth_results[0].append(np.percentile(scores, 25))
                        cloth_results[1].append(np.percentile(scores, 50))
                        cloth_results[2].append(np.percentile(scores, 75))
                run_results[cloth] = cloth_results
            all_results[run_name] = run_results
        all_keys_results[key_new] = all_results



for j, (run_name, _) in enumerate(all_results.items()):

    print(f'###############     {run_name: <20}      ##################')
    means, margins = [], []
    for task in tasks:
        keys = all_keys[task]
        for key_old, key_new in keys.items():
            print(key_new)
            all_results = all_keys_results[key_new][run_name]

            for i, cloth_name in enumerate(categories):
                if cloth_name == 'Jumpsuit':
                    continue
                if cloth_name not in all_results:
                    continue
                cur_res = all_results[cloth_name]
                # if cloth_name not in run_results:
                #     continue
                #
                # cur_res = run_results[cloth_name]

                minus = np.array(cur_res[1]) - np.array(cur_res[0])
                plus = np.array(cur_res[2]) - np.array(cur_res[1])
                margin = np.maximum(minus, plus)
                means.append(np.mean(cur_res[1]))
                margins.append(np.mean(margin))
        print(f'{run_name: <50}     {np.mean(means):.3f}+-{np.mean(margins):.3f}\n\n')
    print(f'{cloth_name: <8}     {np.mean(cur_res[1]):.3f}+-{np.mean(margin):.3f}')
    print('Averaged over all categories')

    print(f'{run_name: <50}     {np.mean(means):.3f}+-{np.mean(margins):.3f}\n\n')

