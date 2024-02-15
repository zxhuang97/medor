# %%
# import
import io
import os
import pathlib
import pdb

import numpy as np
import torch
import wandb
import yaml
import hydra
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import pytorch_lightning.utilities.seed as seed_utils
from pytorch_lightning.loggers import WandbLogger

from garmentnets.datasets.conv_implicit_wnf_dataset import ConvImplicitWNFDataModule
from garmentnets.networks.conv_implicit_wnf import ConvImplicitWNFPipeline
from garmentnets.networks.hrnet_nocs import HRNet2NOCS
from garmentnets.networks.pointnet2_nocs import PointNet2NOCS
from utils.data_utils import update_config, find_best_checkpoint, TensorDict
from visualization.plot import plot_mesh_face
from torch_geometric.data import Dataset, Data, DataLoader
import cv2

def run_task(vv, log_dir, exp_name):
    log_dir = os.path.join('/home/zixuanhu/occlusion_reasoning', log_dir)
    vv['model_path'] = vv['model_path'][vv['cloth_type']]
    cfg = OmegaConf.load(vv['model_path'] + '/config.yaml')
    cfg = update_config(cfg, vv)
    cfg.finetune = True
    cfg.conv_implicit_model.vis_per_items = 1
    cfg.conv_implicit_model.max_vis_per_epoch_val = 40
    cfg.conv_implicit_model.mesh_mode = vv['mesh_mode']
    pred_cfg = OmegaConf.load('garmentnets/config/predict_default.yaml')
    cfg['prediction'] = pred_cfg.prediction
    seed_utils.seed_everything(cfg.seed)
    finetune_cfg = {'opt_mesh_density': vv['opt_mesh_density'],
                    'opt_mesh_init': vv['opt_mesh_init'],
                    'opt_iter_total': vv['opt_iter_total'],
                    'chamfer_mode': 'scipy', 'chamfer3d_w': vv['chamfer3d_w'],
                    'laplacian_w': vv['laplacian_w'], 'normal_w': vv['normal_w'],
                    'edge_w': vv['edge_w'], 'rest_edge_len': 0.,
                    'depth_w': vv['depth_w'], 'silhouette_w': vv['silhouette_w'],
                    'rigid_w': vv['rigid_w'],
                    'obs_consist_w': vv['obs_consist_w'], 'consist_iter': vv['consist_iter'],
                    'table_w': vv['table_w'],
                    'lr': vv['opt_lr'], 'opt_model': vv['opt_model']}
    finetune_cfg = OmegaConf.create(finetune_cfg)
    cfg['finetune_cfg'] = finetune_cfg

    os.makedirs(log_dir, exist_ok=True)
    os.chdir(log_dir)
    print('Current working directory is ', os.getcwd())

    if '/' in cfg.ds:
        cloth_instance = cfg.ds.split('/')[0]
        cfg.cloth_type = ''.join([i for i in cloth_instance if not i.isdigit()])
        cfg.trainname = f'{cfg.ds}/train'
        cfg.valname = f'{cfg.ds}/val'
    elif 'human' in cfg.ds:
        name, ds = cfg.ds.split('_')
        cfg.cloth_type = ds
        cfg.trainname = f'{name}/{ds}/train'
        cfg.valname = f'{name}/{ds}/val'
    else:
        ctype, version = cfg.ds.split('_')
        cfg.cloth_type = ctype
        cfg.trainname = f'{ctype}_dataset_{version}/train'
        cfg.valname = f'{ctype}_dataset_{version}/val'
    cfg.datamodule.batch_size = 1
    # cfg.num_workers = 1
    datamodule = ConvImplicitWNFDataModule(cfg, **cfg.datamodule)
    datamodule.prepare_data()

    batch_size = datamodule.kwargs['batch_size']
    # batch_size = 8
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
    pipeline_model.canon_nocs = pointnet2_model
    pipeline_model.batch_size = batch_size

    model_path = find_best_checkpoint(vv['model_path'])
    # model_path = os.path.join(vv['model_path'], 'epoch=42-val_loss=0.0009.ckpt')
    # model_path = os.path.join(vv['model_path'], 'epoch=52-val_loss=0.0018.ckpt')
    state_dict = torch.load(model_path)['state_dict']
    pipeline_model.load_state_dict(state_dict)
    pipeline_model.to('cuda')
    id = None
    cfg.trainer.resume_from_checkpoint = model_path
    category = cfg.ds
    # cfg.logger.tags.append(category)
    # logger = WandbLogger(
    #     project="Occluded cloth",
    #     name=exp_name,
    #     id=id,
    #     **cfg.logger)
    # cfg.wandb_id = logger.experiment.id
    # logger.log_hyperparams(cfg)
    wandb.init(project='occluded cloth',
               name=exp_name,
               resume='allow',
               # id='2gl5iftu',
               # settings=wandb.Settings(start_method='fork')
               )
    wandb.config.update(cfg, allow_val_change=True)
    OmegaConf.save(cfg, open('config.yaml', 'w'))
    pipeline_model.eval()
    result_dict = TensorDict()
    count = 0
    # for batch_idx, batch in enumerate(datamodule.train_dataloader()):
    for batch_idx, batch in enumerate(datamodule.val_dataloader()):
        batch.scale = 1
        # pdb.set_trace()
        batch = batch.to('cuda')

        results = pipeline_model.predict_mesh(batch,
                                              finetune_cfg=finetune_cfg,
                                              parallel=None,
                                              make_gif=False, get_flat_canon_pose=False)[0]
        metrics = pipeline_model.eval_metrics(batch, results,
                                              real_world=False, traj_id=batch_idx, pick_id=0,
                                              self_log=False, plot=True)

        # pdb.set_trace()
        size = 512
        # metrics['3d_plot'].write_html(f'{batch_idx}_vis.html')
        f = plot_mesh_face(results['warp_field'], results['faces'], show_grid=False)
        # f = plot_mesh_face(batch['cloth_sim_verts'], batch['cloth_tri'], show_grid=False)

        img = np.array(Image.open(io.BytesIO(f.to_image(format='png', width=size, height=size))))
        cv2.imwrite(f'{batch_idx}_vis.png', img[:, :, ::-1])
        f = plot_mesh_face(results['opt_warp_field'], results['faces'], show_grid=False)
        # f = plot_mesh_face(batch['cloth_sim_verts'], batch['cloth_tri'], show_grid=False)

        img = np.array(Image.open(io.BytesIO(f.to_image(format='png', width=size, height=size))))
        cv2.imwrite(f'{batch_idx}_vis_opt.png', img[:, :, ::-1])
        cv2.imwrite(f'{batch_idx}_obs.png', batch.rgb[0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1])
        metrics.pop('3d_plot')



        result_dict.update_by_add(metrics)
        # if batch_idx > 10:
        #     break
        count += 1
    mean_dict = result_dict.get_stats()
    for k, v in mean_dict.items():
        print(k, v)
    # print(mean_dict)
    # pdb.set_trace()
    # trainer.validate(model=pipeline_model, datamodule=datamodule)
