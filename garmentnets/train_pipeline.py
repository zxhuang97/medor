# %%
# import
import argparse
import os
import pathlib
import pdb
import socket

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import pytorch_lightning.utilities.seed as seed_utils
from pytorch_lightning.loggers import WandbLogger
import wandb
# from pytorch_lightning.plugins import DDPPlugin

from datasets.conv_implicit_wnf_dataset import ConvImplicitWNFDataModule
from garmentnets.networks.hrnet_nocs import HRNet2NOCS
from networks.conv_implicit_wnf import ConvImplicitWNFPipeline
from networks.pointnet2_nocs import PointNet2NOCS
from utils.data_utils import update_config, find_best_checkpoint
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

from utils.misc_utils import set_resource


def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test', help='Name of the experiment')
    parser.add_argument('--log_dir', type=str, default='data/release/', help='Logging directory')
    parser.add_argument('--ds', type=str, default='Trousers_v3', help='dataset name')
    parser.add_argument('--cloth_type', type=str, default='Trousers', help='cloth type')
    parser.add_argument('--canon_checkpoint', type=str, default='data/gn_debug/test', help='cloth type')
    parser.add_argument('--resume', type=bool, default=False, help='resume training')
    parser.add_argument("--pair_w", type=float, default=1.,
                        help="Weight for pair loss")
    parser.add_argument("--pred_iter", type=int, default=1,
                        help="Number of iterations for predicting the surface")
    parser.add_argument("--train_canon", default=False, action="store_true")
    parser.add_argument('--input_type', type=str, default='depth',
                        help='input type of the model: pc or depth')
    parser.add_argument("--init_pts_feat", type=bool, default=True,
                        help="Whether to use point position as feature for surface prediction")
    parser.add_argument("--pred_residual", type=bool, default=True)
    parser.add_argument("--end_relu", type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    args = parser.parse_args()
    return args.__dict__


def run_task(args):
    cfg = OmegaConf.load('garmentnets/config/train_pipeline_default.yaml')
    pred_cfg = OmegaConf.load('garmentnets/config/predict_default.yaml')
    cfg = update_config(cfg, args)
    exp_name = cfg.exp_name
    log_dir = os.path.join(cfg.log_dir, exp_name)
    seed_utils.seed_everything(cfg.seed)

    print('Current working directory is ', os.getcwd())
    if args['resume']:
        cfg = OmegaConf.load(os.path.join(log_dir, 'config.yaml'))

    cfg['prediction'] = pred_cfg.prediction
    cfg.finetune = False
    finetune_cfg = {'opt_mesh_density': cfg['opt_mesh_density'],
                    'opt_mesh_init': cfg['opt_mesh_init'],
                    'opt_iter_total': cfg['opt_iter_total'],
                    'chamfer_mode': 'scipy', 'chamfer3d_w': cfg['chamfer3d_w'],
                    'laplacian_w': cfg['laplacian_w'], 'normal_w': cfg['normal_w'],
                    'edge_w': cfg['edge_w'], 'rest_edge_len': 0.,
                    'depth_w': cfg['depth_w'], 'silhouette_w': cfg['silhouette_w'],
                    'obs_consist_w': cfg['obs_consist_w'], 'consist_iter': cfg['consist_iter'],
                    'table_w': cfg['table_w'],
                    'lr': cfg['opt_lr'], 'opt_model': False}
    finetune_cfg = OmegaConf.create(finetune_cfg)
    cfg['finetune_cfg'] = finetune_cfg
    datamodule = ConvImplicitWNFDataModule(cfg,
                                           **cfg.datamodule)
    batch_size = datamodule.kwargs['batch_size']
    cfg['canon_ckpt'] = cfg.get('canon_ckpt', '*.ckpt')

    if cfg.input_type == 'pc':
        pointnet2_model = PointNet2NOCS.load_from_checkpoint(
            find_best_checkpoint(cfg.canon_checkpoint))
    else:
        pointnet2_model = HRNet2NOCS.load_from_checkpoint(
            find_best_checkpoint(cfg.canon_checkpoint))
        os.system(f"cp {find_best_checkpoint(cfg.canon_checkpoint)} {log_dir}/canon.ckpt")
    # pointnet2_model.batch_size = batch_size
    if args['resume']:
        model_path = find_best_checkpoint(log_dir)
        id = cfg.wandb_id
        cfg.trainer.resume_from_checkpoint = model_path
    else:
        model_path = None
        id = None
        cfg.trainer.resume_from_checkpoint = model_path

    category = cfg.ds
    pointnet2_params = dict(pointnet2_model.hparams)
    pipeline_model = ConvImplicitWNFPipeline(
        cfg,
        pointnet2_params=pointnet2_params,
        batch_size=batch_size, **cfg.conv_implicit_model,
        cloth_nocs_aabb=datamodule.cloth_nocs_aabb,
    )
    pipeline_model.pointnet2_nocs = pointnet2_model

    os.makedirs(log_dir, exist_ok=True)
    if rank_zero_only.rank == 0:
        logger = WandbLogger(
            entity="zixuanh",
            project="Occluded cloth",
            name=exp_name,
            settings=wandb.Settings(start_method='thread'),
            id=id,
            save_dir=log_dir,
            **cfg.logger)
        cfg.wandb_id = logger.experiment.id
        logger.log_hyperparams(cfg)
        OmegaConf.save(cfg, open(os.path.join(log_dir, 'config.yaml'), 'w'))
    else:
        logger = None

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=log_dir,
        filename="{epoch}-{val_loss:.4f}",
        monitor='val_loss',
        save_last=True,
        save_top_k=10,
        mode='min',
        save_weights_only=False,
        save_on_train_epoch_end=True)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        checkpoint_callback=True,
        logger=logger,
        max_epochs=cfg.get("max_epochs" , 200),
        check_val_every_n_epoch=1,
        **cfg.trainer
    )

    trainer.fit(model=pipeline_model, datamodule=datamodule)


def main():
    set_resource()
    args = get_default_args()
    run_task(args)


if __name__ == '__main__':
    main()
