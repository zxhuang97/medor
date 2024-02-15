import argparse
import os
import pdb

import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
import pytorch_lightning.utilities.seed as seed_utils
from pytorch_lightning.loggers import WandbLogger

from datasets.conv_implicit_wnf_dataset import ConvImplicitWNFDataModule
from garmentnets.networks.hrnet_nocs import HRNet2NOCS
from networks.pointnet2_nocs import PointNet2NOCS
from utils.data_utils import update_config, find_best_checkpoint
from utils.misc_utils import set_resource


def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test', help='Name of the experiment')
    parser.add_argument('--log_dir', type=str, default='data/gn_test_trousers/',
                        help='Logging directory')
    parser.add_argument('--ds', type=str, default='Trousers_dataset_v2', help='dataset name')
    parser.add_argument('--cloth_type', type=str, default='Trousers', help='cloth type')
    parser.add_argument('--input_type', type=str, default='depth',
                        help='input type of the model: pc or depth')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='max number of workers for data loader')
    parser.add_argument("--input_noise_scale", type=float,
                        default=0.005, help="add noise to the model input (depth/pc)")
    parser.add_argument('--is_debug', default=False, action="store_true",
                        help='debug mode')
    # parser.add_argument('--data_aug', default=False, action="store_true",
    #                     help='whether to use data augmentation during training')
    parser.add_argument('--resume', default=False, action="store_true")
    args = parser.parse_args()
    return args.__dict__


def run_task(args):
    cfg = OmegaConf.load('garmentnets/config/train_pointnet2_default.yaml')
    cfg = update_config(cfg, args)
    exp_name = cfg.exp_name
    log_dir = os.path.join(cfg.log_dir, exp_name)
    seed_utils.seed_everything(cfg.seed)
    os.makedirs(log_dir, exist_ok=True)
    datamodule = ConvImplicitWNFDataModule(cfg, **cfg.datamodule)
    batch_size = datamodule.kwargs['batch_size']

    if cfg.input_type == 'pc':
        model = PointNet2NOCS(cfg, batch_size=batch_size, **cfg.model)
    else:
        model = HRNet2NOCS(cfg, batch_size=batch_size, **cfg.model)
    model.batch_size = batch_size
    if cfg.resume:
        model_path = find_best_checkpoint(log_dir)
        cfg.resume_from_checkpoint = model_path
        id = cfg.wandb_id
    else:
        model_path = None
        cfg.resume_from_checkpoint = None
        id = None
    if model_path is not None:
        model_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(model_dict)
    logger = WandbLogger(
        project="Occluded cloth",
        name=exp_name,
        id=id,
        resume='allow',
        settings=wandb.Settings(start_method='thread'),
        **cfg.logger)
    cfg.wandb_id = logger.experiment.id
    OmegaConf.save(cfg, open(os.path.join(log_dir, 'config.yaml'), 'w'))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=log_dir,
        filename="{epoch}-{val_loss:.4f}",
        monitor='val_loss',
        save_last=True,
        save_top_k=5,
        every_n_epochs=5,
        mode='min',
        save_weights_only=False,
        # every_n_epochs=1,
        save_on_train_epoch_end=True)
    trainer = pl.Trainer(
        benchmark=True,
        callbacks=[checkpoint_callback],
        checkpoint_callback=True,
        logger=logger,
        check_val_every_n_epoch=1,
        fast_dev_run=cfg.is_debug,
        max_epochs=cfg.max_epochs,
        **cfg.trainer)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.resume_from_checkpoint)


def main():
    set_resource()
    args = get_default_args()
    run_task(args)


if __name__ == '__main__':
    main()
