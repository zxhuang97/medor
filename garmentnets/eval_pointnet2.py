# %%
# import
import os
import pathlib

import torch
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import pytorch_lightning.utilities.seed as seed_utils
from pytorch_lightning.loggers import WandbLogger

from garmentnets.datasets.conv_implicit_wnf_dataset import ConvImplicitWNFDataModule
from garmentnets.networks.hrnet_nocs import HRNet2NOCS
from garmentnets.networks.pointnet2_nocs import PointNet2NOCS
from utils.data_utils import update_config, find_best_checkpoint


def run_task(vv, log_dir, exp_name):
    log_dir = os.path.join('/home/zixuanhu/occlusion_reasoning', log_dir)
    cfg = OmegaConf.load(vv['model_path'] + '/config.yaml')
    cfg = update_config(cfg, vv)
    cfg.model.vis_per_items = 3
    cfg.model.max_vis_per_epoch_val = 100
    seed_utils.seed_everything(cfg.seed)

    os.makedirs(log_dir, exist_ok=True)
    os.chdir(log_dir)
    print('Current working directory is ', os.getcwd())

    if cfg.ds == 'Trousers':
        cfg.trainname = '1029_Trousers_train'
        cfg.valname= '1029_Trousers_val'
    datamodule = ConvImplicitWNFDataModule(cfg, **cfg.datamodule)
    batch_size = datamodule.kwargs['batch_size']
    # batch_size = 8
    if cfg.input_type == 'pc':
        model = PointNet2NOCS(cfg, batch_size=batch_size, **cfg.model)
    else:
        model = HRNet2NOCS(cfg, batch_size=batch_size, **cfg.model)
    model.batch_size = batch_size
    model_path = find_best_checkpoint(vv['model_path'])
    # model_path = 'epoch=77-val_loss=1.7064.ckpt'

    model_path = find_best_checkpoint(vv['model_path'])
    # model_path = 'epoch=77-val_loss=1.7064.ckpt'
    model.load_state_dict(torch.load(model_path)['state_dict'])
    # id = cfg.wandb_id
    id = None
    cfg.trainer.resume_from_checkpoint = model_path
    # TODO: adapt to different ds
    category = cfg.ds
    cfg.logger.tags.append(category)
    logger = WandbLogger(
        project="Occluded cloth",
        name=exp_name,
        id=id,
        **cfg.logger)
    cfg.wandb_id = logger.experiment.id
    OmegaConf.save(cfg, open('config.yaml', 'w'))
    logger.log_hyperparams(cfg)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./",
        filename="{epoch}-{val_loss:.4f}",
        monitor='val_loss',
        save_last=True,
        save_top_k=10,
        mode='min',
        save_weights_only=False,
        every_n_epochs=1,
        save_on_train_epoch_end=True)
    trainer = pl.Trainer(
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        benchmark=True,
        callbacks=[checkpoint_callback],
        checkpoint_callback=True,
        logger=logger,
        check_val_every_n_epoch=1,
        fast_dev_run=cfg.is_debug,
        **cfg.trainer)
    trainer.validate(model=model, datamodule=datamodule)
    # # exit()
    # trainer.fit(model=model, datamodule=datamodule)

