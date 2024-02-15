# %%
# import
import io
import os
import pathlib
import pdb

import imageio
import numpy as np
import torch
import yaml
import hydra
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import pytorch_lightning.utilities.seed as seed_utils
from pytorch_lightning.loggers import WandbLogger

from garmentnets.datasets.conv_implicit_wnf_dataset import ConvImplicitWNFDataModule
from garmentnets.networks.conv_implicit_wnf import ConvImplicitWNFPipeline
from garmentnets.networks.hrnet_nocs import HRNet2NOCS
from garmentnets.networks.pointnet2_nocs import PointNet2NOCS
from utils.data_utils import update_config, find_best_checkpoint

from torch_geometric.data import Dataset, Data, DataLoader, Batch

from visualization.plot import all_in_one_plot_v5, plot_mesh_face
from visualization.plot_utils import get_rotating_frames, save_numpy_as_gif
from PIL import Image

def run_task(vv, log_dir, exp_name):
    parallel = Parallel(n_jobs=64, verbose=1)
    log_dir = os.path.join('/home/zixuanhu/occlusion_reasoning', log_dir)
    vv['model_path'] = vv['model_path'][vv['cloth_type']]
    cfg = OmegaConf.load(vv['model_path'] + '/config.yaml')
    # cfg = update_config(cfg, vv)
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
                    'obs_consist_w': vv['obs_consist_w'], 'consist_iter': vv['consist_iter'],
                    'table_w': vv['table_w'],
                    'lr': vv['opt_lr'], 'opt_model': vv['opt_model']}
    finetune_cfg = OmegaConf.create(finetune_cfg)
    cfg['finetune_cfg'] = finetune_cfg

    os.makedirs(log_dir, exist_ok=True)
    os.chdir(log_dir)
    print('Current working directory is ', os.getcwd())

    cfg.trainname = cfg.valname = f'{cfg.cloth_type}_hard_v4'
    # cfg.valname = f'{cfg.cloth_type}_dataset_v2/val'
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
    # model_path = 'epoch=77-val_loss=1.7064.ckpt'

    model_path = find_best_checkpoint(vv['model_path'])
    # model_path = 'epoch=77-val_loss=1.7064.ckpt'
    state_dict = torch.load(model_path)['state_dict']
    pipeline_model.load_state_dict(state_dict)
    pipeline_model.cuda()
    pipeline_model.eval()
    # pipeline_model.state_dict= state_dict
    # id = cfg.wandb_id
    id = None
    cfg.trainer.resume_from_checkpoint = model_path
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

    for i in [0]:
        data = datamodule.val_dataset[i]
        batch_input = Batch.from_data_list([data],
                                           follow_batch=['cloth_tri', 'cloth_nocs_verts'])
        # pdb.set_trace()
        # for batch_idx, batch in enumerate(datamodule.val_dataloader()):
        batch_input.scale = 1
        #     # pdb.set_trace()
        batch_input = batch_input.to('cuda')

        results = pipeline_model.predict_mesh(batch_input,
                                              finetune_cfg=finetune_cfg,
                                              parallel=parallel,
                                              make_gif=False, get_flat_canon_pose=False)[0]

        rgb = batch_input.rgb[0].permute(1, 2, 0).detach().cpu().numpy()
        from visualization.plot import plot_pointclouds
        #
        # f = all_in_one_plot_v5(rgb, batch_input['cloth_sim_verts'], results['warp_field'],
        #                        results['opt_warp_field'],
        #                        batch_input['cloth_tri'], results['faces'])
        # f.show()
        # frames = get_rotating_frames(f, scene_num=3, frames_num=60, parallel=parallel, height=275, width=1000)
        # imageio.mimsave("test.gif", frames)
        ft_results = results['ft_results']

        def render_func(opt_task_verts, pred_cloth_tri, size=720):
            f = plot_mesh_face(opt_task_verts, pred_cloth_tri, show_grid=False, dis=1.2)
            return np.array(Image.open(io.BytesIO(f.to_image(format="png", width=size, height=size))))

        gt_img = render_func(batch_input['cloth_sim_verts'], batch_input['cloth_tri'])
        init_img = render_func(results['warp_field'], results['faces'])
        opt_img = render_func(results['opt_warp_field'], results['faces'])
        imageio.imwrite('gt_img.png', gt_img)
        imageio.imwrite('init_img.png', init_img)
        imageio.imwrite('opt.png', opt_img)
        imageio.imwrite('rgb.png', rgb)

        def prepare_render2(ft_results_list):
            out = []
            for ft_results in ft_results_list:
                out.append(dict(
                    opt_task_verts=ft_results['new_pos'].detach().cpu().numpy(),
                    pred_cloth_tri=results['faces'],
                )
                )
            return out

        # render_gif_list = prepare_render2(ft_results)
        # pdb.set_trace()
        # for x in render_gif_list:
        #     t = all_in_one_plot_v5(**x)
        # frames = parallel(delayed(render_func)(**x) for x in render_gif_list)
        # save_numpy_as_gif(np.array(frames), 'opt.gif', fps=10, add_index_rate=0)
        # imageio.mimsave("test.gif", frames)
        # imageio.mimsave("test2.gif", results['optimization_gif'] )
        print()

    # trainer = pl.Trainer(
    #     # limit_train_batches=0.1,
    #     # limit_val_batches=0.1,
    #     benchmark=True,
    #     callbacks=[checkpoint_callback],
    #     checkpoint_callback=True,
    #     logger=logger,
    #     check_val_every_n_epoch=1,
    #     fast_dev_run=cfg.is_debug,
    #     **cfg.trainer)
    # trainer.test(model=pipeline_model, datamodule=datamodule)
