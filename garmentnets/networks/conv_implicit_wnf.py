import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import wandb
from joblib import delayed
from torch_geometric.data import Batch
from pytorch3d.loss import chamfer_distance
from garmentnets.common.geometry_util import AABBNormalizer, rotate_particles
from garmentnets.common.marching_cubes_util import wnf_to_mesh
from garmentnets.components.aggregator import VolumeFeatureAggregator, pointsAggregate
from garmentnets.components.unet3d import Abstract3DUNet, DoubleConv
from garmentnets.components.mlp import MLP, MLP2
from garmentnets.components.gridding import ArraySlicer
from garmentnets.common.torch_util import to_numpy
from garmentnets.common.visualization_util import (
    get_vis_idxs, render_nocs_pair,
    render_wnf_points_pair)
from garmentnets.components.gridding import VirtualGrid
from utils.finetune_utils import TestTimeFinetuner
from utils.geometry_utils import mesh_downsampling, get_edges_from_tri

from utils.my_decor import auto_numpy
from utils.loss_utils import my_huber, chamfer_distance_my, one_way_chamfer_numpy, \
    two_way_chamfer_vis
from visualization.plot import nocs3d_fig, plot_optimization_step_mesh, plot_without_gt2, \
    all_in_one_plot3
from pytorch_lightning.utilities import rank_zero_only
from mesh_gnn.rollout import free_drop


# TODO: UNet 3D
class UNet3D(pl.LightningModule):
    def __init__(self, in_channels, out_channels, f_maps=64,
                 layer_order='gcr', num_groups=8, num_levels=4):
        super().__init__()
        self.save_hyperparameters()
        self.abstract_3d_unet = Abstract3DUNet(
            in_channels=in_channels, out_channels=out_channels,
            final_sigmoid=False, basic_module=DoubleConv, f_maps=f_maps,
            layer_order=layer_order, num_groups=num_groups,
            num_levels=num_levels, is_segmentation=False)

    def forward(self, data):
        result = self.abstract_3d_unet(data)
        return result


# TODO: Deocder Network
class ImplicitWNFDecoder(pl.LightningModule):
    def __init__(self, nn_channels=(128, 512, 512, 1),
                 batch_norm=True,
                 init_pts_feat=False,
                 pred_residual=False,
                 end_relu=True,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.init_pts_feat = init_pts_feat
        self.pred_residual = pred_residual
        if init_pts_feat:
            nn_channels[0] = 131
        if end_relu:
            self.mlp = MLP(nn_channels, batch_norm=batch_norm)
        else:
            self.mlp = MLP2(nn_channels, batch_norm=batch_norm)

    def forward(self, features_grid, query_points, init_pts=None, noise_vec=None, extend=False):
        """
        features_grid: (N,C,D,H,W)
        query_points: (N,M,3)
        noise_vec: (N,K,C)
        """
        # normalize query points to (-1, 1), which is
        # requried by grid_sample
        query_points_normalized = 2.0 * query_points - 1.0
        # shape (N,C,M,1,1)
        sampled_features = F.grid_sample(
            input=features_grid,
            grid=query_points_normalized.view(
                *(query_points_normalized.shape[:2] + (1, 1, 3))),
            mode='bilinear', padding_mode='border',
            align_corners=True)
        # shape (N,M,C)
        sampled_features = sampled_features.view(
            sampled_features.shape[:3]).permute(0, 2, 1)
        N, M, C = sampled_features.shape

        # shape (N, num_sample, M, C)
        if self.init_pts_feat:
            sampled_features = torch.cat([sampled_features, init_pts], dim=-1)
        out_features = self.mlp(sampled_features)
        if self.pred_residual:
            out_features, init_pts = torch.broadcast_tensors(out_features, init_pts)
            out_features = out_features + init_pts
        return out_features


class ConvImplicitWNFPipeline(pl.LightningModule):
    def __init__(self,
                 cfg,
                 # pointnet params
                 pointnet2_params,
                 # VolumeFeaturesAggregator params
                 volume_agg_params,
                 # unet3d params
                 unet3d_params,
                 # ImplicitWNFDecoder params
                 volume_decoder_params,
                 surface_decoder_params,
                 mc_surface_decoder_params=None,
                 # training params
                 learning_rate=1e-4,
                 loss_type='l2',
                 volume_loss_weight=1.0,
                 surface_loss_weight=1.0,
                 mc_surface_loss_weight=0,
                 volume_classification=False,
                 volume_task_space=False,
                 # vis params
                 vis_per_items=0,
                 max_vis_per_epoch_train=0,
                 max_vis_per_epoch_val=0,
                 batch_size=None,
                 cloth_nocs_aabb=None,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        criterion = None
        if loss_type == 'l2':
            criterion = nn.MSELoss(reduction='mean')
        elif loss_type == 'smooth_l1':
            criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise RuntimeError("Invalid loss_type: {}".format(loss_type))
        # will be replace from outside
        self.pointnet2_nocs = None
        if pointnet2_params['cfg'].input_type == 'depth':
            volume_agg_params.feat_size = 489
        else:
            volume_agg_params.feat_size = 137
        self.train_canon = self.cfg.get('train_canon', False)
        # loss weight
        self.pair_w = self.cfg.get('pair_w', 1)
        self.loss_type = self.cfg.get('loss_type', 'l2')

        self.init_pos_mode = self.cfg.get('init_pos_mode', 'nocs')
        self.pred_iter = self.cfg.get('pred_iter', 1)

        self.volume_agg = VolumeFeatureAggregator(**volume_agg_params)
        self.unet_3d = UNet3D(**unet3d_params)
        self.volume_decoder = ImplicitWNFDecoder(**volume_decoder_params)
        self.surface_decoder = ImplicitWNFDecoder(
            init_pts_feat=cfg.get('init_pts_feat', False),
            pred_residual=cfg.get('pred_residual', False),
            end_relu=cfg.get('end_relu', True),
            **surface_decoder_params)

        self.mc_surface_decoder = None
        if mc_surface_loss_weight > 0:
            self.mc_surface_decoder = ImplicitWNFDecoder(**mc_surface_decoder_params)
        self.criterion = criterion
        self.binary_criterion = nn.BCEWithLogitsLoss()

        self.volume_loss_weight = volume_loss_weight
        self.surface_loss_weight = surface_loss_weight
        self.mc_surface_loss_weight = mc_surface_loss_weight
        self.volume_classification = volume_classification
        self.volume_task_space = volume_task_space
        self.learning_rate = learning_rate
        self.vis_per_items = vis_per_items
        self.max_vis_per_epoch_train = max_vis_per_epoch_train
        self.max_vis_per_epoch_val = max_vis_per_epoch_val
        self.batch_size = batch_size
        # self.cloth_nocs_aabb = cloth_nocs_aabb
        if cloth_nocs_aabb is None:
            cloth_nocs_aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        self.cloth_nocs_aabb = nn.Parameter(torch.tensor(cloth_nocs_aabb), requires_grad=False)
        # self.gdloader = GarmentnetsDataloader(cfg.cloth_type)
        # self.cloth_nocs_aabb = Parameter(summary['nocs_aabb'], requires_grad=False)
        self.aabbNormalizer = AABBNormalizer(self.cloth_nocs_aabb,
                                             rescale=None)
        self.parallel = None

    # forward function for each stage
    # ===============================
    def pointnet2_forward(self, data):
        # pointnet2
        if self.train_canon:
            self.pointnet2_nocs.train()
            self.pointnet2_nocs.requires_grad_(True)
        else:
            self.pointnet2_nocs.eval()
            self.pointnet2_nocs.requires_grad_(False)
        nocs_result = self.pointnet2_nocs(data)

        # pointnet2
        num_sample = int(nocs_result['per_point_logits'].shape[0] // data.batch.shape[0])
        data.pos = data.pos.repeat(num_sample, 1)
        # generate prediction
        nocs_bins = self.pointnet2_nocs.nocs_bins
        pred_logits = nocs_result['per_point_logits']
        pred_logits_bins = pred_logits.reshape(
            (pred_logits.shape[0], nocs_bins, 3))
        # TODO: sampling from this distribution
        nocs_bin_idx_pred = torch.argmax(pred_logits_bins, dim=1)  # num_pts x 3
        pred_confidence_bins = F.softmax(pred_logits_bins, dim=1)
        pred_confidence = torch.squeeze(torch.gather(
            pred_confidence_bins, dim=1,
            index=torch.unsqueeze(nocs_bin_idx_pred, dim=1)))

        vg = self.pointnet2_nocs.get_virtual_grid()
        pred_nocs = vg.idxs_to_points(nocs_bin_idx_pred)  # num_pts x 3, normalized to [0,1]
        nocs_data = Batch(
            x=nocs_result['per_point_features'],
            pos=pred_nocs,
            batch=nocs_result['per_point_batch_idx'],
            sim_points=data['pos'],
            pred_confidence=pred_confidence)

        nocs_result['nocs_data'] = nocs_data
        return nocs_result

    def unet3d_forward(self, pointnet2_result):
        nocs_data = pointnet2_result['nocs_data']
        # volume agg
        in_feature_volume = self.volume_agg(nocs_data)
        # unet3d
        out_feature_volume = self.unet_3d(in_feature_volume)
        unet3d_result = {
            'in_feature_volume': in_feature_volume,
            'out_feature_volume': out_feature_volume
        }
        return unet3d_result

    def volume_decoder_forward(self, unet3d_result, query_points):
        out_feature_volume = unet3d_result['out_feature_volume']
        out_features = self.volume_decoder(out_feature_volume, query_points)
        pred_volume_value = out_features.view(*out_features.shape[:-1])
        decoder_result = {
            'out_features': out_features,
            'pred_volume_value': pred_volume_value
        }
        return decoder_result

    def surface_decoder_forward(self, unet3d_result, query_points, data):
        N, M, _ = query_points.shape
        init_sim_points = query_points.clone().view(-1, 3).detach()
        init_sim_points = self.aabbNormalizer.inverse(init_sim_points)
        init_sim_points = init_sim_points.view(N, M, 3)
        out_feature_volume = unet3d_result['out_feature_volume']

        out_feature = init_sim_points
        for i in range(self.pred_iter):
            out_features = self.surface_decoder(out_feature_volume, query_points, out_feature,
                                                extend=False)
        # if 'T_vec_pred' in data:
        #     # to world frame
        #     scale = data['T_vec_pred'][:, :, 0:1]
        #     trans = data['T_vec_pred'][:, :, 1:]
        #     out_features = out_features + trans
        #     out_features[..., [0, 2]] = out_features[..., [0, 2]] * scale
        #     data.pos = data.pos + trans[data.batch, 0]
        #     data.pos[:, [0, 2]] = data.pos[:, [0, 2]] * scale[data.batch, 0]
        decoder_result = {
            'out_features': out_features,
            'init_sim_points': init_sim_points
        }
        return decoder_result

    def mc_surface_decoder_forward(self, unet3d_result, query_points):
        out_feature_volume = unet3d_result['out_feature_volume']
        out_features = self.mc_surface_decoder(out_feature_volume, query_points)
        decoder_result = {
            'out_features': out_features
        }
        return decoder_result

    @staticmethod
    def get_aabb_scale_offset(aabb, padding=0.05):
        nocs_radius = 0.5 - padding
        radius = torch.max(torch.abs(aabb), dim=1)[0][:, :2]
        radius_scale = torch.min(nocs_radius / radius, dim=1)[0]
        nocs_z = nocs_radius * 2
        z_length = aabb[:, 1, 2] - aabb[:, 0, 2]
        z_scale = nocs_z / z_length
        scale = torch.minimum(radius_scale, z_scale)

        z_max = aabb[:, 1, 2] * scale
        offset = torch.ones((len(aabb), 3), dtype=aabb.dtype, device=aabb.device) * 0.5
        offset[:, 2] = 1 - padding - z_max
        return scale, offset

    # forward
    # =======
    def forward(self, data):
        volume_task_space = self.volume_task_space
        pointnet2_result = self.pointnet2_forward(data)

        volume_query_points = data['volume_query_points']
        surface_query_points = data['surf_query_points']
        num_sample = int(pointnet2_result['nocs_data'].num_graphs // data.num_graphs)
        volume_query_points = volume_query_points.repeat(num_sample, 1, 1)
        surface_query_points = surface_query_points.repeat(num_sample, 1, 1)
        if volume_task_space:
            pointnet2_result = self.apply_volume_task_space(
                data, pointnet2_result)
        unet3d_result = self.unet3d_forward(pointnet2_result)
        volume_decoder_result = self.volume_decoder_forward(
            unet3d_result, volume_query_points)
        surface_decoder_result = self.surface_decoder_forward(
            unet3d_result, surface_query_points, data)

        result = {
            'pointnet2_result': pointnet2_result,
            'unet3d_result': unet3d_result,
            'volume_decoder_result': volume_decoder_result,
            'surface_decoder_result': surface_decoder_result,
            'num_sample': num_sample,
        }
        if self.mc_surface_decoder is not None:
            mc_surface_query_points = data.mc_surf_query_points
            result['mc_surface_decoder_result'] = self.mc_surface_decoder_forward(
                unet3d_result, mc_surface_query_points)
        return result

    # training
    # ========
    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt

    @rank_zero_only
    def vis_batch(self, batch, batch_idx, result, is_train=False, img_size=256):
        volume_classification = self.volume_classification
        decoder_result = result['volume_decoder_result']
        pointnet2_result = result['pointnet2_result']
        surface_decoder_result = result['surface_decoder_result']
        pred_sim_points = surface_decoder_result['out_features']
        surface_query_points = batch['surf_query_points']

        nocs_data = pointnet2_result['nocs_data']
        pred_nocs = nocs_data.pos
        gt_nocs = batch.y
        batch_idxs = batch.batch
        this_batch_size = batch.num_graphs
        input_pc = batch.pos
        query_points = batch.volume_query_points
        gt_volume_value = batch.gt_volume_value
        gt_sim_points = batch.gt_sim_points
        pred_volume_value = decoder_result['pred_volume_value']

        vis_per_items = self.vis_per_items
        batch_size = self.batch_size
        max_vis_per_epoch = None
        prefix = None
        if is_train:
            max_vis_per_epoch = self.max_vis_per_epoch_train
            prefix = 'train_'
        else:
            max_vis_per_epoch = self.max_vis_per_epoch_val
            prefix = 'val_'

        _, selected_idxs, vis_idxs = get_vis_idxs(batch_idx,
                                                  batch_size=batch_size, this_batch_size=this_batch_size,
                                                  vis_per_items=vis_per_items,
                                                  max_vis_per_epoch=max_vis_per_epoch)

        log_data = dict()
        for i, vis_idx in zip(selected_idxs, vis_idxs):
            label = prefix + str(vis_idx)
            is_this_item = (batch_idxs == i)
            this_gt_nocs = to_numpy(gt_nocs[is_this_item])
            this_pred_nocs = to_numpy(pred_nocs[is_this_item])

            this_query_points = to_numpy(query_points[i])
            this_gt_volume_value = to_numpy(gt_volume_value[i])
            this_pred_volume_value = to_numpy(pred_volume_value[i])
            if volume_classification:
                this_pred_volume_value = to_numpy(torch.sigmoid(pred_volume_value[i]))

            # point cloud is in gripper frame, therefore gripper is at 0,0,0
            this_pc = input_pc[is_this_item]
            this_pc_dist_pred = torch.norm(this_pc, p=None, dim=1)

            nocs_img = render_nocs_pair(this_gt_nocs, this_pred_nocs, img_size=img_size)
            wnf_img = render_wnf_points_pair(this_query_points,
                                             this_gt_volume_value, this_pred_volume_value,
                                             img_size=img_size)
            media = {
                f'{prefix}/NOCS_3D': nocs3d_fig(gt_sim_points[i], pred_sim_points[i],
                                                surface_query_points[i]),
                "global_step": self.trainer.global_step
            }
            self.logger.experiment.log(media)
            #
            # log_data[label] = [wandb.Image(img, caption=label)]
        return log_data

    def infer(self, batch, batch_idx, is_train=True):
        volume_loss_weight = self.volume_loss_weight
        surface_loss_weight = self.surface_loss_weight
        mc_surface_loss_weight = self.mc_surface_loss_weight
        volume_classification = self.volume_classification

        result = self(batch)
        volume_decoder_result = result['volume_decoder_result']
        surface_decoder_result = result['surface_decoder_result']
        pred_volume_value = volume_decoder_result['pred_volume_value']
        pred_sim_points = surface_decoder_result['out_features']
        init_sim_points = surface_decoder_result['init_sim_points']

        gt_volume_value = batch.gt_volume_value  # .repeat(num_sample, 1)
        gt_sim_points = batch.gt_sim_points  # .repeat(num_sample, 1, 1)
        B = batch.num_graphs

        nocs_data = result['pointnet2_result']['nocs_data']
        # 1. pack point cloud
        # 2. pack visible particles
        # gt_sim_points = gt_sim_points.squeeze(1)
        # pred_sim_points = pred_sim_points.squeeze(1)

        volume_criterion = self.criterion
        if self.loss_type == 'l2':
            surface_criterion = nn.MSELoss()
        else:
            surface_criterion = my_huber

        surface_loss = 0
        pair_loss = surface_criterion(pred_sim_points, gt_sim_points)

        # Compute chamfer loss between the whole shapes
        chamfer_whole, _ = chamfer_distance_my(pred_sim_points, gt_sim_points, batch_reduction=None, K=5)
        chamfer_whole = chamfer_whole.mean()

        # chamfer_vis = compute_oneway_chamfer()
        surface_loss = surface_loss + self.pair_w * pair_loss

        if volume_classification:
            volume_criterion = self.binary_criterion
        # when sampling, backprop the smallest loss(best sample)
        loss_dict = {
            'pair_loss': pair_loss,
            'chamfer_whole': chamfer_whole,
        }

        loss_dict.update({
            'volume_loss': volume_loss_weight * volume_criterion(pred_volume_value, gt_volume_value),
            'surface_loss': surface_loss_weight * surface_loss})

        surface_decoder_result['out_features'] = pred_sim_points
        batch.pos = batch.pos[:batch.batch.shape[0]]
        volume_decoder_result['pred_volume_value'] = volume_decoder_result['pred_volume_value'][:B]
        if mc_surface_loss_weight > 0:
            mc_surface_decoder_result = result['mc_surface_decoder_result']
            pred_is_point_on_surface_logits = mc_surface_decoder_result['out_features']
            gt_is_point_on_surface = batch.is_query_point_on_surf
            loss_dict['mc_surface_loss'] = mc_surface_loss_weight * self.binary_criterion(
                pred_is_point_on_surface_logits, gt_is_point_on_surface)

        metrics = dict(loss_dict)
        metrics['loss'] = loss_dict['surface_loss'] + loss_dict['volume_loss']

        for key, value in metrics.items():
            log_key = ('train_' if is_train else 'val_') + key
            self.log(log_key, value, sync_dist=True)
        log_data = self.vis_batch(batch, batch_idx, result, is_train=is_train)
        self.logger.log_metrics(log_data, step=self.global_step)
        return metrics

    def get_obs_pts(self, pointnet2_result, surface_query_points):
        """ get the observed coordinates before canonicalization"""
        nocs_bins = self.pointnet2_nocs.nocs_bins
        pixel_grid, mask_grid, std = pointsAggregate(pointnet2_result['nocs_data'],
                                                     grid_shape=(nocs_bins, nocs_bins, nocs_bins),
                                                     reduce_method='mean'
                                                     )

        # use mask_grid to find voxels that contain point cloud
        mask_grid, pixel_grid = mask_grid.squeeze(), pixel_grid.squeeze().permute(1, 2, 3, 0)
        pts_idx = mask_grid.nonzero()
        scatter_points = pts_idx / (nocs_bins - 1)
        query_points = surface_query_points[0]  # only support batch size = 1
        # TODO: check this
        pw_dist = torch.cdist(query_points, scatter_points)
        min_dist, min_idx = pw_dist.min(-1)

        mask_pts = min_dist < (1 / nocs_bins * 2)
        obs_idx = pts_idx[min_idx]
        obs_pts = pixel_grid[obs_idx[:, 0], obs_idx[:, 1], obs_idx[:, 2]]

        out = {
            'obs_pts': obs_pts,
            'mask_pts': mask_pts,
            'std': std  # Std of points that are scattered into the same voxel
        }
        return out

    @auto_numpy
    def get_canon_tgt(self, canon_verts, faces, mesh_edges, env):
        """
        Get the target pose for canonicalization task. Note that we account for the ambiguity by
        generating multiple plausible targets.
        """
        canon_verts = rotate_particles([180, 180, 0], canon_verts)
        if self.cfg.ambiguity_agnostic:
            if self.cfg.cloth_type in ['Trousers', 'Dress', 'Jumpsuit']:
                vs = np.stack([self.aabbNormalizer.rotate_on_table(canon_verts, [180 * i, 0, 0]) for i in
                               range(2)])
            elif self.cfg.cloth_type == 'Skirt':
                num_bins = 12
                vs = np.stack(
                    [self.aabbNormalizer.rotate_on_table(canon_verts, [360 / num_bins * i, 0, 0]) for i
                     in
                     range(num_bins)])
            else:  # Tshirt
                vs = canon_verts[None]
        else:
            vs = canon_verts[None]
        canon_tgts = []
        for v in vs:
            if self.cfg.canon_model == 'linear':
                flat_verts = self.aabbNormalizer.flat_canon_linear(v)
            elif self.cfg.canon_model == 'sim':
                model_input_data = {'pointcloud': v,
                                    'model_face': faces,
                                    'mesh_edges': mesh_edges}
                flat_verts = free_drop(model_input_data, env=env, mode='sim')
            canon_tgts.append(flat_verts)
        return canon_tgts

    def predict_mesh(self, batch,
                     voxel_size=0.025,
                     finetune_cfg=None,
                     parallel=None,
                     env=None,
                     get_flat_canon_pose=False,
                     real_world=False,
                     make_gif=False):
        results_list = []
        # if finetune_cfg.opt_model:
        #     pred_iter = 100
        #     opt_per_iter = 1
        # else:
        pred_iter = 1
        opt_per_iter = finetune_cfg.opt_iter_total

        render_gif_list = []
        pointnet2_result = self.pointnet2_forward(batch)
        nocs_data_ori = pointnet2_result['nocs_data']
        nocs_data_tmp = Batch(x=nocs_data_ori.x,
                              pos=nocs_data_ori.pos,
                              batch=batch.batch,
                              sim_points=nocs_data_ori.sim_points,
                              pred_confidence=nocs_data_ori.pred_confidence)
        pointnet2_result['nocs_data'] = nocs_data_tmp
        unet3d_result = self.unet3d_forward(pointnet2_result)

        # stage 2 generate volume
        vg = VirtualGrid(grid_shape=(self.cfg.prediction.volume_size,) * 3)
        grid_points = vg.get_grid_points(include_batch=False)
        array_slicer = ArraySlicer(grid_points.shape, (64, 64, 64))
        result_volume = torch.zeros(grid_points.shape[:-1], dtype=torch.float32)
        for i in range(len(array_slicer)):
            slices = array_slicer[i]
            query_points = grid_points[slices]
            query_points_gpu = query_points.to(self.device).reshape(1, -1, 3)
            decoder_result = self.volume_decoder_forward(unet3d_result, query_points_gpu)
            pred_volume_value = decoder_result['pred_volume_value'].view(*query_points.shape[:-1])
            result_volume[slices] = pred_volume_value.detach().cpu()
            del pred_volume_value, decoder_result
        pred_volume = result_volume
        wnf_volume = to_numpy(pred_volume)

        nocs_verts, faces = wnf_to_mesh(wnf_volume,
                                        iso_surface_level=self.cfg.prediction.iso_surface_level,
                                        gradient_threshold=self.cfg.get('mc_thres', 0.1),
                                        sigma=self.cfg.prediction.gradient_sigma,
                                        filter=True)
        if nocs_verts.shape[0] == 0:
            nocs_verts = np.ones((1, 3), dtype=np.float32) * np.nan
            faces = np.zeros((1, 3), dtype=np.int64)

        # reverts to original size and put it onto the table
        canon_verts = self.aabbNormalizer.inverse(nocs_verts)

        # Downsample it to original vertex density
        pruned_id, faces = mesh_downsampling(canon_verts, faces, voxel_size=0.005)
        canon_verts = canon_verts[pruned_id]
        nocs_verts = nocs_verts[pruned_id]

        # query the downsampled points
        surface_query_points = torch.from_numpy(nocs_verts.astype(np.float32)).view(1, -1, 3).to(
            self.device)
        surface_decoder_result = self.surface_decoder_forward(
            unet3d_result, surface_query_points, batch)

        warp_field = surface_decoder_result['out_features'].view(-1, 3)
        obs_results = self.get_obs_pts(pointnet2_result, surface_query_points)  # TODO what's this?

        # scale mesh back to task space before downsampling
        downsample_id, ds_f = mesh_downsampling(canon_verts, faces, voxel_size=voxel_size)
        mesh_edges_ds = get_edges_from_tri(ds_f)
        nocs_verts = torch.tensor(nocs_verts, dtype=torch.float32, device=self.device)
        mesh_edges = get_edges_from_tri(faces)

        # transform the prediction into original space, only used when input is scaled
        if "scale" in batch:
            warp_field[:, [0, 2]] *= batch.scale
            batch.pos[:, [0, 2]] *= batch.scale

        if not self.cfg.tt_finetune:
            opt_per_iter = 1

        if finetune_cfg.opt_mesh_density == 'dense':
            mesh_coord = warp_field
            mesh_edges = mesh_edges
            mesh_faces = faces
        else:
            mesh_coord = warp_field[downsample_id]
            mesh_edges = mesh_edges_ds
            mesh_faces = ds_f

        mesh_faces = torch.tensor(mesh_faces, dtype=torch.long, device=self.device)
        all_edge_len = torch.linalg.norm(mesh_coord[mesh_edges[:, 0]] - mesh_coord[mesh_edges[:, 1]],
                                         dim=-1)
        rest_edge_len = all_edge_len.mean().item()
        margin = all_edge_len.std().item() * 1.5
        results = {
            'pointnet2_result': pointnet2_result,
            "unet3d_result": unet3d_result,
            'volume_decoder_result': pred_volume,
            'surface_query_points': surface_query_points,
            'verts': nocs_verts,
            'canon_verts': canon_verts,
            'faces': faces,
            'mesh_edges': mesh_edges,
            'warp_field': warp_field,
            'verts_ds': nocs_verts[downsample_id],
            'faces_ds': ds_f,
            'warp_field_ds': warp_field[downsample_id],
            'mesh_edges_ds': mesh_edges_ds,
            'downsample_id': downsample_id,
            'rest_edge_len': rest_edge_len,
            'rest_edge_margin': margin
        }

        finetune_cfg.rest_edge_len = rest_edge_len
        finetune_cfg.rest_edge_margin = margin
        finetuner = TestTimeFinetuner(finetune_cfg)
        finetuner.register_new_state(batch.depth,
                                     batch.pos,
                                     mesh_coord,
                                     mesh_faces,
                                     mesh_edges,
                                     obs_results=obs_results,
                                     model_params=self.surface_decoder.parameters())
        ft_results_list = []
        losses = {}
        for opt_iter in range(opt_per_iter):
            ft_results = finetuner.step(opt_iter, optimize=self.cfg.tt_finetune)
            for k, v in ft_results.items():
                if 'loss' in k:
                    if k in losses:
                        losses[k].append(v)
                    else:
                        losses[k] = [v]
            ft_results_list.append(ft_results)

        results['opt_state'] = finetuner.optim.state_dict()

        def prepare_render2(ft_results_list):
            out = []
            for ft_results in ft_results_list:
                out.append(dict(
                    gt_v=batch.cloth_sim_verts.detach().cpu().numpy(),
                    gt_f=batch.cloth_tri.detach().cpu().numpy(),
                    pred_v=ft_results['new_pos'].detach().cpu().numpy(),
                    pred_f=mesh_faces.detach().cpu().numpy(),
                    size=600
                )
                )
            return out

        def prepare_render3(ft_results_list):
            out = []
            for ft_results in ft_results_list:
                out.append(dict(
                    pred_v=ft_results['new_pos'],
                    pred_f=mesh_faces.detach().cpu().numpy(),
                    size=600,
                    # views=['top']
                )
                )
            return out

        results['loss_init'] = ft_results_list[0]['loss']
        results['loss_end'] = ft_results_list[-1]['loss']
        results['chamfer3d_loss_mean'] = ft_results_list[-1]['chamfer3d_loss_mean']

        if finetune_cfg.opt_mesh_density == 'dense':
            results['opt_warp_field'] = ft_results['new_pos']
            results['opt_warp_field_ds'] = ft_results['new_pos'][downsample_id]
        else:
            results['opt_warp_field_ds'] = ft_results['new_pos']
        if 'cloth_sim_verts' in batch:
            results['gt_chamf'] = chamfer_distance(batch.cloth_sim_verts.unsqueeze(0),
                                                   ft_results['new_pos'].unsqueeze(0))[0]
        if make_gif:
            prepare_func = prepare_render3 if real_world else prepare_render2
            if len(ft_results_list) > 0:
                render_gif_list += prepare_func(ft_results_list)
            render_rs = parallel(delayed(plot_optimization_step_mesh)(**x) for x in render_gif_list)
            results['optimization_gif'] = np.array(render_rs)
        results['ft_results'] = ft_results_list
        results = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                   for k, v in results.items()}

        if get_flat_canon_pose:  # for canonicalization task
            canon_tgts = self.get_canon_tgt(results['canon_verts'], results['faces'],
                                            results['mesh_edges'], env)
            canon_ds_tgts = np.stack([x[results['downsample_id']] for x in canon_tgts])
            results['flat_verts_ds'] = canon_ds_tgts
            results['flat_verts'] = np.stack(canon_tgts)
        results_list = [results]

        return results_list

    def eval_metrics(self,
                     batch,
                     results,
                     real_world=False,
                     self_log=False,
                     plot=True,
                     ):
        """
        Args:
            traj_id:
            real_world:
            results: Model prediction of all stages and final reconstructed mesh
            batch:
                gt_volume_value: 128x128x128
                cloth_nocs_verts: Nx3
                cloth_sim_verts: Nx3
                cloth_faces_tri: Fx3

        gt_sim_value will be calculated on the fly

        Returns:
            nocs_err_dist: evaluate canonicalization
            chamfer_nocs: evaluate shape completion
            chamfer_sim: evaluate mesh quality(shape completion+warp field prediction)
        """
        fig = None
        pointnet2_result = results['pointnet2_result']
        nocs_pred = pointnet2_result['per_point_logits']
        nocs_pred_bins = nocs_pred.view(-1, 64, 3)
        vg = self.pointnet2_nocs.get_virtual_grid()
        nocs_bin_idx_pred = torch.argmax(nocs_pred_bins, dim=1)
        pred_confidence_bins = F.softmax(nocs_pred_bins, dim=1)
        pred_nocs = vg.idxs_to_points(nocs_bin_idx_pred)

        chamfer_pc_opt = 0
        chamfer_mesh_opt = 0
        if 'rgb' in batch:
            rgb = batch.rgb[0].detach().permute(1, 2, 0)
        else:
            rgb = batch.depth.squeeze()
        depth = batch.depth.squeeze()
        cloth_uv = depth.nonzero()
        depth = depth.unsqueeze(2).repeat(1, 1, 3)

        nocs_err_dist = chamfer_nocs = chamfer_mesh = chamfer_pc = 0
        if real_world:
            img_nocs = torch.ones((200, 200, 3), device=depth.device)
            img_nocs[cloth_uv[:, 0], cloth_uv[:, 1]] = pred_nocs

            pred_task_verts = results['warp_field']
            chamfer_pc = one_way_chamfer_numpy(pred_task_verts, batch.pos)
            if 'opt_warp_field' in results:
                opt_warp_field = results['opt_warp_field']
                chamfer_pc_opt = one_way_chamfer_numpy(opt_warp_field, batch.pos)
            if plot:
                fig = plot_without_gt2(rgb, depth,
                                       img_nocs,
                                       pred_nocs,
                                       results['verts'],
                                       results['warp_field'],
                                       results.get('opt_warp_field', results['warp_field']),
                                       results['faces'],
                                       # chamfer_pc,
                                       batch.pos
                                       )
        else:

            # Compute canonicalization error
            nocs_gt = batch.y
            num_sample = int(pred_nocs.shape[0] // nocs_gt.shape[0])
            expand_nocs_tgt = nocs_gt.repeat(num_sample, 1)
            nocs_err_dist = torch.norm(pred_nocs - expand_nocs_tgt, dim=-1).view(num_sample,
                                                                                 nocs_gt.shape[0]).mean(
                -1)
            # print(nocs_err_dist)
            min_nocs_err_dist, min_nocs_id = nocs_err_dist.min(0)
            nocs_err_dist = nocs_err_dist.mean()

            gt_task_verts = batch['cloth_sim_verts']
            pred_task_verts = torch.tensor(results['warp_field']).cuda()

            chamfer_mesh, _ = chamfer_distance_my(pred_task_verts.unsqueeze(0),
                                                  gt_task_verts.unsqueeze(0))

            gt_nocs_verts = batch['cloth_nocs_verts']
            pred_nocs_verts = torch.tensor(results['verts']).cuda()

            chamfer_nocs, _ = chamfer_distance_my(pred_nocs_verts.unsqueeze(0),
                                                  gt_nocs_verts.unsqueeze(0))
            faces = results['faces']

            opt_warp_field = torch.tensor(results['opt_warp_field']).cuda()
            chamfer_mesh_opt = chamfer_distance_my(gt_task_verts.unsqueeze(0),
                                                   opt_warp_field.unsqueeze(0))
            chamfer_pc_opt = two_way_chamfer_vis(opt_warp_field, faces, batch.pos)
            chamfer_pc = two_way_chamfer_vis(pred_task_verts, faces, batch.pos)

            if plot:
                pred_nocs = pred_nocs.view(num_sample, -1, 3)[min_nocs_id]
                fig = all_in_one_plot3(rgb, depth,
                                       nocs_gt, pred_nocs,
                                       gt_nocs_verts, pred_nocs_verts,
                                       gt_task_verts, pred_task_verts, opt_warp_field,
                                       batch['cloth_tri'], results['faces']
                                       )

        suffix = ''
        metric = {
            'test/nocs_err_dist': nocs_err_dist,
            'test/chamfer_nocs': chamfer_nocs,
            'test/chamfer_mesh': chamfer_mesh,
            'test/chamfer_pc': chamfer_pc,
            'test/chamfer_pc_opt': chamfer_pc_opt,
        }
        if 'opt_warp_field_ds' in results:
            metric['test/chamfer_mesh_opt'] = chamfer_mesh_opt

        if plot:
            media = {
                f'3d_plot' + suffix: fig,
            }
            metric.update(media)
        try:
            if self_log:
                self.logger.experiment.log(metric)
            else:
                wandb.log(metric)
        except:
            print('Wandb is not initialized')
        return metric

    def training_step(self, batch, batch_idx):
        metrics = self.infer(batch, batch_idx, is_train=True)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self.infer(batch, batch_idx, is_train=False)
        return metrics['loss']

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        print(batch)

        # metrics = self.infer(batch, batch_idx, is_train=False)
        batch.scale = 1
        pdb.set_trace()
        results = self.predict_mesh(batch,
                                    finetune_cfg=self.cfg.tt_finetune_cfg,
                                    parallel=self.parallel, make_gif=False, get_flat_canon_pose=False)[0]
        metrics = self.eval_metrics(batch, results,
                                    real_world=False, traj_id=batch_idx, pick_id=0,
                                    self_log=True, plot=True)

        return metrics['test/chamfer_mesh']
