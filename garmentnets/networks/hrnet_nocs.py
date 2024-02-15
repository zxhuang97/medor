import pdb

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch_geometric.data import Batch

from garmentnets.components.hrnet import hrnet32
from garmentnets.common.torch_util import to_numpy
from garmentnets.common.visualization_util import get_vis_idxs, render_nocs_pair, render_confidence_pair
from garmentnets.components.gridding import VirtualGrid
from garmentnets.components.loss import MirrorMSELoss
from garmentnets.components.symmetry import mirror_nocs_points_by_axis
import torch_scatter

# helper functions
# ================
# from utils.loss_utils import kl_normal, JS_categorical_pairwise
# from utils.tensor_utils import get_pairwise_index
from visualization.plot import nocs3d_fig


def local_idx_to_batch_idx(data, batch_idxs, local_idxs):
    batch_global_idxs = torch.arange(
        len(data), dtype=torch.int64, device=data.device)
    global_idxs = list()
    for i in range(len(local_idxs)):
        is_this_batch = (batch_idxs == i)
        this_idxs = batch_global_idxs[is_this_batch]
        global_idx = this_idxs[local_idxs[i]]
        global_idxs.append(global_idx)
    global_idxs_tensor = torch.tensor(
        global_idxs, dtype=local_idxs.dtype, device=local_idxs.device)
    return global_idxs_tensor


def predict_grip_point_nocs(point_cloud, pred_nocs, batch_idxs, batch_size):
    # point cloud is in gripper frame, therefore gripper is at 0,0,0
    batch_global_idxs = torch.arange(len(point_cloud),
                                     dtype=torch.int64, device=point_cloud.device)
    pc_dist_to_gripper = torch.norm(point_cloud, p=None, dim=1)
    global_grip_idxs = list()
    for i in range(batch_size):
        is_this_batch = (batch_idxs == i)
        this_dist = pc_dist_to_gripper[is_this_batch]
        this_idxs = batch_global_idxs[is_this_batch]
        local_grip_idx = torch.argmin(this_dist)
        global_grip_idx = this_idxs[local_grip_idx]
        global_grip_idxs.append(global_grip_idx)
    global_grip_idxs_tensor = torch.tensor(
        global_grip_idxs, dtype=torch.int64, device=pred_nocs.device)

    pred_grip_nocs = pred_nocs[global_grip_idxs_tensor]
    return pred_grip_nocs


# modules
# =======
class HRNet2NOCS(pl.LightningModule):
    def __init__(self,
                 cfg,
                 # # architecture params
                 # feature_dim, batch_norm, dropout,
                 # sa1_ratio, sa1_r,
                 # sa2_ratio, sa2_r,
                 # fp3_k, fp2_k, fp1_k,
                 symmetry_axis=None,
                 nocs_bins=None,
                 # training params
                 learning_rate=1e-4,
                 nocs_loss_weight=1,
                 grip_point_loss_weight=1,
                 # vis params
                 vis_per_items=0,
                 max_vis_per_epoch_train=0,
                 max_vis_per_epoch_val=0,
                 batch_size=None,
                 **kwargs,
                 ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = hrnet32(cfg,
                                in_dim=1,
                                out_dim=nocs_bins * 3,
                                )
        criterion = None
        if symmetry_axis is None:
            criterion = nn.MSELoss()
        else:
            criterion = MirrorMSELoss()
        self.criterion = criterion
        self.grip_point_criterion = nn.MSELoss()

        self.nocs_bins = nocs_bins
        self.learning_rate = learning_rate
        self.nocs_loss_weight = nocs_loss_weight
        self.grip_point_loss_weight = grip_point_loss_weight
        self.vis_per_items = vis_per_items
        self.max_vis_per_epoch_train = max_vis_per_epoch_train
        self.max_vis_per_epoch_val = max_vis_per_epoch_val
        self.batch_size = batch_size
        self.symmetry_axis = symmetry_axis
        self.id_list = []

    def forward(self, data, mode='all'):
        depth = data['depth']
        out = self.backbone(depth, data, mode=mode)
        batch_cloth_uv_n = data['batch']
        cloth_uv = data['cloth_uv']
        features = out['feat'][batch_cloth_uv_n, :, cloth_uv[:, 0], cloth_uv[:, 1]]
        flow_inv = out['flow_inv'][batch_cloth_uv_n, :, cloth_uv[:, 0], cloth_uv[:, 1]]

        _, _, h, w = out['flow_inv'].size()

        result = {
            'per_point_features': features,
            'per_point_logits': flow_inv,
            'per_point_batch_idx': batch_cloth_uv_n,
        }
        return result

    def logits_to_nocs(self, logits):
        nocs_bins = self.nocs_bins
        if nocs_bins is None:
            # directly regress from nn
            return logits

        # reshape
        logits_bins = None
        if len(logits.shape) == 2:
            logits_bins = logits.reshape((logits.shape[0], nocs_bins, 3))
        elif len(logits.shape) == 1:
            logits_bins = logits.reshape((nocs_bins, 3))

        bin_idx_pred = torch.argmax(logits_bins, dim=1, keepdim=False)

        # turn into per-channel classification problem
        vg = self.get_virtual_grid()
        points_pred = vg.idxs_to_points(bin_idx_pred)
        return points_pred

    def get_virtual_grid(self):
        nocs_bins = self.nocs_bins
        device = self.device
        vg = VirtualGrid(lower_corner=(0, 0, 0), upper_corner=(1, 1, 1),
                         grid_shape=(nocs_bins,) * 3, batch_size=1,
                         device=device, int_dtype=torch.int64,
                         float_dtype=torch.float32)
        return vg

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def vis_batch(self, batch, nocs_data, batch_idx, is_train=False):
        pred_nocs = nocs_data.pos
        gt_nocs = batch['y']
        batch_idxs = batch['batch']
        this_batch_size = batch['depth'].size(0)
        depth = batch['depth']


        vis_per_items = self.vis_per_items
        batch_size = self.batch_size
        max_vis_per_epoch = None
        prefix = None
        if vis_per_items <= 0:
            return dict()

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

            img = render_nocs_pair(this_gt_nocs, this_pred_nocs)
            img2 = render_nocs_pair(this_gt_nocs, this_pred_nocs, side="top")
            img = np.concatenate([img, img2], axis=0)
            if hasattr(nocs_data, 'pred_confidence'):
                this_pred_confidence = to_numpy(nocs_data.pred_confidence[is_this_item])
                this_x_confidence = this_pred_confidence[:, 0]
                confidence_img = render_confidence_pair(this_gt_nocs, this_pred_nocs, this_x_confidence)
                img = np.concatenate([img, confidence_img], axis=0)
            # this_obs = batch.rgb[i].detach().permute(1,2,0).cpu().numpy()
            log_data[label] = [wandb.Image(img, caption=batch['dataset_idx'][i]),
                               # wandb.Image(this_obs, caption=batch['dataset_idx'][i])
                               ]

        return log_data


    def get_metrics_regression(self, result, batch):
        criterion = self.criterion
        pred_nocs = result['per_point_logits']
        gt_nocs = batch.y
        nocs_loss = criterion(pred_nocs, gt_nocs)

        pred_grip_point = result['global_logits']
        gt_grip_point = batch.nocs_grip_point
        grip_point_loss = criterion(
            pred_grip_point, gt_grip_point)

        nocs_data = Batch(x=result['per_point_features'],
                          pos=pred_nocs, grip_point=pred_grip_point,
                          batch=result['per_point_batch_idx'])

        loss = self.nocs_loss_weight * nocs_loss \
               + self.grip_point_loss_weight * grip_point_loss

        nocs_err_dist = torch.norm(nocs_data.pos - batch.y, dim=-1).mean()
        grip_err_dist = torch.norm(
            nocs_data.grip_point - batch.nocs_grip_point, dim=-1).mean()

        metrics = {
            'loss': loss,
            'nocs_loss': nocs_loss,
            'grip_point_loss': grip_point_loss,
            'nocs_err_dist': nocs_err_dist,
            'grip_point_err_dist': grip_err_dist
        }
        return metrics, nocs_data

    def get_metrics_bin_simple(self, result, batch):
        nocs_bins = self.nocs_bins
        nocs_pred = result['per_point_logits']
        per_point_features = result['per_point_features']
        nocs_tgt = batch['y']
        batch_cloth_uv = batch['batch']

        # Classification loss for canonicalization prediction
        criterion = nn.CrossEntropyLoss()
        # num_sample = self.num_sample
        num_nodes = batch_cloth_uv.shape[0]

        loss = 0
        metrics = {}
        nocs_pred_bins = nocs_pred.view(num_nodes, nocs_bins, 3)  # (num_samplexN,64,3)

        # Compute the loss of every sample and take the min
        vg = self.get_virtual_grid()
        gt_nocs_idx = vg.get_points_grid_idxs(nocs_tgt)  # Nx3
        # gt_nocs_idx = gt_nocs_idx.repeat(num_sample, 1)  # (num_samplexN,3)
        nocs_loss = criterion(nocs_pred_bins, gt_nocs_idx)  # (N,3)

        # compute confidence
        def compute_nocs_dist(cur_nocs_pred_bins):
            nocs_bin_idx_pred = torch.argmax(cur_nocs_pred_bins, dim=1)
            pred_confidence_bins = F.softmax(cur_nocs_pred_bins, dim=1)
            pred_confidence = torch.squeeze(torch.gather(
                pred_confidence_bins, dim=1,
                index=torch.unsqueeze(nocs_bin_idx_pred, dim=1)))

            pred_nocs = vg.idxs_to_points(nocs_bin_idx_pred)
            nocs_err_dist = torch.norm(pred_nocs - nocs_tgt, dim=-1).mean()
            return pred_nocs, nocs_err_dist, pred_confidence

        loss += self.nocs_loss_weight * nocs_loss

        best_pred_nocs, best_nocs_err_dist, best_pred_confidence = compute_nocs_dist(nocs_pred_bins)

        nocs_data = Batch(x=per_point_features,
                          pos=best_pred_nocs,  # grip_point=pred_grip_point,
                          batch=result['per_point_batch_idx'],
                          pred_confidence=best_pred_confidence)

        metrics.update({
            'loss': loss,
            'nocs_loss': nocs_loss,
            'nocs_err_dist': best_nocs_err_dist,
        })

        return metrics, nocs_data

    def get_metrics_bin_symmetry_helper(self, result, batch, mirror_axis=None):
        nocs_bins = self.nocs_bins

        # mirroring
        gt_nocs = batch.y
        gt_grip_point = batch.nocs_grip_point

        if mirror_axis is not None:
            gt_nocs = mirror_nocs_points_by_axis(gt_nocs, axis=mirror_axis)
            gt_grip_point = mirror_nocs_points_by_axis(gt_grip_point, axis=mirror_axis)

        criterion = nn.CrossEntropyLoss()
        vg = self.get_virtual_grid()

        pred_logits = result['per_point_logits']
        pred_logits_bins = pred_logits.reshape(
            (pred_logits.shape[0], nocs_bins, 3))
        gt_nocs_idx = vg.get_points_grid_idxs(gt_nocs)
        nocs_loss = criterion(pred_logits_bins, gt_nocs_idx)

        pred_global_logits = result['global_logits']
        pred_global_bins = pred_global_logits.reshape(
            (pred_global_logits.shape[0], nocs_bins, 3))
        gt_grip_point_idx = vg.get_points_grid_idxs(gt_grip_point)
        grip_point_loss = criterion(
            pred_global_bins, gt_grip_point_idx)

        nocs_bin_idx_pred = torch.argmax(pred_logits_bins, dim=1)
        pred_confidence_bins = F.softmax(pred_logits_bins, dim=1)
        pred_confidence = torch.squeeze(torch.gather(
            pred_confidence_bins, dim=1,
            index=torch.unsqueeze(nocs_bin_idx_pred, dim=1)))
        pred_nocs = vg.idxs_to_points(nocs_bin_idx_pred)
        grip_bin_idx_pred = torch.argmax(pred_global_bins, dim=1)
        pred_grip_point = vg.idxs_to_points(grip_bin_idx_pred)

        nocs_data = Batch(x=result['per_point_features'],
                          pos=pred_nocs, grip_point=pred_grip_point,
                          batch=result['per_point_batch_idx'],
                          pred_confidence=pred_confidence)

        loss = self.nocs_loss_weight * nocs_loss \
               + self.grip_point_loss_weight * grip_point_loss

        nocs_err_dist = torch.norm(pred_nocs - gt_nocs, dim=-1).mean()
        grip_err_dist = torch.norm(
            pred_grip_point - gt_grip_point, dim=-1).mean()

        metrics = {
            'loss': loss,
            'nocs_loss': nocs_loss,
            'grip_point_loss': grip_point_loss,
            'nocs_err_dist': nocs_err_dist,
            'grip_point_err_dist': grip_err_dist
        }
        return metrics, nocs_data

    def get_metrics_bin_symmetry(self, result, batch):
        symmetry_axis = self.symmetry_axis

        normal_metrics, normal_nocs_data = self.get_metrics_bin_symmetry_helper(
            result, batch, mirror_axis=None)
        mirrored_metrics, mirror_nocs_data = self.get_metrics_bin_symmetry_helper(
            result, batch, mirror_axis=symmetry_axis)

        normal_loss = normal_metrics['loss']
        mirrored_loss = mirrored_metrics['loss']
        final_loss = torch.min(normal_loss, mirrored_loss)
        final_metrics = None
        final_nocs_data = None
        if normal_loss <= mirrored_loss:
            final_metrics = normal_metrics
            final_nocs_data = normal_nocs_data
        else:
            final_metrics = mirrored_metrics
            final_nocs_data = mirror_nocs_data
        final_metrics['loss'] = final_loss
        return final_metrics, final_nocs_data

    def infer(self, batch, batch_idx, is_train=True):
        nocs_bins = self.nocs_bins
        symmetry_axis = self.symmetry_axis
        vis_per_items = self.vis_per_items
        result = self(batch)

        metrics, nocs_data = None, None
        if nocs_bins is None:
            metrics, nocs_data = self.get_metrics_regression(result, batch)
        elif symmetry_axis is None:
            metrics, nocs_data = self.get_metrics_bin_simple(result, batch)
        else:
            metrics, nocs_data = self.get_metrics_bin_symmetry(result, batch)
        for key, value in metrics.items():
            log_key = ('train_' if is_train else 'val_') + key
            self.log(log_key, value)
        if vis_per_items > 0 and self.cfg.use_wandb:
            log_data = self.vis_batch(batch, nocs_data, batch_idx, is_train=is_train)
            self.logger.log_metrics(log_data, step=self.global_step)
        return metrics

    def on_train_epoch_start(self) -> None:
        self.id_list = []

    def on_validation_epoch_start(self) -> None:
        self.id_list = []

    def training_step(self, batch, batch_idx):
        metrics = self.infer(batch, batch_idx, is_train=True)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self.infer(batch, batch_idx, is_train=False)
        return metrics['loss']
