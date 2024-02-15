import pdb

import pytorch_lightning as pl
import torch
import torch_scatter

from garmentnets.components.gridding import VirtualGrid
from garmentnets.components.mlp import MLP


class VolumeFeatureAggregator(pl.LightningModule):
    def __init__(self,
                 feat_size=480,
                 nn_channels=128,
                 batch_norm=True,
                 lower_corner=(0, 0, 0),
                 upper_corner=(1, 1, 1),
                 grid_shape=(32, 32, 32),
                 reduce_method='mean',
                 include_point_feature=True,
                 include_confidence_feature=False):
        super().__init__()
        self.save_hyperparameters()
        self.local_nn = MLP([feat_size, feat_size] + [nn_channels,], batch_norm=batch_norm)
        self.lower_corner = tuple(lower_corner)
        self.upper_corner = tuple(upper_corner)
        self.grid_shape = tuple(grid_shape)
        self.reduce_method = reduce_method
        self.include_point_feature = include_point_feature
        self.include_confidence_feature = include_confidence_feature

    def forward(self, nocs_data):
        local_nn = self.local_nn
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        include_point_feature = self.include_point_feature
        include_confidence_feature = self.include_confidence_feature
        reduce_method = self.reduce_method
        batch_size = nocs_data.num_graphs

        sim_points = nocs_data.sim_points  # pixel coordinate
        points = nocs_data.pos
        nocs_features = nocs_data.x
        batch_idx = nocs_data.batch
        confidence = nocs_data.pred_confidence
        device = points.device
        float_dtype = points.dtype
        int_dtype = torch.int64

        vg = VirtualGrid(
            lower_corner=lower_corner,
            upper_corner=upper_corner,
            grid_shape=grid_shape,
            batch_size=batch_size,
            device=device,
            int_dtype=int_dtype,
            float_dtype=float_dtype)

        # get aggregation target index
        points_grid_idxs = vg.get_points_grid_idxs(points, batch_idx=batch_idx)
        flat_idxs = vg.flatten_idxs(points_grid_idxs, keepdim=False)

        # get features
        features_list = [nocs_features]
        if include_point_feature:
            points_grid_points = vg.idxs_to_points(points_grid_idxs)
            local_offset = points - points_grid_points
            features_list.append(local_offset)
            features_list.append(sim_points)

        if include_confidence_feature:
            features_list.append(confidence)
        features = torch.cat(features_list, axis=-1)

        # per-point transform
        if local_nn is not None:
            features = local_nn(features)

        # scatter
        volume_feature_flat = torch_scatter.scatter(
            src=features.T, index=flat_idxs, dim=-1,
            dim_size=vg.num_grids, reduce=reduce_method)

        # reshape to volume
        feature_size = features.shape[-1]
        volume_feature = volume_feature_flat.reshape(
            (feature_size, batch_size) + grid_shape).permute((1, 0, 2, 3, 4))
        return volume_feature


def pointsAggregate(
        nocs_data,
        lower_corner=(0, 0, 0),
        upper_corner=(1, 1, 1),
        grid_shape=(32, 32, 32),
        reduce_method='mean'
):
    batch_size = nocs_data.num_graphs
    # point cloud
    sim_points = nocs_data.sim_points  # pixel coordinate
    # canonical pose
    points = nocs_data.pos
    batch_idx = nocs_data.batch
    confidence = nocs_data.pred_confidence
    device = points.device
    float_dtype = points.dtype
    int_dtype = torch.int64

    vg = VirtualGrid(
        lower_corner=lower_corner,
        upper_corner=upper_corner,
        grid_shape=grid_shape,
        batch_size=batch_size,
        device=device,
        int_dtype=int_dtype,
        float_dtype=float_dtype)

    # get aggregation target index
    points_grid_idxs = vg.get_points_grid_idxs(points, batch_idx=batch_idx)
    flat_idxs = vg.flatten_idxs(points_grid_idxs, keepdim=False)

    # get features
    features = sim_points
    ones = torch.ones((sim_points.shape[0], 1), device=device, dtype=float_dtype)

    # scatter
    volume_feature_flat = torch_scatter.scatter(
        src=features.T, index=flat_idxs, dim=-1,
        dim_size=vg.num_grids, reduce=reduce_method)
    mask_flat = torch_scatter.scatter(
        src=ones.T, index=flat_idxs, dim=-1,
        dim_size=vg.num_grids, reduce=reduce_method)

    # reshape to volume
    feature_size = features.shape[-1]
    volume_feature = volume_feature_flat.reshape(
        (feature_size, batch_size) + grid_shape).permute((1, 0, 2, 3, 4))
    mask = mask_flat.reshape(
        (1, batch_size) + grid_shape).permute((1, 0, 2, 3, 4))

    std_feature = torch_scatter.scatter_std(src=features.T, index=flat_idxs, dim=-1,
                                            dim_size=vg.num_grids)
    std = std_feature[:, mask_flat[0].bool()].mean()
    return volume_feature, mask, std
