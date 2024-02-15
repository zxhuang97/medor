import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import fps, radius, global_max_pool, knn_interpolate
from torch_geometric.nn.conv import PointConv
import torch.nn.functional as F
from garmentnets.components.mlp import MLP as MLP2

class SAModule(torch.nn.Module):
    """
    Local Set Abstraction (convolution)
    """

    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        # pointnet1 module
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        # furtherest point sampling
        edge_index = None
        with autocast(enabled=False):
            idx = fps(pos, batch, ratio=self.ratio)
            # ball query
            row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                              max_num_neighbors=64)
            edge_index = torch.stack([col, row], dim=0)
            x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    """
    Global Set Abstraction
    """

    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        with autocast(enabled=False):
            # TODO
            x = x.type(torch.float32)
            x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class FPModule(torch.nn.Module):
    """
    Feature Propogation (deconvolution)
    """

    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        with autocast(enabled=False):
            x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
            if x_skip is not None:
                x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointnetDecoder(torch.nn.Module):
    def __init__(self, in_feat_dim=128, sa1_ratio=0.5, sa1_r=0.05, sa2_ratio=0.25, sa2_r=0.1,
                 fp3_k=1, fp2_k=3, fp1_k=3, feature_dim=128, batch_norm=True, dropout=True, use_mlp=False):
        super().__init__()
        self.sa1_module = SAModule(
            sa1_ratio, sa1_r,
            MLP([3 + in_feat_dim, 128, 128, 128], batch_norm=batch_norm))
        self.sa2_module = SAModule(
            sa2_ratio, sa2_r,
            MLP([128 + 3, 128, 128, 256], batch_norm=batch_norm))
        self.sa3_module = GlobalSAModule(
            nn=MLP([256 + 3, 256, 512, 1024], batch_norm=batch_norm))

        self.fp3_module = FPModule(
            k=fp3_k,
            nn=MLP([1024 + 256, 256, 256], batch_norm=batch_norm))
        self.fp2_module = FPModule(
            k=fp2_k,
            nn=MLP([256 + 128, 256, 128], batch_norm=batch_norm))
        self.fp1_module = FPModule(
            k=fp1_k,
            nn=MLP([128 + in_feat_dim, 128, 128, 128], batch_norm=batch_norm))
        self.use_mlp = use_mlp
        # per-point prediction
        output_dim = 3
        if not use_mlp:
            self.lin1 = nn.Linear(128, 128)
            self.lin2 = nn.Linear(128, feature_dim)
            self.lin3 = nn.Linear(feature_dim, output_dim)

            self.dp1 = nn.Dropout(p=0.5, inplace=False) if dropout else lambda x: x
            self.dp2 = nn.Dropout(p=0.5, inplace=False) if dropout else lambda x: x
        else:
            self.mlp = MLP2([128, 128, 128, output_dim])
        self.batch = torch.arange(24, device='cuda:0').unsqueeze(1).expand(-1, 6000).flatten()


    def forward(self, features_grid, query_points, init_pts=None, extend=True):
        query_points_normalized = 2.0 * query_points - 1.0
        sampled_features = F.grid_sample(
            input=features_grid,
            grid=query_points_normalized.view(
                *(query_points_normalized.shape[:2] + (1, 1, 3))),
            mode='bilinear', padding_mode='border',
            align_corners=True)
        sampled_features = sampled_features.view(
            sampled_features.shape[:3]).permute(0, 2, 1)
        N, M, C = sampled_features.shape
        # batch = torch.arange(N, device=sampled_features.device).unsqueeze(1).expand(-1, M).flatten()
        batch = self.batch[:N*6000]
        sa0_out = (sampled_features.reshape(N * M, -1), query_points.view(N * M, -1), batch)
        # sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        # pre-point prediction
        if self.use_mlp:
            logits = self.mlp(x)
        else:
            x = F.relu(self.lin1(x))
            x = self.dp1(x)
            x = self.lin2(x)
            features = self.dp2(x)
            logits = self.lin3(features)
        # import pdb;pdb.set_trace()

        # # global prediction. Removed because we don't need grasp point prediction
        # global_feature, _, _ = sa3_out
        # # x = F.relu(global_feature)
        # # x = self.global_dp1(x)
        # # x = self.global_lin1(x)
        # # x = self.global_dp2(x)
        # # global_logits = self.global_lin2(x)
        #
        # result = {
        #     'per_point_features': features,
        #     'per_point_logits': logits,
        #     'per_point_batch_idx': data.batch,
        #     # 'global_logits': global_logits,
        #     'global_feature': global_feature
        # }
        return logits.view(N, 1, M, -1)
