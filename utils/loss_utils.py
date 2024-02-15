from typing import Union

import numpy as np
import torch
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops import knn_points
from scipy.spatial import ckdtree
from torch.nn import functional as F

from utils.my_decor import auto_numpy
from utils.diff_render_utils import PointCloudRendererWithDepth, get_visibility_by_rendering

from pytorch3d.structures import Pointclouds


def my_huber(pred, gt, w=1, beta=0.01):
    """

    Args:
        pred: N, K, M, 3
        gt:  N, 1, M, 3
        beta:

    Returns:

    """
    tmp_loss = F.smooth_l1_loss(pred, gt, reduction='none', beta=beta) * w
    return tmp_loss


def iou_loss(pred, gt, eps=1e-6):
    dims = tuple(range(pred.ndimension())[1:])
    intersect = (pred * gt).sum(dims)
    union = (pred + gt - pred * gt).sum(dims) + eps
    loss = 1 - (intersect / union).sum() / intersect.nelement()
    return loss

class RenderingLoss(torch.nn.Module):
    def __init__(self):
        super(RenderingLoss, self).__init__()
        self.pc_renderer = PointCloudRendererWithDepth(radius=0.02)

    def forward(self, pred_pts, depth):
        pred_pc = Pointclouds(points=pred_pts)
        fragments = self.pc_renderer(pred_pc)
        render_depth = fragments.zbuf[:, :, :, 0]
        mask = (render_depth > 0).detach() * (depth > 0).detach()
        loss = (render_depth - depth) ** 2 * mask
        return loss.sum() / mask.sum()


@auto_numpy
def one_way_chamfer_numpy(pred_points, tgt_points, ksize=1):
    pred_tree = ckdtree.cKDTree(pred_points)
    _, forward_nn_idx = pred_tree.query(tgt_points, k=ksize)
    forward_distance = np.mean((pred_points[forward_nn_idx] - tgt_points) ** 2)

    return forward_distance.item()


def one_way_chamfer_vis(v, f, pc_tgt, ksize=1):
    if not torch.is_tensor(v):
        v = torch.from_numpy(v).cuda().float()
    if not torch.is_tensor(f):
        f = torch.tensor(f, dtype=torch.long, device=v.device)
    v_vis, _ = get_visibility_by_rendering(v, f)
    v = v.detach().cpu().numpy()
    if torch.is_tensor(pc_tgt):
        pc_tgt = pc_tgt.detach().cpu().numpy()
    pred_tree = ckdtree.cKDTree(v)
    _, forward_nn_idx = pred_tree.query(pc_tgt, k=ksize)
    forward_distance = np.mean((v[forward_nn_idx] - pc_tgt) ** 2)
    return forward_distance


def two_way_chamfer_vis(v, f, pc_tgt, ksize=1):
    if not torch.is_tensor(v):
        v = torch.from_numpy(v).cuda().float()
    if not torch.is_tensor(pc_tgt):
        pc_tgt = torch.from_numpy(pc_tgt).cuda().float()
    if not torch.is_tensor(f):
        f = torch.tensor(f, dtype=torch.long, device=v.device)
    v_vis, _ = get_visibility_by_rendering(v, f)
    vis_verts = v[v_vis]
    chamfer_dists, _ = chamfer_distance_my(x=vis_verts.unsqueeze(0),
                                           y=pc_tgt.unsqueeze(0),
                                           K=ksize)
    return chamfer_dists.mean().detach().cpu().item()


def chamfer_distance_my(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
        K=1
):
    # _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=K)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=K)

    cham_x = x_nn.dists.mean(-1)
    cham_y = y_nn.dists.mean(-1)
    # cham_x = x_nn.dists[..., 0]  # (N, P1)
    # cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if point_reduction == "none":
        return cham_x, cham_y

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals


