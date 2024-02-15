import pdb

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.ops import ball_query, taubin_smoothing
from scipy.spatial import ckdtree

from utils.diff_render_utils import MeshRendererWithDepth, get_visibility_by_rendering
from utils.loss_utils import iou_loss, chamfer_distance_my


class IterChamferScipy:
    def __init__(self, tgt_points, ksize=1):
        # print(tgt_points.device)
        self.tgt_points = tgt_points
        self.tgt_points_np = tgt_points.detach().cpu().numpy()

        self.ksize = ksize

    def query(self, pred_points):
        pred_tree = ckdtree.cKDTree(pred_points.detach().cpu().numpy())
        _, forward_nn_idx = pred_tree.query(self.tgt_points_np, k=self.ksize)
        forward_distance = torch.mean((pred_points[forward_nn_idx] - self.tgt_points) ** 2, -1)

        return forward_distance


def compute_edge_loss(pts, edge, rest_len, margin=0):
    edge_norm = torch.norm(pts[edge[:, 0]] - pts[edge[:, 1]], dim=-1)
    edge_diff = torch.abs(edge_norm - rest_len)
    edge_diff = torch.relu(edge_diff - margin) ** 2
    return edge_diff.mean()


class TestTimeFinetuner(object):
    def __init__(self, finetune_cfg):
        self.cfg = finetune_cfg
        self.chamfer3d_w = self.cfg.chamfer3d_w

        self.laplacian_w = self.cfg.laplacian_w
        self.normal_w = self.cfg.normal_w
        self.rigid_w = self.cfg.get('rigid_w', 0)
        self.edge_w = self.cfg.edge_w
        self.rest_edge_len = self.cfg.rest_edge_len
        self.rest_edge_margin = self.cfg.rest_edge_margin

        self.depth_w = self.cfg.depth_w
        self.silhouette_w = self.cfg.silhouette_w

        self.obs_consist_w = self.cfg.obs_consist_w
        self.consist_iter = self.cfg.consist_iter

        self.table_w = self.cfg.table_w
        self.collision_w = self.cfg.get('collision_w', 0)
        self.collision_radius = self.cfg.get('collision_radius', 0.004)

        if self.depth_w > 0 or self.silhouette_w > 0:
            self.renderer = MeshRendererWithDepth()
        self.depth = None
        self.mask = None
        self.pointcloud = None
        self.pred_pos = None
        self.pred_f = None
        self.pred_mesh = None
        self.pred_edge = None
        self.optim = None
        self.obs_results = None
        self.deform_verts = None
        self.chamfer_calculator = None
        self.opt_reset = None

    def register_new_state(self, depth, pointcloud, pred_pos, pred_f, pred_edge, obs_results=None,
                           model_params=None):
        self.opt_reset = False
        self.depth = depth[0]
        self.mask = self.depth > 0
        self.pointcloud = pointcloud
        self.pred_pos = pred_pos
        # self.pred_pos.requires_grad = True
        self.pred_f = pred_f.long()
        self.pred_edge = pred_edge
        self.obs_results = obs_results
        self.pred_mesh = Meshes(verts=[self.pred_pos.detach()], faces=[self.pred_f])
        verts_shape = self.pred_mesh.verts_packed().shape
        self.deform_verts = torch.full(verts_shape, 0.0, device=pred_pos.device, requires_grad=True)
        params_to_optim = [self.deform_verts]
        self.optim = optim.Adam(params_to_optim, lr=self.cfg.lr)

        if self.cfg.chamfer_mode != 'faiss' and self.chamfer_calculator is None:
            pointcloud.requires_grad = False
            self.chamfer_calculator = IterChamferScipy(pointcloud, 1)

    def step(self, opt_iter, optimize=True):
        """
        Compute the loss, and optimize
        Args:
            pred_pos: Nx3
            edges: Ex2

        Returns:

        """
        loss = 0
        results = {}
        # Optimize a delta vector instead of the vertices
        cur_mesh = self.pred_mesh.offset_verts(self.deform_verts)
        src_pts = cur_mesh.verts_packed()
        faces = cur_mesh.faces_packed()

        if self.obs_consist_w > 0 and opt_iter == self.consist_iter and not self.opt_reset:
            self.opt_reset = True
            self.optim = optim.Adam([self.deform_verts], self.cfg.lr)
        if self.chamfer3d_w > 0:
            if self.cfg.chamfer_mode == 'bidirect':
                v_vis, _ = get_visibility_by_rendering(src_pts, faces)
                vis_verts = src_pts[v_vis]
                chamfer3d_loss_pts, _ = chamfer_distance_my(x=vis_verts.unsqueeze(0),
                                                            y=self.pointcloud.unsqueeze(0),
                                                            K=1)
            else:
                v_vis, _ = get_visibility_by_rendering(src_pts, faces)
                vis_verts = src_pts[v_vis]
                chamfer3d_loss_pts = self.chamfer_calculator.query(vis_verts.squeeze(0))
            chamfer3d_loss_mean = chamfer3d_loss_pts.mean()
            results['chamfer3d_loss_mean'] = chamfer3d_loss_mean
            loss += self.chamfer3d_w * chamfer3d_loss_mean

        if self.rigid_w > 0:
            rigid_loss = (self.deform_verts[self.pred_edge[:, 0]] - self.deform_verts[
                self.pred_edge[:, 1]]) ** 2
            rigid_loss = rigid_loss.mean()
            results['rigid_loss'] = rigid_loss
            loss += self.rigid_w * rigid_loss

        if self.collision_w > 0:
            dists = ball_query(src_pts.unsqueeze(0),
                               src_pts.unsqueeze(0),
                               K=5,
                               radius=self.collision_radius,
                               return_nn=False
                               ).dists
            collision_loss = -dists.mean()
            results['collision_loss'] = collision_loss
            loss += self.collision_w * collision_loss

        if self.laplacian_w > 0:
            laplacian_loss = mesh_laplacian_smoothing(cur_mesh)
            results['laplacian_loss'] = laplacian_loss
            loss += self.laplacian_w * laplacian_loss
        if self.normal_w > 0:
            normal_loss = mesh_normal_consistency(cur_mesh)
            results['normal_loss'] = normal_loss
            loss += self.normal_w * normal_loss
        if self.edge_w > 0:
            # edge_loss = mesh_edge_loss(cur_mesh, self.rest_edge_len)
            edge_loss = compute_edge_loss(src_pts, self.pred_edge, self.rest_edge_len,
                                          self.rest_edge_margin)
            results['edge_loss'] = edge_loss
            loss += self.edge_w * edge_loss
        if self.depth_w > 0 or self.silhouette_w > 0:
            images, render_depth = self.renderer(cur_mesh)
            self.depth[self.depth == 0] = -1
            # render_depth[render_depth==-1] = 0
            render_mask = images[:, :, :, 3]
            if self.depth_w > 0:
                mask = (self.depth > -1).detach() * (render_depth > -1).detach()
                depth_loss = F.mse_loss(render_depth, self.depth, reduction='none')
                depth_loss = depth_loss * mask
                depth_loss = depth_loss.sum() / mask.sum()
                results['depth_loss'] = depth_loss
                loss += self.depth_w * depth_loss
            if self.silhouette_w > 0:
                silhouette_loss = iou_loss(self.mask, render_mask)
                if silhouette_loss.item() > 0.0:
                    results['silhouette_loss'] = silhouette_loss
                    loss += self.silhouette_w * silhouette_loss
        if self.obs_consist_w > 0:
            mask = self.obs_results['mask_pts'][:, None]
            obs_pts = self.obs_results['obs_pts']
            std = self.obs_results['std']
            obs_consist_loss = F.mse_loss(src_pts, obs_pts, reduction='none')
            obs_consist_loss = obs_consist_loss * mask
            obs_consist_loss = obs_consist_loss.sum() / mask.sum()
            results['obs_consist_loss'] = obs_consist_loss
            if opt_iter < self.consist_iter:
                loss += self.obs_consist_w * obs_consist_loss
        if self.table_w > 0:
            height = cur_mesh.verts_packed()[:, 1]
            table_loss = (torch.relu(0.005 - height)) ** 2
            table_loss = table_loss.mean()
            results['table_loss'] = table_loss
            loss += self.table_w * table_loss

        if optimize:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        results['loss'] = loss
        results['new_pos'] = cur_mesh.verts_packed()
        return results
