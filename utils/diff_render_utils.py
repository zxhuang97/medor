import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    FoVPerspectiveCameras, FoVOrthographicCameras
)
import numpy as np


class MeshRendererWithDepth(nn.Module):
    def __init__(self, cam_height=0.65, device='cuda:0'):
        super().__init__()
        sigma = 1e-10
        R, T = look_at_view_transform(dist=cam_height, elev=90, azim=180)
        camera = FoVPerspectiveCameras(device=device, R=R, T=T, fov=45)
        raster_settings_soft = RasterizationSettings(
            image_size=200,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=1,
            perspective_correct=False,
        )

        self.rasterizer = MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings_soft
        )
        self.shader = SoftSilhouetteShader()

    def forward(self, meshes_world, **kwargs):
        """return rendered images and depth"""
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf[:, :, :, 0]


class PointCloudRendererWithDepth(nn.Module):
    def __init__(self, cam_height=0.65, img_size=200, radius=0.03, pts_px=5, device='cuda:0'):
        super().__init__()
        R, T = look_at_view_transform(dist=cam_height, elev=90, azim=180)
        camera = FoVPerspectiveCameras(device=device, R=R, T=T, fov=45)
        raster_settings = PointsRasterizationSettings(
            image_size=img_size,
            radius=radius,
            points_per_pixel=pts_px
        )
        self.rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)

    def forward(self, meshes_world, **kwargs):
        """return rendered images and depth"""
        fragments = self.rasterizer(meshes_world, **kwargs)
        return fragments


class VisPointCloudCompute(nn.Module):
    def __init__(self, cam_height=0.65, img_size=200, radius=0.03, pts_px=5, device='cuda:0'):
        super().__init__()
        self.renderer = PointCloudRendererWithDepth(cam_height=cam_height,
                                                    img_size=img_size,
                                                    radius=radius,
                                                    pts_px=pts_px,
                                                    device=device)

    def forward(self, pts):
        pc = Pointclouds(points=pts)
        frag = self.renderer(pc)
        idx = frag.idx.long().unique()[1:]
        vis = torch.zeros(np.prod(pts.shape[:2]), device=pts.device, dtype=pts.dtype)
        vis[idx] = 1.
        return vis.view(*pts.shape[:2])


def get_visibility_by_rendering(v, f, sigma=1e-10, image_size=200):
    # return visibility map and depth of each vertex
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=1,
        perspective_correct=False,
    )

    mesh = Meshes(verts=[v], faces=[f])
    packed_verts = mesh.verts_packed()
    visibility_map = torch.zeros(packed_verts.shape[0], dtype=torch.bool)  # (V,)
    #
    # for i in range(4):
    R, T = look_at_view_transform(dist=0.65, elev=90, azim=180)
    camera = FoVPerspectiveCameras(device=v.device, R=R, T=T, fov=45)
    # camera = FoVOrthographicCameras(device=device, znear=0.0, R=R, T=T)
    rasterizer = MeshRasterizer(
        cameras=camera,
        raster_settings=raster_settings
    )

    # Get the output from rasterization
    fragments = rasterizer(mesh)

    # pix_to_face is of shape (N, H, W, 1)
    pix_to_face = fragments.pix_to_face

    # (F, 3) where F is the total number of faces across all the meshes in the batch
    packed_faces = mesh.faces_packed()
    # (V, 3) where V is the total number of verts across all the meshes in the batch
    # Indices of unique visible faces
    visible_faces = pix_to_face.unique()  # (num_visible_faces )

    # Get Indices of unique visible verts using the vertex indices in the faces
    visible_verts_idx = packed_faces[visible_faces]  # (num_visible_faces,  3)
    unique_visible_verts_idx = torch.unique(visible_verts_idx)  # (num_visible_verts, )

    # Update visibility indicator to 1 for all visible vertices
    visibility_map[unique_visible_verts_idx] = True
    depth = fragments.zbuf[:, :, :, 0]
    return visibility_map.numpy(), depth
