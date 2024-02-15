from typing import Optional, Tuple

import igl
import numpy as np
import torch

from scipy.spatial.transform import Rotation as R
from torch import nn

def build_line(length=1.0, num_nodes=50):
    verts = np.zeros((num_nodes, 3), dtype=np.float32)
    verts[:, 0] = np.linspace(0, length, num_nodes)
    edges = np.empty((num_nodes - 1, 2), dtype=np.uint32)
    edges[:, 0] = range(0, num_nodes - 1)
    edges[:, 1] = range(1, num_nodes)
    return verts, edges


def build_rectangle(width=0.45, height=0.32, width_num_node=23, height_num_node=17):
    """
    Row major, row corresponds to width
    """
    # width_num_node = int(np.round(width / grid_size)) + 1
    # height_num_node = int(np.round(height / grid_size)) + 1

    print("Creating a rectangular grid with the following parameters:")
    print("Width:", width)
    print("Height:", height)
    print("W nodes::", width_num_node)
    print("H nodes:", height_num_node)

    def xy_to_index(x_idx, y_idx):
        # Assumes the following layout in imagespace - 0 is to the top left of the image
        #
        #        0          cloth_x_size+0    ...  cloth_y_size*cloth_x_size - cloth_x_size + 0
        #        1          cloth_x_size+1    ...  cloth_y_size*cloth_x_size - cloth_x_size + 1
        #        2          cloth_x_size+2    ...  cloth_y_size*cloth_x_size - cloth_x_size + 2
        #       ...
        #  cloth_x_size-1   cloth_x_size*2-1  ...  cloth_y_size*cloth_x_size - 1
        # return x_idx * width_num_node + y_idx
        return y_idx * height_num_node + x_idx

    verts = np.zeros((width_num_node * height_num_node, 3), dtype=np.float32)
    uv = np.zeros((width_num_node * height_num_node, 2), dtype=np.float32)
    edges_temp = []
    faces_temp = []
    for x in range(height_num_node):
        for y in range(width_num_node):
            curr_idx = xy_to_index(x, y)
            verts[curr_idx, 0] = x * height / (height_num_node - 1)
            verts[curr_idx, 1] = y * width / (width_num_node - 1)
            uv[curr_idx, 0] = x / (height_num_node - 1)
            uv[curr_idx, 1] = y / (width_num_node - 1)

            if x + 1 < height_num_node:
                edges_temp.append([curr_idx, xy_to_index(x + 1, y)])
            if y + 1 < width_num_node:
                edges_temp.append([curr_idx, xy_to_index(x, y + 1)])
            if x + 1 < height_num_node and y + 1 < width_num_node:
                faces_temp.append([curr_idx, xy_to_index(x + 1, y), xy_to_index(x + 1, y + 1), xy_to_index(x, y + 1)])

    edges = np.array(edges_temp, dtype=np.uint32)
    faces = np.array(faces_temp, dtype=np.uint32)
    return verts, edges, faces, uv


def faces_to_edges(faces):
    edges_set = set()
    for face in faces:
        for i in range(1, len(face)):
            edge_pair = (face[i - 1], face[i])
            edge_pair = tuple(sorted(edge_pair))
            edges_set.add(edge_pair)
    edges = np.array(list(edges_set), dtype=np.int)
    return edges


def rotate_particles(angle, pos):
    r = R.from_euler('zyx', angle, degrees=True)
    rot_mat = r.as_matrix()
    if torch.is_tensor(pos):
        new_pos = pos.clone()[:, :3]
        rot_mat = torch.tensor(rot_mat, dtype=pos.dtype, device=pos.device)
    else:
        new_pos = pos.copy()[:, :3]

    center = new_pos.mean(0)
    new_pos -= center
    new_pos = new_pos @ rot_mat
    new_pos += center
    return new_pos


def translate_particles(new_pos, pos):
    """ Translate the cloth so that it lies on the ground with center at pos """
    center = np.mean(pos, axis=0)
    center[1] = np.min(pos, axis=0)[1]
    pos[:, :3] -= center[:3]
    pos[:, :3] += np.asarray(new_pos)
    return pos


def instance_normalize(cloth_nocs):
    aabb = np.stack([cloth_nocs.min(0), cloth_nocs.max(0)])
    center = np.mean(aabb, axis=0)
    edge_lengths = aabb[1] - aabb[0]
    scale = 1 / np.max(edge_lengths + 0.1)
    result_center = np.ones((3,), dtype=aabb.dtype) / 2
    cloth_nocs = (cloth_nocs - center) * scale + result_center
    return cloth_nocs


def nocs_img_normalize(img_nocs, depth, aabb):
    center = np.mean(aabb, axis=0)
    edge_lengths = aabb[1] - aabb[0]
    scale = 1 / np.max(edge_lengths + 0.1)
    result_center = np.ones((3,), dtype=aabb.dtype) / 2
    u, v = depth.nonzero()
    nocs = img_nocs[u, v]
    nocs = (nocs - center) * scale + result_center
    img_nocs[u, v] = nocs
    return img_nocs


def nocs_instance_normalize(cloth_nocs, img_nocs, depth):
    aabb = np.stack([cloth_nocs.min(0), cloth_nocs.max(0)])
    return nocs_img_normalize(img_nocs, depth, aabb)


class AABBNormalizer(nn.Module):
    def __init__(self, aabb, rescale):
        """
        1. Transform a mesh in NOCS space to task space
        2. Transform a T-pose mesh in task space to flatten pose

        Args:
            aabb:
            rescale:
        """

        super().__init__()
        # center = np.mean(aabb, axis=0)
        # edge_lengths = aabb[1] - aabb[0]
        # scale = 1 / np.max(edge_lengths + 0.1)
        # result_center = np.ones((3,), dtype=aabb.dtype) / 2
        # rewrite the code above into torch
        center = torch.mean(aabb, dim=0)
        edge_lengths = aabb[1] - aabb[0]
        scale = 1 / torch.max(edge_lengths + 0.1)
        result_center = torch.ones((3,), dtype=aabb.dtype) / 2
        self.center = nn.Parameter(center, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.result_center = nn.Parameter(result_center, requires_grad=False)

    def __call__(self, data):
        center = self.center
        scale = self.scale
        result_center = self.result_center
        result = (data - center) * scale + result_center
        return result

    def inverse(self, data):
        center = self.center
        scale = self.scale
        result_center = self.result_center

        if not torch.is_tensor(data):
            center = center.detach().cpu().numpy()
            scale = scale.detach().cpu().numpy()
            result_center = result_center.detach().cpu().numpy()


        # if torch.is_tensor(data):
        #     center = torch.tensor(center, dtype=data.dtype, device=data.device)
        #     scale = torch.tensor(scale, dtype=data.dtype, device=data.device)
        #     result_center = torch.tensor(result_center, dtype=data.dtype, device=data.device)
        result = (data - result_center) / scale + center
        return result

    @staticmethod
    def rotate_on_table(data, angle):
        # 1. Flip the mesh
        new_pos = rotate_particles(angle, data)

        # 2. Translate the garment so that it lies on the ground and centers at O
        center = new_pos.mean(0)
        center[1] = new_pos.min(0)[1] - 0.005
        new_pos[:, :3] -= center[:3]
        return new_pos

    # def flat_canon_linear(self, data, align=False):
    #     # TODO: batch operation
    #     data = rotate_particles([180, 180, 0], data)
    #     # apply pre-computed linear transformation to obtain flat pose
    #     scale = np.array(self.to_canon['scale']).reshape(1, 3)
    #     bias = np.array(self.to_canon['bias']).reshape(1, 3)
    #
    #     if torch.is_tensor(data):
    #         scale = torch.tensor(scale, dtype=data.dtype, device=data.device)
    #         bias = torch.tensor(bias, dtype=data.dtype, device=data.device)
    #     new_pos = data * scale + bias
    #
    #     center = new_pos.mean(0)
    #     if torch.is_tensor(data):
    #         center[1] = new_pos.min(0)[0][1] - 0.005
    #     else:
    #         center[1] = new_pos.min(0)[1] - 0.005
    #     new_pos[:, :3] -= center[:3]
    #     return new_pos




def get_aabb(coords):
    """
    Axis Aligned Bounding Box
    Input:
    coords: (N, C) array
    Output:
    aabb: (2, C) array
    """
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    aabb = np.stack([min_coords, max_coords])
    return aabb


def buffer_aabb(aabb, buffer):
    result_aabb = aabb.copy()
    result_aabb[0] -= buffer
    result_aabb[1] += buffer
    return result_aabb


def quads2tris(quads):
    assert (isinstance(quads, np.ndarray))
    assert (len(quads.shape) == 2)
    assert (quads.shape[1] == 4)

    # allocate new array
    tris = np.zeros((quads.shape[0] * 2, 3), dtype=quads.dtype)
    tris[0::2] = quads[:, [0, 1, 2]]
    tris[1::2] = quads[:, [0, 2, 3]]
    return tris


def barycentric_interpolation(query_coords: np.array, verts: np.array, faces: np.array) -> np.array:
    """
    Input:
    query_coords: np.array[M, 3] float barycentric coorindates
    verts: np.array[N, 3] float vertecies
    faces: np.array[M, 3] int face index into verts, 1:1 coorespondace to query_coords

    Output
    result: np.array[M, 3] float interpolated points
    """
    assert (len(verts.shape) == 2)
    result = np.zeros((len(query_coords), verts.shape[1]), dtype=verts.dtype)
    for c in range(verts.shape[1]):
        for i in range(query_coords.shape[1]):
            result[:, c] += query_coords[:, i] * verts[:, c][faces[:, i]]
    return result


def mesh_sample_barycentric(
        verts: np.ndarray, faces: np.ndarray,
        num_samples: int, seed: Optional[int] = None,
        face_areas: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample points (as their barycentric coordinate) on suface

    Input:
    verts: np.array[N, 3] float mesh vertecies
    faces: np.array[M, 3] int mesh face index into verts
    num_sampels: int
    seed: int random seed
    face_areas: np.array[M, 3] per-face areas

    Output:
    barycentric_all: np.array[num_samples, 3] float sampled barycentric coordinates
    selected_face_idx: np.array[num_samples,3] int sampled faces, 1:1 coorespondance to barycentric_all
    """
    # generate face area
    if face_areas is None:
        face_areas = igl.doublearea(verts, faces)
    face_areas = face_areas / np.sum(face_areas)
    assert (len(face_areas) == len(faces))

    rs = np.random.RandomState(seed=seed)
    # select faces
    selected_face_idx = rs.choice(
        len(faces), size=num_samples,
        replace=True, p=face_areas).astype(faces.dtype)

    # generate random barycentric coordinate
    barycentric_uv = rs.uniform(0, 1, size=(num_samples, 2))
    not_triangle = (np.sum(barycentric_uv, axis=1) >= 1)
    barycentric_uv[not_triangle] = 1 - barycentric_uv[not_triangle]

    barycentric_all = np.zeros((num_samples, 3), dtype=barycentric_uv.dtype)
    barycentric_all[:, :2] = barycentric_uv
    barycentric_all[:, 2] = 1 - np.sum(barycentric_uv, axis=1)

    return barycentric_all, selected_face_idx
