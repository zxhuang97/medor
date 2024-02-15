import numpy as np


import trimesh
from scipy import spatial
from scipy.spatial.ckdtree import cKDTree
import igl
import open3d as o3d

from menpo3d.barycentric import barycentric_coordinates_of_pointcloud, \
    barycentric_coordinate_interpolation
from menpo.shape.mesh.base import TriMesh, PointCloud
from utils.camera_utils import intrinsic_from_fov, get_matrix_world_to_camera


def query_uv_barycentric(query_uv, target_uv_verts, target_uv_faces):
    """
    Input:
    query_uv: (V, 2) float array for query uv points
    target_uv_verts: (V, 2) float array for target uv vertex
    target_uv_faces: (F, 3) int array for uv face vertices

    Output:
    barycentric: (V, 3) float array for projected barycentric coordinate
    proj_face_idx: (V,) int array for cooresponding face index
    """
    assert (target_uv_faces.shape[1] == 3)
    uv_verts_3d = np.zeros((len(target_uv_verts), 3), dtype=target_uv_verts.dtype)
    uv_verts_3d[:, :2] = target_uv_verts

    ray_origins = np.zeros((len(query_uv), 3), dtype=query_uv.dtype)
    ray_origins[:, :2] = query_uv
    ray_origins[:, 2] = 1
    ray_directions = np.zeros((len(query_uv), 3), dtype=query_uv.dtype)
    ray_directions[:, 2] = -1

    mesh = trimesh.Trimesh(vertices=uv_verts_3d, faces=target_uv_faces, use_embree=True)
    intersector = mesh.ray
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins, ray_directions, multiple_hits=False)

    # convert to query_uv index
    proj_face_idx = np.full(len(query_uv), fill_value=-1, dtype=np.int64)
    proj_face_idx[index_ray] = index_tri

    # get barycentric coordinates
    corners = list()
    for i in range(3):
        corners.append(uv_verts_3d[target_uv_faces[index_tri, i]].astype(np.float64))
    barycentric = np.full((len(query_uv), 3), fill_value=-1, dtype=target_uv_verts.dtype)
    barycentric[index_ray] = igl.barycentric_coordinates_tri(locations.astype(np.float64), *corners)

    # in case of miss
    if len(index_ray) < len(query_uv):
        miss_idxs = np.nonzero(proj_face_idx == -1)[0]
        kdtree = cKDTree(target_uv_verts)
        _, nn_vert_idxs = kdtree.query(query_uv[miss_idxs], k=1)

        vert_face_idx_map = np.zeros(len(target_uv_verts), dtype=np.int64)
        face_idxs = np.tile(np.arange(len(target_uv_faces)), (3, 1)).T
        vert_face_idx_map[target_uv_faces] = face_idxs

        miss_face_idxs = vert_face_idx_map[nn_vert_idxs]
        proj_face_idx[miss_idxs] = miss_face_idxs
        for i in range(len(miss_idxs)):
            miss_face = target_uv_faces[miss_face_idxs[i]]
            is_vertex = (miss_face == nn_vert_idxs[i])
            assert (is_vertex.sum() == 1)
            this_bc = is_vertex.astype(barycentric.dtype)
            barycentric[miss_idxs[i]] = this_bc

    return barycentric, proj_face_idx


def barycentric_interpolation(query_coords, target_verts, target_faces):
    result = np.zeros((len(query_coords), 3), dtype=target_verts.dtype)
    for c in range(target_verts.shape[1]):
        for i in range(query_coords.shape[1]):
            result[:, c] += \
                query_coords[:, i] * target_verts[:, c][target_faces[:, i]]
    return result


def get_barycentric_pc(pc, coords, faces, ptr_attrs):
    """
    Register noisy pointcloud to the mesh and compute the barycentric coefficients
    Args:
        pc:
        coords:
        faces:
        ptr_attrs: a dictionary that store the quantity to be interpolated
    Returns:

    """
    PC = PointCloud(pc)
    mesh = TriMesh(coords, trilist=faces)
    bc, proj_face_ind = barycentric_coordinates_of_pointcloud(mesh, PC)
    out = {}
    for k, v in ptr_attrs.items():
        out[k] = barycentric_coordinate_interpolation(mesh, v, bc, proj_face_ind)
    return out


def get_world_coords(rgb, depth, matrix_world_to_camera=None, env=None):
    height, width = depth.shape[:2]
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    # Apply back-projection: K_inv @ pixels * depth
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x = np.linspace(0, width - 1, width).astype(np.float)
    y = np.linspace(0, height - 1, height).astype(np.float)
    u, v = np.meshgrid(x, y)
    one = np.ones((height, width, 1))
    x = (u - u0) * depth / fx
    y = (v - v0) * depth / fy
    z = depth
    cam_coords = np.dstack([x, y, z, one])
    if matrix_world_to_camera is None:
        matrix_world_to_camera = get_matrix_world_to_camera(
            env.camera_params[env.camera_name]['pos'], env.camera_params[env.camera_name]['angle'])
    # convert the camera coordinate back to the world coordinate using the rotation and translation matrix
    cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)
    world_coords = np.linalg.inv(matrix_world_to_camera) @ cam_coords  # 4 x (height x width)
    world_coords = world_coords.transpose().reshape((height, width, 4))

    return world_coords


def get_edges_from_tri(fs):
    mesh_edges = set()
    for f in fs:
        mesh_edges.add(tuple(sorted((f[0], f[1]))))
        mesh_edges.add(tuple(sorted((f[0], f[2]))))
        mesh_edges.add(tuple(sorted((f[1], f[2]))))
    mesh_edges = np.stack(list(mesh_edges), 0).astype(np.long)
    return mesh_edges


def mesh_downsampling(v, f, voxel_size=0.025):
    v = v
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                  o3d.utility.Vector3iVector(f))
    mesh_smp = m.simplify_vertex_clustering(voxel_size=voxel_size)
    ds_v, ds_f = np.array(mesh_smp.vertices), np.array(mesh_smp.triangles)
    pair_dis = spatial.distance.cdist(ds_v, v)
    nn = pair_dis.argmin(1)

    return nn, ds_f


def mesh_downsampling_decimation(v, f, num_f=5000):
    v = v
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                  o3d.utility.Vector3iVector(f))
    mesh_smp = m.simplify_quadric_decimation(target_number_of_triangles=num_f)
    ds_v, ds_f = np.array(mesh_smp.vertices), np.array(mesh_smp.triangles)
    pair_dis = spatial.distance.cdist(ds_v, v)
    nn = pair_dis.argmin(1)
    return nn, ds_f


# def voxelize_pointcloud(pointcloud, voxel_size):
#     cloud = pcl.PointCloud(pointcloud)
#     sor = cloud.make_voxel_grid_filter()
#     sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
#     pointcloud = sor.filter()
#     pointcloud = np.asarray(pointcloud)
#     return pointcloud


def quads2tris(F):
    out = []
    for f in F:
        if len(f) == 3:
            out += [f]
        elif len(f) == 4:
            out += [[f[0], f[1], f[2]],
                    [f[0], f[2], f[3]]]
        else:
            print("This should not happen...")
    return np.array(out, np.int32)


def readOBJ(file, to_tri=True):
    V, Vt, F, Ft = [], [], [], []
    with open(file, 'r') as f:
        T = f.readlines()
    for t in T:
        # 3D vertex
        if t.startswith('v '):
            v = [float(n) for n in t.replace('v ', '').split(' ')]
            V += [v]
        # UV vertex
        elif t.startswith('vt '):
            v = [float(n) for n in t.replace('vt ', '').split(' ')]
            Vt += [v]
        # Face
        elif t.startswith('f '):
            idx = [n.split('/') for n in t.replace('f ', '').split(' ')]
            f = [int(n[0]) - 1 for n in idx]
            F += [f]
            # UV face
            if '/' in t:
                f = [int(n[1]) - 1 for n in idx]
                Ft += [f]
    V = np.array(V, np.float32)
    Vt = np.array(Vt, np.float32)
    if Ft:
        assert len(F) == len(
            Ft), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces'
    else:
        Vt, Ft = None, None
    if to_tri:
        F = quads2tris(F)
        if Ft:
            Ft = quads2tris(Ft)
    return V, F, Vt, Ft
