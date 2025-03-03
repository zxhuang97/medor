import pdb

import imageio
import numpy as np
import os

import zarr as zarr
from numcodecs import Blosc
from scipy import spatial
from scipy.spatial.ckdtree import cKDTree
# import iglpo
import cv2
import scipy
import glob
from menpo3d.barycentric import barycentric_coordinates_of_pointcloud, \
    barycentric_coordinate_interpolation
from menpo.shape.mesh.base import TriMesh, PointCloud
import h5py
from joblib import Parallel, delayed
from multiprocessing import Pool
import tqdm
from utils.cloth3d.DataReader.IO import readOBJ, writeOBJ
from utils.cloth3d.DataReader.util import quads2tris
from utils.data_utils import read_h5_dict, store_h5_dict
from utils.camera_utils import get_matrix_world_to_camera, intrinsic_from_fov
from garmentnets.datasets.conv_implicit_wnf_dataset import GarmentnetsDataloader
from utils.geometry_utils import query_uv_barycentric, get_world_coords
import torch

# global root

camera_params = {'default_camera': {'pos': np.array([-0.0, 0.65, 0.0]),
                                    'angle': np.array([0, -np.pi / 2., 0.]),
                                    'width': 720,
                                    'height': 720},
                 'top_down_camera3': {
                     'pos': np.array([0, 1.3, 0]),
                     'angle': np.array([0, -90 / 180 * np.pi, 0]),
                     'width': 720,
                     'height': 720
                 }}
camera_params = camera_params['default_camera']
matrix_world_to_camera = get_matrix_world_to_camera(camera_params)


def get_pointcloud(depth, cam_mat):
    world_coordinates = get_world_coords(depth, depth, matrix_world_to_camera=cam_mat)
    world_coords = world_coordinates[:, :, :3].reshape((-1, 3))
    pointcloud_positions = world_coords[depth.flatten() > 0]
    return pointcloud_positions


def uv_to_world_pos(u, v, z):
    height, width = 720, 720
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees
    x0 = K[0, 2]
    y0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    matrix_world_to_camera = get_matrix_world_to_camera(camera_params)
    one = np.ones(u.shape)

    x = (v - x0) * z / fx
    y = (u - y0) * z / fy
    cam_coords = np.stack([x, y, z, one], axis=1)
    cam2world = np.linalg.inv(matrix_world_to_camera).T
    world_coords = cam_coords @ cam2world
    return world_coords


def get_observable_particle_index(world_coords, particle_pos):
    particle_pos = particle_pos[:, :3]
    estimated_world_coords = world_coords[:, :3]
    distance = scipy.spatial.distance.cdist(estimated_world_coords, particle_pos)
    estimated_particle_idx = np.argmin(distance, axis=1)
    estimated_particle_idx = np.unique(estimated_particle_idx)
    vis_n = np.zeros(particle_pos.shape[0])
    vis_p = np.array(estimated_particle_idx, dtype=np.int32)
    vis_n[vis_p] = 1
    return vis_n


def get_observable_particle_index2(world_coords, particle_pos):
    # perform the matching of pixel particle to real particle
    particle_pos = particle_pos[:, :3]
    estimated_world_coords = world_coords[:, :3]
    # distance = scipy.spatial.distance.cdist(estimated_world_coords, particle_pos)
    # estimated_particle_idx = np.argmin(distance, axis=1)
    tree = spatial.cKDTree(particle_pos)
    _, estimated_particle_idx = tree.query(estimated_world_coords, k=1)
    estimated_particle_idx = np.unique(estimated_particle_idx)
    vis_n = np.zeros(particle_pos.shape[0])
    vis_p = np.array(estimated_particle_idx, dtype=np.int32)
    vis_n[vis_p] = 1
    return vis_n


def visibility2(base_data, index):
    coords_f = np.load(f'{base_data}/coords/{str(index).zfill(6)}_coords_after.npy')
    depth_f = np.load(f'{base_data}/rendered_images/{str(index).zfill(6)}_depth_after.npy')
    uv_n_f = np.load(f'{base_data}/knots/{str(index).zfill(6)}_knots_after.npy')
    if depth_f.shape[0] < 720:
        depth_f = cv2.resize(depth_f, (720, 720), interpolation=cv2.INTER_NEAREST)
    cloth_mask = depth_f > 0.1
    depth_uv = np.stack(cloth_mask.nonzero(), 1)
    depth_z = depth_f[depth_uv[:, 0], depth_uv[:, 1]]
    world_coords = uv_to_world_pos(depth_uv[:, 0], depth_uv[:, 1], depth_z)
    vis_n = get_observable_particle_index2(world_coords, coords_f)
    vis_n = vis_n.astype(np.float)
    coords_f[:, 3] = vis_n
    np.save(f'{base_data}/coords/{str(index).zfill(6)}_coords_after.npy', coords_f)


def get_barycentric_cloth_pc(pc, coords, faces, canon_nocs):
    #     print(pc.shape, coords.shape, faces.shape, canon_nocs.shape)
    PC = PointCloud(pc)
    mesh = TriMesh(coords, trilist=faces)
    bc, proj_face_ind = barycentric_coordinates_of_pointcloud(mesh, PC)
    pc_nocs = barycentric_coordinate_interpolation(mesh, canon_nocs, bc, proj_face_ind)
    return pc_nocs


data_names = ["cloth_id", "flat_depth", "canon_flat_coords"]
CLOTH_3D_PATH = "dataset/cloth3d"


def curate_cloth(args):
    """
    Compute the following items for dataset:
    1. visibility
    2. pc_nocs
    3. nocs
    4. wnf
    todo
    4. masks of two layers
    5. barycentric for flat canon(to get flow and coordinate prediction for each vertex)
    """
    # global root
    base_data, index, ctype = args
    data_path = os.path.join(base_data, 'data', f'{index:05d}_3d.h5')
    # pdb.set_trace()
    # if os.path.exists(data_path):
    #     return
    # else:
    #     pc = np.load(f'{base_data}/pointcloud/{str(index).zfill(6)}_pc_after.npy')
    index_c = index - index % 5
    # index_c = index
    canon_dict = read_h5_dict(f"{base_data}/canon/{index_c:06d}.h5")
    # out = read_h5_dict(f"{base_data}/canon/{index:05d}_3d.h5")
    # if 'cloth_nocs_verts' in canon_dict:
    #     return
    cloth_id = canon_dict['cloth_id']
    meta_path = os.path.join(base_data, 'nocs', f'{cloth_id:05d}_3d.h5')
    # x1 = read_h5_dict(os.path.join(base_data, 'data', f'{index:05d}_3d.h5'))
    # meta_path = os.path.join(base_data, 'nocs', f'{cloth_id:05d}_3d.h5')
    # x2 = read_h5_dict(meta_path)
    # if 'cloth_id' in x1:
    #     return
    # if x1 is not None and x2 is not None:
    #     return
    coords = np.load(f'{base_data}/coords/{str(index).zfill(6)}_coords_after.npy')[:, :3]
    depth_f = np.load(f'{base_data}/rendered_images/{str(index).zfill(6)}_depth_after.npy')
    rgb = imageio.imread(f'{base_data}/images/{str(index).zfill(6)}_rgb_after.png')

    # if not os.path.exists(f'{base_data}/pointcloud/{str(index).zfill(6)}_pc_after.npy'):
    pc = get_pointcloud(depth_f, matrix_world_to_camera)
    # print(cloth_id)
    # nocs, wnf = gdloader.get_sample(cloth_id.item())
    # ["downsample_id", "triangles", "mesh_edges", "nocs", "wnf"]
    nocs_data = read_h5_dict(f"{CLOTH_3D_PATH}/{ctype}/nocs/{cloth_id:04d}_info.h5",
                             data_names=['nocs', 'wnf'])

    nocs = nocs_data['nocs']
    wnf = nocs_data['wnf']

    _, F = readOBJ(f"{CLOTH_3D_PATH}/{ctype}/mesh/{cloth_id:04d}.obj")[:2]
    F = quads2tris(F)
    # pdb.set_trace()
    pc_nocs = get_barycentric_cloth_pc(pc, coords, F, nocs)

    all_uvs = np.array(depth_f.nonzero()).T
    canon_img = np.zeros((720, 720, 3), dtype=np.float32)
    canon_img[all_uvs[:, 0], all_uvs[:, 1]] = pc_nocs
    canon_img_rs = cv2.resize(canon_img, (200, 200), interpolation=cv2.INTER_NEAREST)

    depth = cv2.resize(depth_f, (200, 200), interpolation=cv2.INTER_NEAREST)
    img_pc = np.zeros((200, 200, 3), dtype=np.float32)
    ds_pc = get_pointcloud(depth, matrix_world_to_camera)
    img_pc[depth > 0] = ds_pc

    out = {'rgb': rgb,
           'depth': depth, 'img_pc': img_pc, 'img_nocs': canon_img_rs,
           'pc_sim': pc, 'pc_nocs': pc_nocs,
           "cloth_sim_verts": coords,
           "cloth_id": cloth_id, }
    if not os.path.exists(data_path):
        store_h5_dict(data_path, out)

    if not os.path.exists(meta_path):
        out2 = {
            'cloth_nocs_verts': nocs,
            'cloth_faces_tri': F,
            'wnf': wnf,
        }
        store_h5_dict(meta_path, out2)
    # nocs_data_old = read_h5_dict(meta_path)
    # for k, v in nocs_data_old.items():
    #     assert np.sum(np.abs(v - out2[k])) == 0, f"nocs data mismatch {k}"
    #
    # pdb.set_trace()

    # misc_group = root.require_group('misc', overwrite=True)


print('done')
cloth_types = ["Trousers", 'Dress', 'Skirt', 'Tshirt', 'Jumpsuit']
cloth_types = ['Tshirt']

for cloth_type in cloth_types:
    gdloader = GarmentnetsDataloader(cloth_type)
    for p in ['train']:
    # for p in ['test']:
        print(f'Working on {cloth_type}   {p}')
        if p == 'train':
            num = 20000
            # num = 100
        elif p == 'val':
            num = 2000
            # num = 10
        elif p == 'test':
            num = 40
        base_data = f'dataset/{cloth_type}_dataset_v2/{p}'
        # base_data = f'dataset2/{cloth_type}_hard_v4'

        os.makedirs(os.path.join(base_data, 'data'), exist_ok=True)
        os.makedirs(os.path.join(base_data, 'nocs'), exist_ok=True)
        store_h5_dict(os.path.join(base_data, "summary_new.h5"), {
            "nocs_aabb": gdloader.nocs_aabb,
            "len": num,
            'pos': np.array([-0.0, 0.65, 0.0]),
            'angle': np.array([0, -np.pi / 2., 0.]),
            'width': 720,
            'height': 720,
        })
        batch = [(base_data, i, cloth_type) for i in range(num)]
        # batch = batch[:]
        results = Parallel(n_jobs=32, verbose=True)(delayed(curate_cloth)(param) for param in batch)
        # for i, b in enumerate(tqdm.tqdm(batch)):
        #     curate_cloth(b)
