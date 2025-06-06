import pdb

import cv2
import numpy as np
import scipy.optimize as opt
import scipy
from scipy import spatial
from scipy.spatial.distance import cdist


#
#
# def build_depth_from_pointcloud(pointcloud, matrix_world_to_camera, imsize):
#     height, width = imsize
#     pointcloud = np.concatenate([pointcloud, np.ones((len(pointcloud), 1))], axis=1)  # n x 4
#     camera_coordinate = matrix_world_to_camera @ pointcloud.T  # 3 x n
#     camera_coordinate = camera_coordinate.T  # n x 3
#     K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees
#
#     u0 = K[0, 2]
#     v0 = K[1, 2]
#     fx = K[0, 0]
#     fy = K[1, 1]
#
#     x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
#     u = np.rint((x * fx / depth + u0).astype("int"))
#     v = np.rint((y * fy / depth + v0).astype("int"))
#
#     us = u.flatten()
#     vs = v.flatten()
#     depth = depth.flatten()
#
#     depth_map = dict()
#     for u, v, d in zip(us, vs, depth):
#         if depth_map.get((u, v)) is None:
#             depth_map[(u, v)] = []
#             depth_map[(u, v)].append(d)
#         else:
#             depth_map[(u, v)].append(d)
#
#     depth_2d = np.zeros((height, width))
#     for u in range(width):
#         for v in range(height):
#             if (u, v) in depth_map.keys():
#                 depth_2d[v][u] = np.min(depth_map[(u, v)])
#
#     return depth_2d

#
# def pixel_coord_np(width, height):
#     """
#     Pixel in homogenous coordinate
#     Returns:
#         Pixel coordinate:       [3, width * height]
#     """
#     x = np.linspace(0, width - 1, width).astype(np.int)
#     y = np.linspace(0, height - 1, height).astype(np.int)
#     [x, y] = np.meshgrid(x, y)
#     return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])


def get_rotation_matrix(angle, axis):
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle)
    c = np.cos(angle)

    m = np.zeros((4, 4))

    m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
    m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
    m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
    m[0][3] = 0.0

    m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
    m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
    m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
    m[1][3] = 0.0

    m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
    m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
    m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
    m[2][3] = 0.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 0.0
    m[3][3] = 1.0

    return m


def uv_to_world_pos(u, v, z, K, matrix_world_to_camera):
    x0 = K[0, 2]
    y0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    one = np.ones(u.shape, np.float32)

    x = (v - x0) * z / fx
    y = (u - y0) * z / fy
    cam_coords = np.stack([x, y, z, one], axis=1)
    cam2world = np.linalg.inv(matrix_world_to_camera).T
    world_coords = cam_coords @ cam2world
    return world_coords[:, :3].astype(np.float32)


def get_pointcloud(depth, matrix_world_to_camera, get_img=False):
    height, width = depth.shape
    u, v = depth.nonzero()
    z = depth[u, v]
    K = intrinsic_from_fov(height, width, 45).astype(np.float32)  # the fov is 90 degrees
    pc = uv_to_world_pos(u, v, z, K, matrix_world_to_camera)
    if get_img:
        img_pc = np.zeros((height, width, 3), dtype=np.float32)
        img_pc[u, v] = pc
        return pc, img_pc
    return pc


#
# def get_world_coords(rgb, depth, env, particle_pos=None):
#     height, width, _ = rgb.shape
#     K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees
#
#     # Apply back-projection: K_inv @ pixels * depth
#     u0 = K[0, 2]
#     v0 = K[1, 2]
#     fx = K[0, 0]
#     fy = K[1, 1]
#
#     x = np.linspace(0, width - 1, width).astype(np.float)
#     y = np.linspace(0, height - 1, height).astype(np.float)
#     u, v = np.meshgrid(x, y)
#     one = np.ones((height, width, 1))
#     x = (u - u0) * depth / fx
#     y = (v - v0) * depth / fy
#     z = depth
#     cam_coords = np.dstack([x, y, z, one])
#
#     matrix_world_to_camera = get_matrix_world_to_camera(
#         env.camera_params[env.camera_name])
#
#     # convert the camera coordinate back to the world coordinate using the rotation and translation matrix
#     cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)
#     world_coords = np.linalg.inv(matrix_world_to_camera) @ cam_coords  # 4 x (height x width)
#     world_coords = world_coords.transpose().reshape((height, width, 4))
#
#     return world_coords

#
def get_observable_particle_index(pointcloud, particle_pos):
    # height, width, _ = rgb.shape
    # perform the matching of pixel particle to real particle
    particle_pos = particle_pos[:, :3]

    distance = scipy.spatial.distance.cdist(pointcloud, particle_pos)
    # Each point in the point cloud will cover at most two particles.
    # Particles not covered will be deemed occluded
    estimated_particle_idx = np.argpartition(distance, 2)[:, :1].flatten()
    estimated_particle_idx = np.unique(estimated_particle_idx)

    return np.array(estimated_particle_idx, dtype=np.int32)


# def get_observable_particle_index_old(world_coords, particle_pos, rgb, depth):
#     height, width, _ = rgb.shape
#     # perform the matching of pixel particle to real particle
#     particle_pos = particle_pos[:, :3]
#
#     estimated_world_coords = np.array(world_coords)[np.where(depth > 0)][:, :3]
#
#     distance = scipy.spatial.distance.cdist(estimated_world_coords, particle_pos)
#     estimated_particle_idx = np.argmin(distance, axis=1)
#     estimated_particle_idx = np.unique(estimated_particle_idx)
#
#     return np.array(estimated_particle_idx, dtype=np.int32)


# def get_observable_particle_index_3(pointcloud, mesh, threshold=0.0216):
#     ### bi-partite graph matching
#     distance = scipy.spatial.distance.cdist(pointcloud, mesh)
#     distance[distance > threshold] = 1e10
#     row_idx, column_idx = opt.linear_sum_assignment(distance)
#
#     distance_mapped = distance[np.arange(len(pointcloud)), column_idx]
#     bad_mapping = distance_mapped > threshold
#     if np.sum(bad_mapping) > 0:
#         column_idx[bad_mapping] = np.argmin(distance[bad_mapping], axis=1)
#
#     return pointcloud, column_idx


def get_mapping_from_pointcloud_to_partile_nearest_neighbor(pointcloud, particle):
    distance = scipy.spatial.distance.cdist(pointcloud, particle)
    nearest_idx = np.argmin(distance, axis=1)
    return nearest_idx


#
# def get_observable_particle_index_4(pointcloud, mesh, threshold=0.0216):
#     # perform the matching of pixel particle to real particle
#     estimated_world_coords = pointcloud
#
#     distance = scipy.spatial.distance.cdist(estimated_world_coords, mesh)
#     estimated_particle_idx = np.argmin(distance, axis=1)
#
#     return pointcloud, np.array(estimated_particle_idx, dtype=np.int32)
#
#
# def get_observable_particle_pos(world_coords, particle_pos):
#     # perform the matching of pixel particle to real particle
#     particle_pos = particle_pos[:, :3]
#     distance = scipy.spatial.distance.cdist(world_coords, particle_pos)
#     estimated_particle_idx = np.argmin(distance, axis=1)
#     observable_particle_pos = particle_pos[estimated_particle_idx]
#
#     return observable_particle_pos


def get_matrix_world_to_camera(camera_param=None, cam_pos=None, cam_angle=None):
    if camera_param is None:
        cam_x, cam_y, cam_z = cam_pos
        cam_x_angle, cam_y_angle, cam_z_angle = cam_angle
    else:
        cam_x, cam_y, cam_z = camera_param['pos']
        cam_x_angle, cam_y_angle, cam_z_angle = camera_param['angle']
    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.zeros((4, 4))
    translation_matrix[0][0] = 1
    translation_matrix[1][1] = 1
    translation_matrix[2][2] = 1
    translation_matrix[3][3] = 1
    translation_matrix[0][3] = - cam_x
    translation_matrix[1][3] = - cam_y
    translation_matrix[2][3] = - cam_z

    return rotation_matrix @ translation_matrix


def project_to_image(matrix_world_to_camera, world_coordinate, height=360, width=360):
    world_coordinate = np.concatenate([world_coordinate, np.ones((len(world_coordinate), 1))],
                                      axis=1)  # n x 4
    camera_coordinate = matrix_world_to_camera @ world_coordinate.T  # 3 x n
    camera_coordinate = camera_coordinate.T  # n x 3
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    u = (x * fx / depth + u0).astype("int")
    v = (y * fy / depth + v0).astype("int")

    return u, v



def _get_depth(matrix, vec, height):
    """ Get the depth such that the back-projected point has a fixed height"""
    return (height - matrix[1, 3]) / (vec[0] * matrix[1, 0] + vec[1] * matrix[1, 1] + matrix[1, 2])

def get_world_coor_from_image(u, v, matrix_world_to_camera, all_depth, depth=None):
    height, width = all_depth.shape
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    matrix = np.linalg.inv(matrix_world_to_camera)

    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    if depth is None:
        depth = all_depth[v][u]
        if depth == 0:
            vec = ((u - u0) / fx, (v - v0) / fy)
            depth = _get_depth(matrix, vec, 0.005)  # Height to be the particle radius

    # Loop through each pixel in the image
    # Apply equation in fig 3
    x = (u - u0) * depth / fx
    y = (v - v0) * depth / fy
    z = depth
    cam_coords = np.array([x, y, z, 1])
    cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)

    world_coord = matrix @ cam_coords  # 4 x (height x width)
    world_coord = world_coord.reshape(4)
    return world_coord[:3]


def get_target_pos(pos, u, v, image_size, matrix_world_to_camera, depth, verts_vis=None,
                   pick_vis_only=True):
    coor = get_world_coor_from_image(u, v, matrix_world_to_camera, depth)
    dists = cdist(coor[None], pos)[0]
    sorted_idx = np.argsort(dists)
    idx = np.argmin(dists)
    if pick_vis_only and verts_vis is not None:
        for i in sorted_idx:
            if verts_vis[i]:
                idx = i
                break
    return pos[idx] + np.array([0, 0.01, 0])


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


def get_visible(matrix_world_to_camera, coords, depth):
    """Get knots that are visible in the depth image.

    Returns
        vis: bool list of length knots indicating visibility (1 visible, 0 occluded)
    """

    if depth.shape[0] < 720:
        depth = cv2.resize(depth, (720, 720), interpolation=cv2.INTER_NEAREST)
    cloth_mask = depth > 0.1
    depth_uv = np.stack(cloth_mask.nonzero(), 1)

    # pdb.set_trace()
    depth_z = depth[depth_uv[:, 0], depth_uv[:, 1]]
    h, w = depth.shape
    K = intrinsic_from_fov(h, w, 45)
    world_coords = uv_to_world_pos(depth_uv[:, 0], depth_uv[:, 1], depth_z, K, matrix_world_to_camera)
    # vis_n = get_observable_particle_index(world_coords, coords)
    vis_n = get_observable_particle_index2(world_coords, coords)
    vis_n = vis_n.reshape(-1, 1)
    return vis_n
