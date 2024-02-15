import json
import os
import random
import socket
from collections import namedtuple

import cv2
import h5py
import numpy as np
import pyflex
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from softgym.envs.bimanual_env import BimanualEnv

from utils.camera_utils import get_pointcloud, get_matrix_world_to_camera, intrinsic_from_fov, get_rotation_matrix

Experience = namedtuple('Experience', ('obs', 'goal', 'act', 'rew', 'nobs', 'done'))



def get_harris(mask, thresh=0.2):
    """Harris corner detector
    Params
    ------
        - mask: np.float32 image of 0.0 and 1.0
        - thresh: threshold for filtering small harris values    Returns
    -------
        - harris: np.float32 array of
    """
    # Params for cornerHarris:
    # mask - Input image, it should be grayscale and float32 type.
    # blockSize - It is the size of neighbourhood considered for corner detection
    # ksize - Aperture parameter of Sobel derivative used.
    # k - Harris detector free parameter in the equation.
    # https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345
    harris = cv2.cornerHarris(mask, blockSize=5, ksize=5, k=0.01)
    harris[harris < thresh * harris.max()] = 0.0  # filter small values
    harris[harris != 0] = 1.0
    harris_dilated = cv2.dilate(harris, kernel=np.ones((7, 7), np.uint8))
    harris_dilated[mask == 0] = 0
    return harris_dilated

def random_sample_from_masked_image(img_mask, num_samples, min_r=None, max_r=None):
    """
    Samples num_samples (row, column) convention pixel locations from the masked image
    Note this is not in (u,v) format, but in same format as img_mask
    :param img_mask: numpy.ndarray
        - masked image, we will select from the non-zero entries
        - shape is H x W
    :param num_samples: int
        - number of random indices to return
    :return: List of np.array
    """
    sampled_idx_list = []
    if max_r is not None:
        idx_tuple = img_mask.nonzero()
        num_nonzero = len(idx_tuple[0])
        if num_nonzero == 0:
            return []
        ind = random.sample(range(0, num_nonzero), 1)
        a = idx_tuple[0][ind]
        b = idx_tuple[1][ind]
        first = [a, b]
        h, w = img_mask.shape
        y, x = np.ogrid[-a:h - a, -b:w - b]
        min_mask = x * x + y * y <= min_r * min_r
        max_mask = x * x + y * y > max_r * max_r
        img_mask[min_mask] = 0
        img_mask[max_mask] = 0
        num_samples -= 1
        # cv2.imshow("img",img_mask)
        # cv2.waitKey(0)

    idx_tuple = img_mask.nonzero()
    num_nonzero = len(idx_tuple[0])
    if num_nonzero == 0:
        empty_list = []
        return empty_list
    rand_inds = random.sample(range(0, num_nonzero), num_samples)

    for i, idx in enumerate(idx_tuple):
        if max_r is not None:
            sampled_idx_list.append(np.append(first[i], idx[rand_inds]))
        else:
            sampled_idx_list.append(idx[rand_inds])

    # print(sampled_idx_list)

    return sampled_idx_list

def particle_uv_pos(camera_params, idx=None, particle_pos=None):
    # from cam coord to world coord
    if 'default_camera' in camera_params:
        cam_x, cam_y, cam_z = camera_params['default_camera']['pos'][0], camera_params['default_camera']['pos'][1], \
                              camera_params['default_camera']['pos'][2]
        cam_x_angle, cam_y_angle, cam_z_angle = camera_params['default_camera']['angle'][0], \
                                                camera_params['default_camera']['angle'][1], \
                                                camera_params['default_camera']['angle'][2]
        K = intrinsic_from_fov(camera_params['default_camera']['height'], camera_params['default_camera']['width'], 45)
    else:
        cam_x, cam_y, cam_z = camera_params['pos'][0], camera_params['pos'][1], camera_params['pos'][2]
        cam_x_angle, cam_y_angle, cam_z_angle = camera_params['angle'][0], camera_params['angle'][1],camera_params['angle'][2]
        K = intrinsic_from_fov(camera_params['height'], camera_params['width'], 45)
    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # print("rot mat:\n",rotation_matrix)

    # get translation matrix: from world to camera
    translation_matrix = np.eye(4)
    translation_matrix[0][3] = - cam_x
    translation_matrix[1][3] = - cam_y
    translation_matrix[2][3] = - cam_z

    # print("trans mat:\n",translation_matrix)

    if particle_pos is None:
        particle_pos = pyflex.get_positions().reshape(-1, 4)
    if idx is not None:
        world_coord = np.ones(4)
        world_coord[:3] = particle_pos[idx][:3]
        x, y, z, _ = rotation_matrix @ translation_matrix @ world_coord

        # print("Intrinsic: \n",K)
        x0 = K[0, 2]
        y0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        v = (x * fx / z) + x0
        u = (y * fy / z) + y0

        return u, v
    else:
        world_coord = np.ones((particle_pos.shape[0], 4))
        world_coord[:, :3] = particle_pos[:, :3]
        cam_coord = rotation_matrix @ translation_matrix @ world_coord.T  # 4 x N

        x0 = K[0, 2]
        y0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        uv = np.zeros((particle_pos.shape[0], 2))
        uv[:, 1] = ((cam_coord[0] * fx) / cam_coord[2]) + x0  # v
        uv[:, 0] = ((cam_coord[1] * fy) / cam_coord[2]) + y0  # u

        return uv

def rotate_particles(angle, pos):
    r = R.from_euler('zyx', angle, degrees=True)
    center = np.ones((3,)) / 2
    pos -= center
    new_pos = pos.copy()[:, :3]
    new_pos = r.apply(new_pos)
    new_pos += center
    return new_pos


def to_nocs(data, aabb):
    center = np.mean(aabb, axis=0)
    edge_lengths = aabb[1] - aabb[0]
    scale = 1 / np.max(edge_lengths + 0.1)
    result_center = np.ones((3,), dtype=aabb.dtype) / 2
    # normalizer = AABBNormalizer(aabb)
    result = (data - center) * scale + result_center
    return result


class DatasetGenerator(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        # self.gn_data = GarmentnetsDataloader(cfgs['cloth_type'])
        print('?????? ', cfgs['cloth_type'])
        self.env = BimanualEnv(camera_name=cfgs['camera_name'],
                               cloth_type=cfgs['cloth_type'],
                               use_depth=cfgs['img_type'] == 'depth',
                               use_cached_states=False,
                               render=True,
                               horizon=cfgs['horizon'],
                               use_desc=False,
                               action_repeat=1,
                               camera_width=cfgs['cam_size'],
                               camera_height=cfgs['cam_size'],
                               headless=cfgs['headless'],
                               rect=cfgs['rect'],
                               remove_invalid=cfgs['remove_invalid'])
        self.env.eval_flag = cfgs['eval_flag']
        self.matrix_world_to_camera = get_matrix_world_to_camera(
            self.env.config['camera_params'][self.env.config['camera_name']])
        print("env created")

        # self.em = EdgeMasker(self.env, cfgs['cloth_type'], tshirtmap_path=None, edgethresh=cfgs['edgethresh'])

    def makedirs(self):
        save_folder = os.path.join(self.cfgs['dataset_folder'], self.cfgs['dataset_name'])
        if self.cfgs['debug']:
            os.system('rm -r %s' % save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            os.makedirs(os.path.join(save_folder, 'images'))
            os.makedirs(os.path.join(save_folder, 'coords'))
            os.makedirs(os.path.join(save_folder, 'pointcloud'))
            os.makedirs(os.path.join(save_folder, 'canon'))
            os.makedirs(os.path.join(save_folder, 'rendered_images'))
            os.makedirs(os.path.join(save_folder, 'knots'))
            os.makedirs(os.path.join(save_folder, 'edge_masks'))
        return save_folder

    def get_masked(self, img):
        """Just used for masking goals, otherwise we use depth"""
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, np.array([0., 15., 0.]), np.array([255, 255., 255.]))
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return morph

    def get_rgbd(self):
        """ returns rgb, depth and mask
        """
        img = self.env.get_image(self.env.camera_height, self.env.camera_width)
        rgbd = self.env.get_rgbd2(show_picker=False)
        depth = rgbd[:, :, 3]
        mask = depth > 0
        return img, depth, mask

    def line_pt_dir(self, a, b, p):
        ax, ay = a
        bx, by = b
        px, py = p

        bx -= ax
        by -= ay
        px -= ax
        py -= ay

        cross_prod = bx * py - by * px

        # right of line
        if cross_prod > 0:
            return 1

        # left of line
        if cross_prod < 0:
            return -1

        # on the line
        return 0

    def get_rand_action(self, img, depth, edgemasks, coords, action_type='pickplace', single_scale=True,
                        debug_idx=None, canon_flat_uv=None):
        """ returns random action for given mode
        params::
        action_type: qnet, pickplace, debug
        """
        small_depth = cv2.resize(depth, (200, 200), interpolation=cv2.INTER_NEAREST)
        clothmask = small_depth > 0
        if np.random.uniform() < self.cfgs['actmaskprob']:
            if self.cfgs['use_corner']:
                harris_corners = get_harris(clothmask.astype(np.float32))
                true_corners = self.get_true_corner_mask(clothmask, canon_flat_uv)
                if np.sum(true_corners) > 2 and np.random.uniform() < self.cfgs['truecratio']:
                    mask = true_corners > 0
                elif np.sum(harris_corners) > 2:
                    mask = harris_corners > 0
                else:
                    mask = clothmask
            else:  # deprecated
                all_mask, fge_mask, ce_mask = edgemasks

                if np.sum(ce_mask != 0) > 2 and np.random.uniform() < self.cfgs[
                    'cemaskratio']:  # sample from cloth edge mask
                    mask = ce_mask > 0
                else:  # sample from fg edge mask
                    mask = fge_mask > 0
        else:  # Cloth mask
            mask = clothmask

        if action_type == 'qnet':
            # sample until valid action found
            while True:
                pick_idx = random_sample_from_masked_image(mask, 1)
                u, v = pick_idx[0][0], pick_idx[1][0]
                angle_idx = np.random.randint(0, 8)
                angle = np.deg2rad(angle_idx * 45)
                width_idx = np.random.randint(3)
                width = width_idx * 25.0

                pick_u1 = int(np.clip(u + np.sin(angle) * width, 10, 190))
                pick_v1 = int(np.clip(v + np.cos(angle) * width, 10, 190))
                pick_u2 = int(np.clip(u - np.sin(angle) * width, 10, 190))
                pick_v2 = int(np.clip(v - np.cos(angle) * width, 10, 190))

                if mask[pick_u1, pick_v1] and mask[pick_u2, pick_v2]:
                    break

            print(f"qnet act: {u},{v} angle: {angle} width: {width}")
            print(f"u1,v1 {pick_u1, pick_v1} u2,v2 {pick_u2, pick_v2}")

            # fold toward center
            if self.line_pt_dir([pick_u1, pick_v1], [pick_u2, pick_v2], [100, 100]) < 0:
                fold_dir = angle - (np.pi / 2)
            else:
                fold_dir = angle + (np.pi / 2)

            # sample fold length
            dist = np.random.uniform(25, 75)
            place_u1 = int(np.clip(pick_u1 + dist * np.sin(fold_dir), 10, 190))
            place_v1 = int(np.clip(pick_v1 + dist * np.cos(fold_dir), 10, 190))
            place_u2 = int(np.clip(pick_u2 + dist * np.sin(fold_dir), 10, 190))
            place_v2 = int(np.clip(pick_v2 + dist * np.cos(fold_dir), 10, 190))

            pick1 = [pick_u1, pick_v1]
            place1 = [place_u1, place_v1]
            pick2 = [pick_u2, pick_v2]
            place2 = [place_u2, place_v2]

            return np.array([angle_idx, width_idx, u, v]), np.array([pick1, place1, pick2, place2])

        if action_type == 'pickplace':
            # returns two arrays of x, and y positions with num_pick number of values
            pick_idx = random_sample_from_masked_image(mask, 2)
            pick_u1, pick_v1 = pick_idx[0][0], pick_idx[1][0]
            pick_u2, pick_v2 = pick_idx[0][1], pick_idx[1][1]

            angle = np.arctan2(pick_u1 - pick_u2, pick_v1 - pick_v2)
            # fold toward center
            if self.line_pt_dir([pick_u1, pick_v1], [pick_u2, pick_v2], [100, 100]) < 0:
                angle -= (np.pi / 2)
            else:
                angle += (np.pi / 2)

            dist = np.random.uniform(25, self.cfgs['max_act'])  # default: 25,100

            dist1 = dist2 = dist

            place_u1 = int(np.clip(pick_u1 + dist1 * np.sin(angle), 25, 175))
            place_v1 = int(np.clip(pick_v1 + dist1 * np.cos(angle), 25, 175))
            place_u2 = int(np.clip(pick_u2 + dist2 * np.sin(angle), 25, 175))
            place_v2 = int(np.clip(pick_v2 + dist2 * np.cos(angle), 25, 175))

            pick1 = [pick_u1, pick_v1]
            place1 = [place_u1, place_v1]
            pick2 = [pick_u2, pick_v2]
            place2 = [place_u2, place_v2]

            print(f"angle: {np.rad2deg(angle)} dist: {dist1} {dist2}")

            return np.array([pick1, place1, pick2, place2]), np.array([pick1, place1, pick2, place2])

    def store_data(self, data_names, data, path):
        hf = h5py.File(path, 'w')
        for i in range(len(data_names)):
            hf.create_dataset(data_names[i], data=data[i])
        hf.close()

    def save_canon_data(self, idx, cloth_id, dataset_path,
                        flat_depth, canon_flat_coords):
        data_names = ["cloth_id", "flat_depth", "canon_flat_coords"]
        self.store_data(data_names,
                        [cloth_id, flat_depth, canon_flat_coords],
                        os.path.join(dataset_path, 'canon', "%06d.h5" % idx))

    def save_data(self, idx, coords, img, depth, dataset_path, beforeact=False):
        save_time = 'before' if beforeact else 'after'
        # all_mask, fge_mask, ce_mask = edgemasks
        # if uv is None:
        #     uv = td.particle_uv_pos(self.env.camera_params[self.env.camera_name], None)
        #     uv[:, [1, 0]] = uv[:, [0, 1]]
        rgb_img = Image.fromarray(img, 'RGB')
        rgb_img.save(os.path.join(dataset_path, 'images', '%06d_rgb_%s.png' % (idx, save_time)))
        # cv2.imwrite(os.path.join(dataset_path, 'rendered_images', '%06d_depth_%s.png' % (idx, save_time)), depth * 250)

        np.save(os.path.join(dataset_path, 'rendered_images', '%06d_depth_%s.npy' % (idx, save_time)), depth)
        np.save(os.path.join(dataset_path, 'coords', '%06d_coords_%s.npy' % (idx, save_time)), coords)
        # np.save(os.path.join(dataset_path, 'knots', '%06d_knots_%s.npy' % (idx, save_time)), uv)
        # if pointcloud is not None:
        #     np.save(os.path.join(dataset_path, 'pointcloud', '%06d_pc_%s.npy' % (idx, save_time)),
        #             pointcloud)

    def get_obs(self):
        coords = pyflex.get_positions().reshape(-1, 4)
        img, depth, mask = self.get_rgbd()

        if self.cfgs['use_corner']:
            edgemasks = None
        else:
            all_mask, fge_mask, ce_mask = self.em.get_act_mask(self.env, coords, img, depth, mask)
            edgemasks = (all_mask, fge_mask, ce_mask)
        return coords, img, depth, edgemasks

    def generate(self):
        min_reward = 0
        max_reward = -10000

        # load goals
        goals = []
        for g in self.cfgs['goals']:
            if g is not None:
                if self.cfgs['img_type'] == 'color':
                    goal = cv2.imread(f"../goals/{g}.png")
                    goal = cv2.cvtColor(goal, cv2.COLOR_BGR2RGB)
                    goal_mask = self.get_masked(goal) != 0
                    goal[goal_mask == False, :] = 0
                elif self.cfgs['img_type'] == 'depth':
                    goal = cv2.imread(f"../goals/{g}_depth.png")
                elif self.cfgs['img_type'] == 'desc':
                    goal = cv2.imread(f"../goals/{g}_desc.png")

                goal_pos = np.load('../goals/particles/{}.npy'.format(g))[:, :3]
            else:
                goal = g
                goal_pos = None
            goals.append([goal, goal_pos])

        save_folder = self.makedirs()
        # buf = []

        # check if dataset exist`s to resume

        idx_buf = []  # buffer with only indexes, no images
        idx = 0
        ep = 0
        id = None
        while ep < self.cfgs['num_eps']:
            print("ep ", ep)
            goal, goal_pos = random.choice(goals)

            # reset and save canonical info
            obs, id = self.env.reset(given_goal=goal, given_goal_pos=goal_pos, id=id)
            canon_flat_coords, canon_img, canon_depth, _ = self.get_obs()
            canon_flat_uv = particle_uv_pos(self.env.camera_params[self.env.camera_name],
                                               particle_pos=canon_flat_coords)
            # canon_flat_uv = td.particle_uv_pos(self.env.camera_params[self.env.camera_name],
            #                                    particle_pos=canon_flat_coords)
            self.save_canon_data(idx, id, save_folder, canon_depth, canon_flat_coords,
                                 )

            done = False
            while not done:
                coords, img, depth, edgemasks = self.get_obs()
                print(idx)
                if self.cfgs['use_drop'] and idx % 5 == 0 and np.random.rand() > 0.5:
                    self.env.lift_and_drop(render=False)
                    print('random drop')
                else:
                    buf_act, action = self.get_rand_action(img, depth, edgemasks, coords,
                                                           action_type=self.cfgs['action_type'], debug_idx=ep,
                                                           canon_flat_uv=canon_flat_uv)
                    next_state, reward, done, info = self.env.step(action,
                                                                   record_continuous_video=self.cfgs['video'],
                                                                   pickplace=True,
                                                                   on_table=self.cfgs['on_table'])
                coords_next, img_next, depth_next, _ = self.get_obs()

                uv = particle_uv_pos(self.env.camera_params[self.env.camera_name], None)
                # uv[:, [1, 0]] = uv[:, [0, 1]]
                out_range = np.logical_or(np.logical_or(uv[:, 0] < 0, uv[:, 1] < 0),
                                          np.logical_or(uv[:, 0] >= 719, uv[:, 1] >= 719))
                uv_valid = ~out_range
                # check if out of screen
                if not np.all(uv_valid):
                    print('out table need roll back')
                    # self.env.reset(given_goal=goal, given_goal_pos=goal_pos, id=id)
                    idx = idx - (idx % cfgs['horizon'])
                    ep -= 1
                    done = False
                    break

                # pointcloud = get_pointcloud(depth_next, self.matrix_world_to_camera)
                # print(f"size of point cloud  ", pointcloud.shape[0])
                #
                # # 3D NOCS Nx3
                # cloth_nocs_certs = canon_nocs_v[nn]

                # todo
                self.save_data(idx, coords_next, img_next, depth_next, save_folder,
                               beforeact=False)

                # state = copy.deepcopy(next_state)
                # self.env.render(mode='rgb_array')
                idx += 1
                if idx % 5 == 0:
                    id = None

            # if (ep % 500) == 0:
            #     print("saving...")
            #     # torch.save(buf, os.path.join(save_folder,f'{self.cfgs["dataset_name"]}.buf'))
            #     torch.save(idx_buf, os.path.join(save_folder, f'{self.cfgs["dataset_name"]}_idx.buf'))

            ep += 1

        # torch.save(buf, os.path.join(save_folder,f'{self.cfgs["dataset_name"]}.buf'))
        # print("saving...")
        # torch.save(idx_buf, os.path.join(save_folder, f'{self.cfgs["dataset_name"]}_idx.buf'))

        # create knots info
        # print("create knots info...")
        # knots = os.listdir(os.path.join(save_folder, 'knots'))
        # knots.sort()
        # kdict = {}
        # for i, name in enumerate(knots):
        #     knot = np.load(os.path.join(save_folder, 'knots', name))
        #     knot = np.reshape(knot, (knot.shape[0], 1, knot.shape[1]))
        #     kdict[str(i)] = knot.tolist()
        # with open(os.path.join(save_folder, 'images', 'knots_info.json'), 'w') as f:
        #     json.dump(kdict, f)

        # print(f"min reward: {min_reward}, max reward: {max_reward}")
        # np.save(os.path.join(save_folder, f'rewards.npy'), [min_reward, max_reward])

    def get_corner_particles(self, uv):
        # state = self.env.reset(given_goal=None, given_goal_pos=None)
        # uv = td.particle_uv_pos(self.env.camera_params[self.env.camera_name], None)
        # uv[:, [1, 0]] = uv[:, [0, 1]]
        uv = (uv / self.cfgs['cam_size']) * 199

        # corner 1
        dists = np.linalg.norm((uv - np.array([0, 0])), axis=1)
        c1 = dists.argmin()

        # corner 2
        dists = np.linalg.norm((uv - np.array([0, 200])), axis=1)
        c2 = dists.argmin()

        # corner 3
        dists = np.linalg.norm((uv - np.array([200, 200])), axis=1)
        c3 = dists.argmin()

        # corner 4
        dists = np.linalg.norm((uv - np.array([200, 0])), axis=1)
        c4 = dists.argmin()

        # u,v = uv[c1]
        # print(u,v)
        # action = [[u,v],[175,175],[u,v],[175,175]]
        # self.env.step(action, pickplace=True)

        return c1, c2, c3, c4

    def get_particle_uv(self, idx):
        uv = particle_uv_pos(self.env.camera_params[self.env.camera_name], None)
        uv[:, [1, 0]] = uv[:, [0, 1]]
        uv = (uv / self.cfgs['cam_size']) * 199
        u, v = uv[idx]
        return u, v

    def get_true_corner_mask(self, clothmask, canon_flat_uv, r=4):
        self.corners = self.get_corner_particles(canon_flat_uv)
        true_corners = np.zeros((200, 200))
        for c in self.corners:
            b, a = self.get_particle_uv(c)
            h, w = true_corners.shape
            y, x = np.ogrid[-a:h - a, -b:w - b]
            cmask = x * x + y * y <= r * r
            true_corners[cmask] = 1

        true_corners = true_corners * clothmask
        return true_corners


if __name__ == '__main__':
    eval_flag = False
    cloth_type = 'Trousers'  # Trousers, Tshirt, Dress, Skirt, Jumpsuit, Top  square rectangle
    dir_name = f'{cloth_type}_dataset_final'
    remove_invalid = True
    for eval_flag in [False]:
        num_eps = 4000  # 2200 # 3000
        horizon = 5
        if eval_flag:
            num_eps /= 10
        use_drop = True
        action_type = 'pickplace'  # qnet # pickplace # debug
        img_type = 'depth'  # color # depth, # desc
        edgethresh = 10 if cloth_type == 'tshirt' else 5
        actmaskprob = 0.9  # 0.9

        cemaskratio = 0.0  # ratio of how often to sample cloth edge mask
        on_table = False
        truecratio = 0.0
        cam_size = 720
        use_corner = True
        max_act = 150
        video = True
        set_type = 'val' if eval_flag else 'train'
        cfgs = {'debug': False, 'cam_size': cam_size, 'num_eps': num_eps, 'img_type': img_type,
                'use_drop': use_drop,
                'cloth_type': cloth_type,
                'max_act': max_act,
                'camera_name': 'top_down_camera',
                'video': video,
                # 'camera_name': 'default_camera',
                'rect': False, 'action_type': action_type, 'edgethresh': edgethresh, 'actmaskprob': actmaskprob,
                'cemaskratio': cemaskratio, 'tshirtmap_path': None, 'on_table': on_table, 'horizon': horizon,
                'state_dim': 200 * 200 * 3, 'dataset_folder': '', 'action_dim': 7,
                'dataset_name': f'{dir_name}/{set_type}',
                'desc_path': False, 'goals': [None], 'use_corner': use_corner, 'truecratio': truecratio,
                'headless': True,
                'eval_flag': eval_flag,
                'remove_invalid':remove_invalid}

        # if cloth_type == 'towel':
        #     cfgs['goals'] = [f'towel_train_{i}' for i in range(32)]
        # elif cloth_type == 'tshirt':
        #     cfgs['goals'] = ['tsf', 'two_tsf', 'three_step', 'tstsf', 'partial_horz_1', 'partial_vert_1', 'partial_vert_2', 'partial_diag_0', 'partial_diag_1', 'partial_diag_2']

        hostname = socket.gethostname()
        print("hostname: ", hostname)
        cfgs['dataset_folder'] = '/home/zixuanhu/occlusion_reasoning/dataset'
        dataset = DatasetGenerator(cfgs)
        # dataset.get_corner_particles()
        dataset.generate()
        dataset.env.close()
        # del dataset.env
        # del dataset
