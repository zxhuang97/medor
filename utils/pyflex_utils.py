import cv2
import numpy as np
import pyflex

from softgym.utils.misc import vectorized_range, vectorized_meshgrid


def visualize(env, particle_positions, shape_positions, config_id, sample_idx=None, picked_particles=None, show=False):
    """ Render point cloud trajectory without running the simulation dynamics"""
    env.reset(config_id=config_id)
    frames = []
    for i in range(len(particle_positions)):
        particle_pos = particle_positions[i]
        shape_pos = shape_positions[i]
        p = pyflex.get_positions().reshape(-1, 4)
        p[:, :3] = [0., -0.1, 0.]  # All particles moved underground
        if sample_idx is None:
            p[:len(particle_pos), :3] = particle_pos
        else:
            p[:, :3] = [0, -0.1, 0]
            p[sample_idx, :3] = particle_pos
        pyflex.set_positions(p)
        set_shape_pos(shape_pos)
        rgb = env.get_image(env.camera_width, env.camera_height)
        frames.append(rgb)
        if show:
            if i == 0: continue
            picked_point = picked_particles[i]
            phases = np.zeros(pyflex.get_n_particles())
            for id in picked_point:
                if id != -1:
                    phases[sample_idx[int(id)]] = 1
            pyflex.set_phases(phases)
            img = env.get_image()

            cv2.imshow('picked particle images', img[:, :, ::-1])
            cv2.waitKey()

    return frames


def set_shape_pos(pos):
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    shape_states[:, 3:6] = pos.reshape(-1, 3)
    shape_states[:, :3] = pos.reshape(-1, 3)
    pyflex.set_shape_states(shape_states)


def set_picker_pos(pos):
    shape_states = pyflex.get_shape_states().reshape((-1, 14))
    shape_states[1, :3] = -1
    shape_states[1, 3:6] = -1

    shape_states[0, :3] = pos
    shape_states[0, 3:6] = pos
    pyflex.set_shape_states(shape_states)
    pyflex.step()


def pc_reward_model(pos, cloth_particle_radius=0.00625, downsample_scale=3):
    cloth_particle_radius *= downsample_scale
    pos = np.reshape(pos, [-1, 3])
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 2])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 2])
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / 100.
    pos2d = pos[:, [0, 2]]

    offset = pos2d - init
    slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
    slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)

    grid = np.zeros(10000)  # Discretization
    listx = vectorized_range(slotted_x_low, slotted_x_high)
    listy = vectorized_range(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid(listx, listy)
    idx = listxx * 100 + listyy
    idx = np.clip(idx.flatten(), 0, 9999)
    grid[idx] = 1

    res = np.sum(grid) * span[0] * span[1]
    return res


def coverage_reward(pos, cloth_particle_radius=0.00625, downsample_scale=3):
    cloth_particle_radius *= downsample_scale
    pos = np.reshape(pos, [-1, 3])

    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 2])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 2])
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / 100.
    pos2d = pos[:, [0, 2]]

    offset = pos2d - init
    slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
    slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)
    # Method 1
    grid = np.zeros(10000)  # Discretization
    listx = vectorized_range(slotted_x_low, slotted_x_high)
    listy = vectorized_range(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid(listx, listy)
    idx = listxx * 100 + listyy
    idx = np.clip(idx.flatten(), 0, 9999)
    grid[idx] = 1

    # just hard code here. TODO: change this according to different cloth size
    # filter out those that exploits the model error

    # max_possible_span = (60 * 0.00625) ** 2
    res = np.sum(grid) * span[0] * span[1]

    # if use_heuristic_reward:
    #     # if res > max_possible_span:
    #     #     res = 0
    # if np.any(pos[:, 1] > 0.00625 * 15):
    #     res = 0

    return res
