import cv2
import numpy as np
import pyflex

from deprecated.utils_old.data_utils import get_matrix_world_to_camera
from visualization.plot_utils import write_number

from utils.camera_utils import project_to_image


def set_shape_pos(pos):
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    shape_states[:, 3:6] = pos.reshape(-1, 3)
    shape_states[:, :3] = pos.reshape(-1, 3)
    pyflex.set_shape_states(shape_states)


def visualize_rollout(env=None, particle_positions=None, shape_positions=None, config_id=None,
                      sample_idx=None, picked_particles=None, show=False, edges=None, goal=None, score=None,
                      draw_goal_flow=False):
    dummy_config = env.cached_configs[0]
    dummy_config['v'] = np.zeros((20000, 3))
    env.reset(config=dummy_config)
    frames = []
    cam_params = env.camera_params[env.camera_name]
    mat_world2cam = get_matrix_world_to_camera(cam_params)
    for i in range(len(particle_positions)):
        particle_pos = particle_positions[i]
        shape_pos = shape_positions[i]
        p = pyflex.get_positions().reshape(-1, 4)
        p[:, :3] = [0., -0.1, 0.]
        if sample_idx is None:
            p[:len(particle_pos), :3] = particle_pos
        else:
            p[:, :3] = [0, -0.1, 0]
            p[sample_idx, :3] = particle_pos
        pyflex.set_positions(p)
        set_shape_pos(shape_pos)
        rgb = env.get_image(env.camera_width, env.camera_height)
        u, v = project_to_image(matrix_world_to_camera=mat_world2cam,
                                world_coordinate=particle_pos,
                                height=env.camera_height, width=env.camera_width)
        if goal is not None:
            u_g, v_g = project_to_image(matrix_world_to_camera=mat_world2cam,
                                        world_coordinate=goal,
                                        height=env.camera_height, width=env.camera_width)
            if edges is not None:
                for edge_idx in range(edges.shape[1]):
                    s = edges[0][edge_idx]
                    r = edges[1][edge_idx]
                    start = (u_g[s], v_g[s])
                    end = (u_g[r], v_g[r])
                    color = (0, 255, 0)
                    thickness = 1
                    rgb = cv2.line(rgb, start, end, color, thickness)
            if draw_goal_flow and u.shape[0] == u_g.shape[0]:
                for i in range(u.shape[0]):
                    start = (u[i], v[i])
                    end = (u_g[i], v_g[i])
                    color = (0, 0, 255)
                    thickness = 1
                    rgb = cv2.line(rgb, start, end, color, thickness)

        if score is not None:
            write_number(rgb, score)

        if edges is not None:
            for edge_idx in range(edges.shape[1]):
                s = edges[0][edge_idx]
                r = edges[1][edge_idx]
                start = (u[s], v[s])
                end = (u[r], v[r])
                color = (255, 0, 0)
                thickness = 1
                rgb = cv2.line(rgb, start, end, color, thickness)

        frames.append(rgb)
        if show:
            if i == 0: continue
            picked_point = picked_particles[i]
            phases = np.zeros(pyflex.get_n_particles())
            # print(picked_point)
            for id in picked_point:
                if id != -1:
                    phases[sample_idx[int(id)]] = 1
            pyflex.set_phases(phases)
            img = env.get_image()

            cv2.imshow('picked particle images', img[:, :, ::-1])
            cv2.waitKey()

    return frames
