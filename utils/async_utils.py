import multiprocessing as mp
from multiprocessing import Pipe, Process
import os.path as osp
import numpy as np

from visualization.plot_utils import save_numpy_as_gif
from visualization.vids import visualize_rollout

render_env = None





def init_io_worker(env_class, env_args):
    global render_env
    render_env = env_class(**env_args)
    for c in render_env.cached_configs:
        c['radius'] = render_env.cloth_particle_radius


def async_vis_io(**kwargs):
    global render_env
    frames_gt = visualize_rollout(env=render_env,
                                  particle_positions=kwargs['particle_pos_gt'],
                                  shape_positions=kwargs['shape_pos'],
                                  sample_idx=kwargs['gt_ds_id'],
                                  edges=kwargs['edges_gt'],
                                  goal=kwargs['goal_gt'],
                                  score=kwargs['score'],
                                  draw_goal_flow=kwargs['draw_goal_flow']
                                  )

    frames_model = visualize_rollout(env=render_env,
                                     particle_positions=kwargs['particle_pos_model'],
                                     shape_positions=kwargs['shape_pos'],
                                     sample_idx=kwargs['sample_idx'],
                                     edges=kwargs['edges_model'],
                                     goal=kwargs['goal_model'],
                                     draw_goal_flow=kwargs['draw_goal_flow'],
                                     )
    combined_frames = [np.hstack([frame_gt, frame_model]) for (frame_gt, frame_model) in
                       zip(frames_gt, frames_model)]

    save_numpy_as_gif(np.array(combined_frames), osp.join(kwargs['logdir'], '{}-{}.gif'.format(
        kwargs['episode_idx'], kwargs['pick_try_idx']
    )), fps=10)