import io
import os

import cv2
import numpy as np
import plotly.graph_objects as go
import tqdm
from PIL import Image
from joblib import delayed
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def write_number(img, number, pos=None, color=(0, 0, 0)):  # Inplace modification
    img = np.ascontiguousarray(img, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w, _ = img.shape
    if pos is None:
        bottomLeftCornerOfText = (w // 2, 70)
    else:
        bottomLeftCornerOfText = (int(w * pos[0]), int(h * pos[1]))
    fontScale = 1
    fontColor = color
    thickness = 2
    if isinstance(number, str):
        cv2.putText(img, '{}'.format(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, thickness)
    elif isinstance(number, int):
        cv2.putText(img, str(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, thickness)
    else:
        cv2.putText(img, '{:.2f}'.format(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, thickness)
    return img


def save_numpy_as_gif(array, filename, fps=20, scale=1.0, add_index_rate=5):
    """Creates a gif given a stack of images using moviepy
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    if add_index_rate > 0:
        for i in range(array.shape[0]):
            array[i] = write_number(array[i], i * add_index_rate)
    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps)
    clip.write_gif(filename, fps=fps)
    # return clip


def get_plot_view(view='top', dis=2.):
    view_dict = {
        'top': dict(
            up=dict(x=0, y=2, z=0),
            eye=dict(x=0., y=dis, z=0.)
        ),
        'front': dict(
            up=dict(x=0, y=2, z=0),
            eye=dict(x=0, y=0, z=dis)
        ),
        'side': dict(
            up=dict(x=0, y=2, z=0),
            eye=dict(x=dis, y=0, z=0.)
        ),
        'tilt': dict(
            up=dict(x=0, y=1, z=0),
            eye=dict(x=dis / 2, y=dis, z=-dis)
        ),
    }
    return view_dict[view]


def make_grid(array, ncol=5, padding=0, pad_value=120, index_img=False):
    """ numpy version of the make_grid function in torch. Dimension of array: NHWC """
    if len(array.shape) == 3:  # In case there is only one channel
        array = np.expand_dims(array, 3)
    N, H, W, C = array.shape
    if N % ncol > 0:
        res = ncol - N % ncol
        array = np.concatenate([array, np.ones([res, H, W, C])])
        N = array.shape[0]
    nrow = N // ncol
    idx = 0
    grid_img = None
    for i in range(nrow):
        row = np.pad(array[idx], [[padding if i == 0 else 0, padding], [padding, padding], [0, 0]],
                     constant_values=pad_value, mode='constant')
        for j in range(1, ncol):
            idx += 1
            cur_img = np.pad(array[idx], [[padding if i == 0 else 0, padding], [0, padding], [0, 0]],
                             constant_values=pad_value, mode='constant')
            if index_img:
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (150, 150)
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cur_img = cv2.putText(cur_img, str(idx), org, font, fontScale, color, thickness, cv2.LINE_AA)
            row = np.hstack([row, cur_img])
        idx += 1
        if i == 0:
            grid_img = row
        else:
            grid_img = np.vstack([grid_img, row])
    return grid_img.astype(np.int32)


def _volume_trace(volume):
    # render a volumetric representation
    h, w, z = volume.shape
    X, Y, Z = np.meshgrid(np.arange(h),
                          np.arange(w),
                          np.arange(z))

    # print('?', np.mean(volume[Y.flatten(), X.flatten(), Z.flatten()] == volume.flatten()))

    trace = go.Volume(
        x=Y.flatten(),
        y=X.flatten(),
        z=Z.flatten(),
        value=volume.flatten(),
        opacity=0.3,
        surface_count=21
    )
    return trace


def update_scene_layout(fig, scene_id=1, pts=None, center=True, view='top', dis=2, show_grid=True):
    """
    Configure the layout of scene for 3d plot
    """
    camera = get_plot_view(view, dis=dis)
    scene_cfg = dict(
        xaxis=dict(nticks=10),
        yaxis=dict(nticks=10),
        zaxis=dict(nticks=10),
        aspectratio=dict(x=1, y=1, z=1),
        camera=camera,
    )
    if pts is not None and center:
        mean = pts.mean(axis=0)
        max_x = np.abs(pts[:, 0] - mean[0]).max()
        max_y = np.abs(pts[:, 1] - mean[1]).max()
        max_z = np.abs(pts[:, 2] - mean[2]).max()
        all_max = max(max(max_x, max_y), max_z)
        for i, axis in enumerate(['xaxis', 'yaxis', 'zaxis']):
            scene_cfg[axis]['range'] = [mean[i] - all_max, mean[i] + all_max]

    if not show_grid:
        for axis in ['xaxis', 'yaxis', 'zaxis']:
            scene_cfg[axis].update(dict(
                showticklabels=False,
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"))
            scene_cfg[axis + '_title'] = ''

    fig.update_layout({f'scene{scene_id}': scene_cfg})


def rotate_z(x, y, z, theta):
    w = x + 1j * y
    return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z


def get_rotating_frames(fig, scene_num=1, frames_num=60, width=600, height=600, parallel=None):
    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5
    step = 6.26 / frames_num

    def rotate_and_save(fig, t):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        for i in range(scene_num):
            fig.update_layout(
                {f'scene{i + 1}_camera_eye': dict(x=xe, y=ye, z=ze)}
            )
        return Image.open(io.BytesIO(fig.to_image(format="png", width=width, height=height)))

    if parallel is None:
        frames = []
        for t in tqdm.tqdm(np.arange(0, 6.26, step)):
            frames.append(rotate_and_save(fig, t))
    else:
        frames = parallel(delayed(rotate_and_save)(fig, t) for t in np.arange(0, 6.26, step))
    return frames
