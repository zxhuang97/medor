from typing import Any

from typing import Dict

import cv2
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from joblib import delayed
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff

from utils.geometry_utils import get_edges_from_tri
import io
from PIL import Image
from utils.my_decor import auto_numpy

from visualization.plot_utils import get_plot_view, make_grid, _volume_trace, \
    update_scene_layout


def pointcloud(T_chart_points: np.ndarray, downsample=5, colors=None, colorscale=None) -> go.Scatter3d:
    marker_dict: Dict[str, Any] = {"size": 1.8}
    if colorscale is not None and colors is not None:
        marker_dict['color'] = colors
        marker_dict['colorscale'] = colorscale
    elif colors is not None:
        if isinstance(colors, str):
            marker_dict['color'] = colors
        else:
            try:
                a = [f"rgb({r}, {g}, {b})" for r, g, b in colors][::downsample]
                marker_dict["color"] = a
            except:
                marker_dict["color"] = colors[::downsample]

    return go.Scatter3d(
        x=T_chart_points[::downsample, 0],
        y=T_chart_points[::downsample, 1],
        z=T_chart_points[::downsample, 2],
        mode="markers",
        marker=marker_dict,
    )


@auto_numpy
def plot_pointcloud(data: np.ndarray, fig=None, downsample=5, colors=None, colorscale=None, row=None, col=None, id=1,
                    view='top', dis=2, show_grid=True, center=True):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(pointcloud(data, downsample, colors, colorscale=colorscale), row=row, col=col)
    update_scene_layout(fig, scene_id=id, pts=data, center=center, view=view, dis=dis, show_grid=show_grid)

    return fig


@auto_numpy
def nocs3d_fig(gt_nocs, pred_nocs, color=None):
    fig = make_subplots(rows=1,
                        cols=2,
                        column_width=[0.5, 0.5],
                        specs=[[{'type': 'Scatter3d'}, {'type': 'Scatter3d'}]],
                        subplot_titles=['gt warp field', 'pred warp field'],
                        horizontal_spacing=0.03,
                        vertical_spacing=0.05)
    if color is None:
        color = gt_nocs
    plot_pointcloud(gt_nocs, fig=fig, colors=color, downsample=1, row=1, col=1, id=1)
    plot_pointcloud(pred_nocs, fig=fig, colors=color, downsample=1, row=1, col=2, id=2)
    nocs_err = np.linalg.norm(gt_nocs - pred_nocs, ord=2, axis=-1).mean()
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        height=600, width=1800, margin=dict(l=0, r=0, b=0, t=40),
        title={
            'text': f"NOCS err {nocs_err:04f}",
            'y': 0.99,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        }
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


@auto_numpy
def volume_fig(volume):
    fig = go.Figure()
    fig.add_trace(_volume_trace(volume))
    return fig


@auto_numpy
def plot_mesh_face(v, f, fig=None, colors=None, view='top', row=None, col=None, id=1, dis=2, opacity=0.6,
                   show_grid=True):
    if fig is None:
        fig = go.Figure()
    fig = fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                  i=f[:, 0], j=f[:, 1], k=f[:, 2], colorscale='RdYlBu',
                                  opacity=opacity), row=row, col=col)
    tri_pts = v[f]
    Xe = []
    Ye = []
    Ze = []
    for T in tri_pts:
        Xe.extend([T[k % 3][0] for k in range(4)] + [None])
        Ye.extend([T[k % 3][1] for k in range(4)] + [None])
        Ze.extend([T[k % 3][2] for k in range(4)] + [None])
    fig = fig.add_trace(go.Scatter3d(x=Xe,
                                     y=Ye,
                                     z=Ze,
                                     mode='lines'
                                     ), row=row, col=col)
    update_scene_layout(fig, scene_id=id, pts=v, view=view, show_grid=show_grid, dis=dis)
    fig.update_layout(
        {
            'margin': dict(l=0, r=0, b=0, t=0),
            'legend': dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        }
    )
    return fig


def pair_mesh_fig(v_gt, f_gt, v_pred, f_pred):
    fig = make_subplots(rows=3,
                        cols=2,
                        column_width=[0.5, 0.5],
                        specs=[[{'type': 'Scatter3d'}, {'type': 'Scatter3d'}],
                               [{'type': 'Scatter3d'}, {'type': 'Scatter3d'}],
                               [{'type': 'Scatter3d'}, {'type': 'Scatter3d'}]],
                        subplot_titles=['gt_mesh', 'pred_mesh'],
                        horizontal_spacing=0.03,
                        vertical_spacing=0.05)
    plot_mesh_face(v_gt, f_gt, fig=fig, downsample=5, row=1, col=1, id=1)
    plot_mesh_face(v_pred, f_pred, fig=fig, downsample=5, row=1, col=2, id=2)
    plot_mesh_face(v_gt, f_gt, fig=fig, downsample=5, row=2, col=1, id=3)
    plot_mesh_face(v_pred, f_pred, fig=fig, downsample=5, row=2, col=2, id=4)
    plot_mesh_face(v_gt, f_gt, fig=fig, downsample=5, row=3, col=1, id=5)
    plot_mesh_face(v_pred, f_pred, fig=fig, downsample=5, row=3, col=2, id=6)
    return fig


@auto_numpy
def plot_mesh_edge(v, e, fig=None, downsample=5, colors=None, row=None, col=None, id=1):
    if fig is None:
        fig = go.Figure()
    plot_pointcloud(v, fig=fig, colors=colors, downsample=downsample, row=row, col=col, id=id)
    tri_pts = v[e]  # N x 2 x3
    Xe = []
    Ye = []
    Ze = []
    for T in tri_pts:
        Xe.extend([T[k][0] for k in range(2)] + [None])
        Ye.extend([T[k][1] for k in range(2)] + [None])
        Ze.extend([T[k][2] for k in range(2)] + [None])
    fig = fig.add_trace(go.Scatter3d(x=Xe,
                                     y=Ye,
                                     z=Ze,
                                     mode='lines'
                                     ), row=row, col=col)
    return fig


@auto_numpy
def pair_mesh_flex_fig(v_gt, f_gt, v_pred, e_pred):
    fig = make_subplots(rows=1,
                        cols=2,
                        column_width=[0.5, 0.5],
                        specs=[[{'type': 'Scatter3d'}, {'type': 'Scatter3d'}]],
                        subplot_titles=['gt_mesh', 'pred_mesh'],
                        horizontal_spacing=0.03,
                        vertical_spacing=0.05)
    plot_mesh_face(v_gt, f_gt, fig=fig, downsample=5, row=1, col=1, id=1)
    plot_mesh_edge(v_pred, e_pred, fig=fig, downsample=5, row=1, col=2, id=2)
    return fig


def triple_mesh_flex_fig(v_gt, f_gt, v_mc, f_mc, v_pred, e_pred):
    fig = make_subplots(rows=1,
                        cols=3,
                        column_width=[0.33, 0.33, 0.33],
                        specs=[[{'type': 'Scatter3d'}, {'type': 'Scatter3d'}, {'type': 'Scatter3d'}]],
                        subplot_titles=['gt_mesh', "mc", 'neighbor'],
                        horizontal_spacing=0.03,
                        vertical_spacing=0.05)
    plot_mesh_face(v_gt, f_gt, fig=fig, downsample=5, row=1, col=1, id=1)
    plot_mesh_face(v_mc, f_mc, fig=fig, downsample=5, row=1, col=2, id=2)
    plot_mesh_edge(v_pred, e_pred, fig=fig, downsample=5, row=1, col=3, id=3)
    return fig


@auto_numpy
def plot_mesh_face_v2(v, f, fig=None, colors=None, view='top', row=None, col=None, id=1):
    if fig is None:
        fig = go.Figure()
    fig = fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                  i=f[:, 0], j=f[:, 1], k=f[:, 2], vertexcolor=colors,
                                  opacity=0.8), row=row, col=col)

    edges = get_edges_from_tri(f)
    tri_pts = v[edges]  # N x 2 x3
    Xe = []
    Ye = []
    Ze = []
    for T in tri_pts:
        Xe.extend([T[k][0] for k in range(2)] + [None])
        Ye.extend([T[k][1] for k in range(2)] + [None])
        Ze.extend([T[k][2] for k in range(2)] + [None])
    fig = fig.add_trace(go.Scatter3d(x=Xe,
                                     y=Ye,
                                     z=Ze,
                                     mode='lines'
                                     ), row=row, col=col)
    mean = v.mean(axis=0)
    max_x = np.abs(v[:, 0] - mean[0]).max()
    max_y = np.abs(v[:, 1] - mean[1]).max()
    max_z = np.abs(v[:, 2] - mean[2]).max()
    all_max = max(max(max_x, max_y), max_z)
    camera = get_plot_view(view, dis=1.5)
    fig.update_layout(
        {f'scene{id}': dict(
            xaxis=dict(nticks=10, range=[mean[0] - all_max, mean[0] + all_max]),
            yaxis=dict(nticks=10, range=[mean[1] - all_max, mean[1] + all_max]),
            zaxis=dict(nticks=10, range=[mean[2] - all_max, mean[2] + all_max]),
            aspectratio=dict(x=1, y=1, z=1),
            camera=camera,
        ),
            'margin': dict(l=0, r=0, b=0, t=40),
            'legend': dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        }
    )
    return fig

@auto_numpy
def all_in_one_plot3(rgb, depth,
                     gt_nocs, pred_nocs,
                     gt_nocs_verts, pred_nocs_verts,
                     gt_task_verts, pred_task_verts, opt_task_verts,
                     gt_cloth_tri, pred_cloth_tri):
    fig = make_subplots(rows=1,
                        cols=4,
                        specs=[[{'type': 'image'},
                                {'type': 'Scatter3d'},
                                {'type': 'Scatter3d'}, {'type': 'Scatter3d'}],
                               ],
                        subplot_titles=('RGB', 'Initial predicted mesh',
                                        'Fine-tuned predicted mesh',
                                        'Ground-truth mesh'
                                        ),
                        column_widths=[0.2, 0.26, 0.26, 0.26],
                        horizontal_spacing=0.005,
                        vertical_spacing=0.05)

    fig.add_trace(px.imshow(rgb, aspect="equal").data[0], row=1, col=1)

    plot_mesh_face(pred_task_verts, pred_cloth_tri, fig=fig, row=1, col=2, id=1)
    plot_mesh_face(opt_task_verts, pred_cloth_tri, fig=fig, row=1, col=3, id=2)
    plot_mesh_face(gt_task_verts, gt_cloth_tri, fig=fig, row=1, col=4, id=3)

    fig.update_yaxes(automargin=True)
    fig.update_layout(
        height=275, width=1100, margin=dict(l=0, r=0, b=0, t=40),
        title={
            'text': f"",
            'y': 0.99,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        }
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


@auto_numpy
def plot_without_gt2(rgb, depth, img_nocs, nocs, canon_verts, warp_field, opt_warp_field, faces, pc):
    fig = make_subplots(rows=9,
                        cols=1,
                        column_width=[1 / 1] * 1,
                        specs=[[{'type': 'image'}],
                               [{'type': 'image'}], [{'type': 'image'}],
                               [{'type': 'Scatter3d'}], [{'type': 'Scatter3d'}],
                               [{'type': 'Scatter3d'}], [{'type': 'Scatter3d'}], [{'type': 'Scatter3d'}],
                               [{'type': 'xy'}]
                               ],
                        subplot_titles=['RGB', 'Depth', 'Canonical surface mapping', 'NOCS point cloud',
                                        'Canonical shape', 'Initial observed shape', 'Fine-tuned observed space',
                                        'Point cloud', 'Depth Dist'],
                        horizontal_spacing=0.00,
                        vertical_spacing=0.00)

    rgb = rgb.astype(np.uint8)
    # fig.add_trace(px.imshow(Image.fromarray(depth_o)).data[0], row=1, col=1)
    fig.add_trace(px.imshow(Image.fromarray(rgb)).data[0], row=1, col=1)

    fig.add_trace(px.imshow(depth, aspect="equal").data[0], row=2, col=1)
    # fig.add_trace(px.imshow(go.Image(z=depth)).data[0], row=2, col=1)

    img_nocs = img_nocs * 255
    img_nocs = img_nocs.astype(np.uint8)
    fig.add_trace(px.imshow(Image.fromarray(img_nocs)).data[0], row=3, col=1)

    plot_pointcloud(nocs, fig=fig, colors=nocs, downsample=1, row=4, col=1, id=1)

    plot_mesh_face_v2(canon_verts, faces, fig=fig, colors=canon_verts, row=5, col=1, id=2)

    plot_pointcloud(pc, fig=fig, row=6, col=1, id=3)
    plot_pointcloud(pc, fig=fig, row=7, col=1, id=4)
    plot_mesh_face_v2(warp_field, faces, fig=fig, colors=canon_verts, row=6, col=1, id=3)
    plot_mesh_face_v2(opt_warp_field, faces, fig=fig, colors=canon_verts, row=7, col=1, id=4)

    plot_pointcloud(pc, fig=fig, row=8, col=1, id=5)
    depth_values = depth[depth > 0.45]

    fig.update_layout(showlegend=False)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.add_trace(go.Histogram(x=depth_values, histnorm='probability',
                               xbins=dict(
                                   start=0.58,
                                   end=0.65,
                                   size=0.0003
                               )),
                  row=9,
                  col=1)

    fig.update_yaxes(automargin=True)
    fig.update_layout(
        height=4000, width=500, margin=dict(l=0, r=0, b=0, t=40),
        title={
            # 'text': f"Chamfer distance with point cloud {chamfer_pc:04f}",
            'y': 0.99,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        }
    )
    fig.update(layout_coloraxis_showscale=False)
    #     fig.update(showlegend=False)
    return fig


def plot_optimization_step(gt_nocs=None, pred_nocs=None, gt_pos=None, pred_pos=None, error=None, size=200,
                           views=None, **kwargs):
    """
    Visualize a step of optimization by 3 views
    3x3
    Each row: gt, canon plot, error plot,
    Three rows: Top, front, side views.
    """
    if views is None:
        views = ['top', 'front', 'side']
    if error is None:
        error = pred_pos[:, 1]
    img_list = []

    for view in views:
        f = plot_pointcloud(gt_pos, colors=gt_nocs, view=view, **kwargs)
        img_list.append(f.to_image(format="png", width=size, height=size))
        f = plot_pointcloud(pred_pos, colors=pred_nocs, view=view, **kwargs)
        img_list.append(f.to_image(format="png", width=size, height=size))
        f = plot_pointcloud(pred_pos, colors=error, colorscale='sunsetdark', view=view, **kwargs)
        img_list.append(f.to_image(format="png", width=size, height=size))
    img_stack = np.stack([np.array(Image.open(io.BytesIO(img))) for img in img_list])
    out = make_grid(img_stack, ncol=3)

    return out

def plot_optimization_step_mesh_single(gt_v, gt_f, pred_v, pred_f, size=200,
                                views=None, **kwargs):
    """
    Visualize a step of optimization by 3 views
    3x3
    Each row: gt, canon plot, error plot,
    Three rows: Top, front, side views.
    """
    if views is None:
        views = ['top', 'front', 'side']
    img_list = []
    f = plot_mesh_face(gt_v, gt_f, view='top', **kwargs)
    gt_img = f.to_image(format="png", width=size, height=size)
    for view in views:
        img_list.append(gt_img)
        f = plot_mesh_face(pred_v, pred_f, view=view, **kwargs)
        img_list.append(f.to_image(format="png", width=size, height=size))
    img_stack = np.stack([np.array(Image.open(io.BytesIO(img))) for img in img_list])
    out = make_grid(img_stack, ncol=1)

    return out

def plot_optimization_step_mesh(gt_v, gt_f, pred_v, pred_f, size=200,
                                views=None, **kwargs):
    """
    Visualize a step of optimization by 3 views
    3x3
    Each row: gt, canon plot, error plot,
    Three rows: Top, front, side views.
    """
    if views is None:
        views = ['top', 'front', 'side']
    img_list = []

    for view in views:
        f = plot_mesh_face(gt_v, gt_f, view=view, **kwargs)
        img_list.append(f.to_image(format="png", width=size, height=size))
        f = plot_mesh_face(pred_v, pred_f, view=view, **kwargs)
        img_list.append(f.to_image(format="png", width=size, height=size))
    img_stack = np.stack([np.array(Image.open(io.BytesIO(img))) for img in img_list])
    out = make_grid(img_stack, ncol=2)

    return out



def plot_flow_quiver(
        flow_img,
        downsample=10):
    if flow_img.shape[0] == 2:
        flow_img = flow_img.transpose(1, 2, 0)
    h, w, _ = flow_img.shape
    bg = np.ones((h, w, 3))
    fig = px.imshow(bg)
    ys, xs = np.where((flow_img != 0).any(axis=2))
    n = len(xs)
    ind = np.random.choice(np.arange(n), size=int(n / downsample), replace=False)
    flu = flow_img[ys[ind], xs[ind], 1]
    flv = flow_img[ys[ind], xs[ind], 0]
    fig_q = ff.create_quiver(xs[ind], ys[ind], flu, flv, scale=1)
    for d in fig_q.data:
        fig.add_trace(go.Scatter(x=d['x'], y=d['y']), row=1, col=1)
    return fig


def plot_pc_tracking_frame(pc, mesh_v=None, mesh_f=None, picker_pos=None, render_size=350, views=['tilt'], mean=None,
                           all_max=None):
    if views is None:
        views = ['top', 'front']
    img_list = []

    f = go.Figure()
    color = 'red' if mesh_v is not None else None
    f = plot_pointcloud(pc, f, colors=color)
    if picker_pos is not None:
        f.add_trace(go.Scatter3d(x=[picker_pos[0]],
                                 y=[picker_pos[1]],
                                 z=[picker_pos[2]],
                                 mode='markers',
                                 marker=dict(
                                     size=12,
                                     opacity=0.8
                                 ),
                                 ))
    if mesh_v is not None and mesh_f is not None:
        plot_mesh_face(mesh_v, mesh_f, f)
    elif mesh_v is not None:
        plot_pointcloud(mesh_v, f)
    if mean is None or all_max is None:
        mean = pc.mean(axis=0)
        max_x = np.abs(pc[:, 0] - mean[0]).max()
        max_y = np.abs(pc[:, 1] - mean[1]).max()
        max_z = np.abs(pc[:, 2] - mean[2]).max()
        all_max = max(max(max_x, max_y), max_z)
    for view in views:
        camera = get_plot_view(view, dis=1.4)
        f.update_layout(
            {f'scene1': dict(
                xaxis=dict(nticks=10, range=[mean[0] - all_max, mean[0] + all_max]),
                yaxis=dict(nticks=10, range=[mean[1] - all_max, mean[1] + all_max]),
                zaxis=dict(nticks=10, range=[mean[2] - all_max, mean[2] + all_max]),
                aspectratio=dict(x=1, y=1, z=1),
                camera=camera,
            )
            }
        )
        f.update_layout(showlegend=True)
        img_list.append(f.to_image(format="png", width=render_size, height=render_size))
    img_stack = np.stack([np.array(Image.open(io.BytesIO(img))) for img in img_list])
    out = make_grid(img_stack, ncol=len(views))
    return out


def plot_pc_track_gif(traj, parallel, ds_rate=2):
    # traj: a dictionary of list
    # parallel: joblib Parallel object
    pc_list, picker_list, rgb_list = traj['cloth_pc'], traj['picker_pos'], traj['rgb']
    pc_list, picker_list, rgb_list = pc_list[::ds_rate], picker_list[::ds_rate], rgb_list[::ds_rate]
    all_pc = np.concatenate(pc_list)
    mean = all_pc.mean(axis=0)
    max_x = np.abs(all_pc[:, 0] - mean[0]).max()
    max_y = np.abs(all_pc[:, 1] - mean[1]).max()
    max_z = np.abs(all_pc[:, 2] - mean[2]).max()
    all_max = max(max(max_x, max_y), max_z)
    vis_data = [dict(pc=pc, picker_pos=picker[:3], render_size=500, views=['top', 'front', 'tilt'],
                     mean=mean, all_max=all_max) for pc, picker in zip(pc_list, picker_list)]
    xs = parallel(delayed(plot_pc_tracking_frame)(**x) for x in vis_data)
    rgb_pc = np.stack(
        [np.concatenate([cv2.resize(r, (500, 500)), x[:, :, :3]], 1) for r, x in zip(rgb_list, xs)])
    return rgb_pc


def plot_pc_rollout_gif(pc_traj, picker_traj, model_positions, rgb_list, model_face=None, parallel=None, ds_rate=2):
    # traj: a dictionary of list
    # parallel: joblib Parallel object
    pc_list, picker_list, model_positions = pc_traj[::ds_rate], picker_traj[::ds_rate], model_positions[::ds_rate]
    all_pc = np.concatenate(pc_list)
    mean = all_pc.mean(axis=0)
    max_x = np.abs(all_pc[:, 0] - mean[0]).max()
    max_y = np.abs(all_pc[:, 1] - mean[1]).max()
    max_z = np.abs(all_pc[:, 2] - mean[2]).max()
    all_max = max(max(max_x, max_y), max_z)
    vis_data = [dict(pc=pc, mesh_v=mv, mesh_f=model_face, picker_pos=picker[:3], render_size=500,
                     views=['top', 'front', 'tilt'],
                     mean=mean, all_max=all_max) for pc, mv, picker in zip(pc_list, model_positions, picker_list)]
    xs = parallel(delayed(plot_pc_tracking_frame)(**x) for x in vis_data)
    rgb_pc = np.stack(
        [np.concatenate([cv2.resize(r, (500, 500)), x[:, :, :3]], 1) for r, x in zip(rgb_list, xs)])
    return rgb_pc


def create_figure(fixed_range=2) -> go.Figure:
    fig = go.Figure()
    if fixed_range is not None:
        fig.update_layout(
            width=700,
            scene_aspectmode="manual",
            scene=dict(
                xaxis=dict(nticks=10, range=[-2, 2]),
                yaxis=dict(nticks=10, range=[-2, 2]),
                zaxis=dict(nticks=10, range=[-2, 2]),
                aspectratio=dict(x=1, y=1, z=1),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )
    else:
        fig.update_layout(width=700)
    return fig


def seg_3d_figure(
        data: np.ndarray, labels: np.ndarray, labelmap=None, sizes=None, fig=None, show_grid=True,
):
    # Create a figure.
    if fig is None:
        fig = go.Figure()

    # Colormap.
    cols = np.array(pc.qualitative.Alphabet)
    labels = labels.astype(int)
    for label in np.unique(labels):
        subset = data[np.where(labels == label)]
        # subset = np.squeeze(subset)
        if sizes is None:
            subset_sizes = 1.5
        else:
            subset_sizes = sizes[np.where(labels == label)]
        color = cols[label % len(cols)]
        if labelmap is not None:
            legend = labelmap[label]
        else:
            legend = str(label)
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker={"size": subset_sizes, "color": color, "line": {"width": 0}},
                x=subset[:, 0],
                y=subset[:, 1],
                z=subset[:, 2],
                name=legend,
            )
        )
    fig.update_layout(showlegend=True)

    # This sets the figure to be a cube centered at the center of the pointcloud, such that it fits
    # all the points.
    update_scene_layout(fig, pts=data, show_grid=show_grid)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=1.0, y=0.75),
    )

    return fig


@auto_numpy
def plot_pointclouds(pcs, fig=None, show_grid=True):
    labels = [np.ones(pc.shape[0]) * i for i, pc in enumerate(pcs)]
    labels = np.concatenate(labels)
    pcs = np.concatenate(pcs)
    return seg_3d_figure(pcs, labels, fig=fig, show_grid=show_grid)

def _flow_traces_v2(
        pos, flows, sizeref=0.05, scene="scene", flowcolor="red", name="flow"
):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = np.all(flows == 0.0, axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flows[~nonzero_flows]
    n_dest = n_pos + n_flows * sizeref
    for i in range(len(n_pos)):
        x_lines.append(n_pos[i][0])
        y_lines.append(n_pos[i][1])
        z_lines.append(n_pos[i][2])
        x_lines.append(n_dest[i][0])
        y_lines.append(n_dest[i][1])
        z_lines.append(n_dest[i][2])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        scene=scene,
        line=dict(color=flowcolor, width=5),
        name=name,
        opacity=0.7
    )

    return [lines_trace]


@auto_numpy
def plot_dense_pc_flow(pc, flow, pc_nxt=None, fig=None, thres=0, ds_rate=1, sizeref=1):
    pc = pc[::ds_rate]
    flow = flow[::ds_rate]
    # pc = torch.tensor(pc).detach().cpu()
    # flow = flow.detach().cpu()

    f_norm = np.linalg.norm(flow, axis=-1)
    select = f_norm > thres
    pc = pc[select]
    flow = flow[select]
    f_norm = np.linalg.norm(flow, axis=-1)
    #     print(f_norm.shape)
    print(f_norm.mean(), np.quantile(f_norm, 0.9), np.quantile(f_norm, 0.98))
    if fig is None:
        fig = go.Figure()
    plot_pointcloud(pc, fig=fig, downsample=1)
    fig.add_traces(_flow_traces_v2(pc, flow, sizeref=sizeref))
    if pc_nxt is not None:
        plot_pointcloud(pc_nxt, fig=fig, downsample=1)
    fig.update_layout(height=500, width=800)
    return fig

