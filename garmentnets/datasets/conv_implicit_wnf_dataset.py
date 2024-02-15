import copy
import os
import pathlib
import pdb
import pickle
import socket
from typing import Tuple, Optional

import cv2
import igl
import numpy as np
import pandas as pd
import imageio
import pytorch_lightning as pl
import torch
from torch.nn import Parameter
import zarr
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from torch_geometric.data import Dataset, Data, DataLoader as DataLoader_g

from garmentnets.common.cache import file_attr_cache
from garmentnets.common.geometry_util import (
    barycentric_interpolation, mesh_sample_barycentric)
from garmentnets.components.gridding import nocs_grid_sample

from utils.data_utils import read_h5_dict, MyData, my_collate

import torchvision.transforms.functional as TF

from visualization.plot import plot_pointcloud


# helper functions
# ================
def _get_groups_df(samples_group):
    rows = dict()
    for key, group in samples_group.items():
        rows[key] = group.attrs.asdict()
    groups_df = pd.DataFrame(data=list(rows.values()), index=rows.keys())
    groups_df.drop_duplicates(inplace=True)
    groups_df['group_key'] = groups_df.index
    return groups_df


# data sets
# =========
class ConvImplicitWNFDataset(Dataset):
    def __init__(self,
                 input_type: str,
                 data_path: str,
                 num_pc_sample: int = 6000,
                 num_volume_sample: int = 0,
                 num_surface_sample: int = 0,
                 num_mc_surface_sample: int = 0,
                 # mixed sampling config
                 surface_sample_ratio: float = 0,
                 surface_sample_std: float = 0.05,
                 # surface sample noise
                 surface_normal_noise_ratio: float = 0,
                 surface_normal_std: float = 0,
                 # data augumentaiton
                 enable_augumentation: bool = True,
                 random_rot_range: Tuple[float, float] = (-90, 90),
                 num_views: int = 4,
                 pc_noise_std: float = 0,
                 # volume config
                 volume_size: int = 128,
                 volume_group: str = 'nocs_winding_number_field',
                 tsdf_clip_value: Optional[float] = None,
                 volume_absolute_value: bool = False,
                 include_volume: bool = False,
                 # random seed
                 static_epoch_seed: bool = False,
                 is_test: bool = False,
                 get_vis=False,
                 sample_edge=False,
                 # catch all
                 **kwargs):
        """
        If static_point_sample is True, the points sampled for each index
        will be identical each time being called.
        """
        super().__init__()
        self.input_type = input_type
        self.data_path = data_path
        self.num_pc_sample = num_pc_sample
        self.num_volume_sample = num_volume_sample
        self.num_surface_sample = num_surface_sample
        self.num_mc_surface_sample = num_mc_surface_sample
        # mixed sampling config
        self.surface_sample_ratio = surface_sample_ratio
        self.surface_sample_std = surface_sample_std
        # surface sample noise
        self.surface_normal_noise_ratio = surface_normal_noise_ratio
        self.surface_normal_std = surface_normal_std
        # data augumentaiton
        self.enable_augumentation = enable_augumentation
        self.random_rot_range = random_rot_range
        self.num_views = num_views
        assert (num_views > 0)
        self.pc_noise_std = pc_noise_std
        # volume config
        self.volume_size = volume_size
        self.volume_group = volume_group
        self.tsdf_clip_value = tsdf_clip_value
        self.volume_absolute_value = volume_absolute_value
        self.include_volume = include_volume
        self.volume_task_space = False
        # random seed
        self.static_epoch_seed = static_epoch_seed
        self.is_test = is_test
        meta_path = os.path.join(self.data_path, "summary_new.h5")
        if os.path.exists(meta_path):
            summary = read_h5_dict(meta_path)
            self.cloth_nocs_aabb = summary['nocs_aabb']
            self.length = summary['len']
        print("Dataset length: ", self.length)
        self.get_vis = get_vis
        self.sample_edge = sample_edge

    def __len__(self):
        return self.length

    def data_io(self, idx: int) -> dict:
        """
        read dataset from disk

        Args:
            idx:

        Returns:
            pc
            pc_nocs
            mesh_verts
            mesh_nocs
            mesh_faces
            wnf
        """

        idx = idx % self.length
        tsdf_clip_value = self.tsdf_clip_value
        volume_absolute_value = self.volume_absolute_value
        num_volume_sample = self.num_volume_sample
        load_list = ['depth', 'pc_sim', 'pc_nocs', 'cloth_sim_verts',
                     'img_nocs', 'img_pc', 'cloth_id'
                     ]
        if self.is_test:
            load_list.append("rgb")
        data = read_h5_dict(
            os.path.join(self.data_path, 'data', f'{idx:05d}_3d.h5'),
            load_list
        )
        cloth_id = data['cloth_id']
        nocs_data = read_h5_dict(os.path.join(self.data_path, 'nocs', f'{cloth_id:05d}_3d.h5'),
                                 ['cloth_nocs_verts', 'cloth_faces_tri', 'wnf'])

        data.update(nocs_data)
        if self.input_type == 'depth':
            data['depth'] = torch.tensor(data['depth']).unsqueeze(0)
            data['img_nocs'] = torch.tensor(data['img_nocs']).permute(2, 0, 1)
            data['img_pc'] = torch.tensor(data['img_pc']).permute(2, 0, 1)
        else:
            data.pop('img_nocs')
            data.pop('img_pc')

        if 'rgb' in data:
            data['rgb'] = torch.tensor(data['rgb']).permute(2, 0, 1)
        else:
            data['rgb'] = None
        # volume io
        if num_volume_sample > 0:
            raw_volume = data['wnf']
            volume = np.expand_dims(raw_volume, (0, 1)).astype(np.float32)
            if tsdf_clip_value is not None:
                scaled_volume = volume / tsdf_clip_value
                volume = np.clip(scaled_volume, -1, 1)
            if volume_absolute_value:
                volume = np.abs(volume)
            data['volume'] = volume

        data.pop('wnf')
        return data

    def get_base_data(self, idx: int, data_in: dict) -> dict:
        """
        Get non-volumetric data as numpy arrays
        """
        num_pc_sample = self.num_pc_sample
        static_epoch_seed = self.static_epoch_seed

        # train with pointcloud as input
        dataset_idx = np.array([idx])

        if self.input_type == 'pc':
            seed = idx if static_epoch_seed else None
            rs = np.random.RandomState(seed=seed)
            all_idxs = np.arange(len(data_in['pc_sim']))
            selected_idxs = rs.choice(all_idxs, size=num_pc_sample, replace=False)

            pc_sim = data_in['pc_sim'][selected_idxs].astype(np.float32)
            pc_nocs = data_in['pc_nocs'][selected_idxs].astype(np.float32)
            data = {
                'x': np.copy(pc_sim),
                'y': pc_nocs,
                'pos': pc_sim,
                'dataset_idx': dataset_idx,
                'rgb': data_in['rgb']
            }
        else:
            # train with depth as input
            data = {
                'depth': data_in['depth'],
                'img_nocs': data_in['img_nocs'],
                'img_pc': data_in['img_pc'],
                'dataset_idx': dataset_idx,
                'rgb': data_in['rgb']
            }
            if 'vis' in data_in:
                data['vis'] = data_in['vis']
        if self.is_test:
            data['cloth_sim_verts'] = data_in['cloth_sim_verts']
            data['cloth_nocs_verts'] = data_in['cloth_nocs_verts']
            data['cloth_tri'] = data_in['cloth_faces_tri']

        return data

    def get_volume_sample(self, idx: int, data_in: dict) -> dict:
        """
        Sample points by interpolating the volume.
        """
        volume_group = self.volume_group
        num_volume_sample = self.num_volume_sample
        static_epoch_seed = self.static_epoch_seed
        surface_sample_ratio = self.surface_sample_ratio
        surface_sample_std = self.surface_sample_std

        seed = idx if static_epoch_seed else None
        rs = np.random.RandomState(seed=seed)
        query_points = None
        if surface_sample_ratio == 0:
            query_points = rs.uniform(low=0, high=1, size=(num_volume_sample, 3)).astype(np.float32)
        else:
            # combine uniform and near-surface sample
            num_uniform_sample = int(num_volume_sample * surface_sample_ratio)
            num_surface_sample = num_volume_sample - num_uniform_sample
            uniform_query_points = rs.uniform(
                low=0, high=1, size=(num_uniform_sample, 3)).astype(np.float32)

            cloth_nocs_verts = data_in['cloth_nocs_verts']
            cloth_faces_tri = data_in['cloth_faces_tri']
            sampled_barycentric, sampled_face_idxs = mesh_sample_barycentric(
                verts=cloth_nocs_verts, faces=cloth_faces_tri,
                num_samples=num_surface_sample, seed=seed)
            sampled_faces = cloth_faces_tri[sampled_face_idxs]
            sampled_nocs_points = barycentric_interpolation(
                sampled_barycentric, cloth_nocs_verts, sampled_faces)
            surface_noise = rs.normal(loc=(0,) * 3, scale=(surface_sample_std,) * 3,
                                      size=(num_surface_sample, 3))
            surface_query_points = sampled_nocs_points + surface_noise
            mixed_query_points = np.concatenate(
                [uniform_query_points, surface_query_points], axis=0).astype(np.float32)
            query_points = np.clip(mixed_query_points, 0, 1)

        sample_values_torch = nocs_grid_sample(
            torch.from_numpy(data_in['volume']),
            torch.from_numpy(query_points))
        sample_values_numpy = sample_values_torch.view(
            sample_values_torch.shape[:-1]).numpy()
        if volume_group == 'nocs_occupancy_grid':
            # make sure number is either 0 or 1 for occupancy
            sample_values_numpy = (sample_values_numpy > 0.1).astype(np.float32)
        data = {
            'volume_query_points': query_points,
            'gt_volume_value': sample_values_numpy
        }
        data = self.reshape_for_batching(data)
        return data

    def get_surface_sample(self, idx: int, data_in: dict) -> dict:
        num_surface_sample = self.num_surface_sample
        static_epoch_seed = self.static_epoch_seed
        volume_task_space = self.volume_task_space
        surface_normal_noise_ratio = self.surface_normal_noise_ratio
        surface_normal_std = self.surface_normal_std

        cloth_nocs_verts = data_in['cloth_nocs_verts']
        cloth_sim_verts = data_in['cloth_sim_verts']
        cloth_faces_tri = data_in['cloth_faces_tri']

        seed = idx if static_epoch_seed else None
        sampled_barycentric, sampled_face_idxs = mesh_sample_barycentric(
            verts=cloth_nocs_verts, faces=cloth_faces_tri,
            num_samples=num_surface_sample, seed=seed)

        sampled_faces = cloth_faces_tri[sampled_face_idxs]

        sampled_nocs_points = barycentric_interpolation(
            sampled_barycentric, cloth_nocs_verts, sampled_faces)
        sampled_sim_points = barycentric_interpolation(
            sampled_barycentric, cloth_sim_verts, sampled_faces)
        if surface_normal_noise_ratio != 0:
            # add noise in normal direction
            num_points_with_noise = int(num_surface_sample * surface_normal_noise_ratio)
            nocs_vert_normals = igl.per_vertex_normals(cloth_nocs_verts, cloth_faces_tri)
            sampled_nocs_normals = barycentric_interpolation(
                sampled_barycentric[:num_points_with_noise], nocs_vert_normals,
                sampled_faces[:num_points_with_noise])
            rs = np.random.RandomState(seed)
            offset = rs.normal(0, surface_normal_std, size=num_points_with_noise)
            offset_vectors = (sampled_nocs_normals.T * offset).T
            aug_sampled_nocs_points = sampled_nocs_points[:num_points_with_noise] + offset_vectors
            sampled_nocs_points[:num_points_with_noise] = aug_sampled_nocs_points

        # !!!
        # Pytorch Geometric concatinate all elements by dim 0 except
        # attributes with word (face/index), which will be concnatinated by last dimention
        # face is in surface. Use surf instead.
        data = {
            'surf_query_points': sampled_nocs_points,
            'gt_sim_points': sampled_sim_points
        }
        if self.get_vis:
            data['vis'] = barycentric_interpolation(
                sampled_barycentric, data_in['vis'][:, None], sampled_faces)
        data = self.reshape_for_batching(data)
        return data

    def conv_augmentation(self, depth, img_nocs, img_pc, rs, rgb=None):
        random_rot_range = self.random_rot_range
        rot_angle = rs.uniform(*random_rot_range)
        out = {'depth': TF.affine(depth, angle=rot_angle, translate=[0, 0], scale=1.0, shear=0),
               'img_nocs': TF.affine(img_nocs, angle=rot_angle, translate=[0, 0], scale=1.0, shear=0),
               'img_pc': TF.affine(img_pc, angle=rot_angle, translate=[0, 0], scale=1.0, shear=0),
               'rot_angle': rot_angle, 'rgb': rgb}
        if rgb is not None:
            out['rgb'] = TF.affine(rgb, angle=rot_angle, translate=[0, 0], scale=1.0, shear=0)
        return out

    def rotation_augumentation(self, idx: int, data: dict) -> dict:
        static_epoch_seed = self.static_epoch_seed
        random_rot_range = self.random_rot_range
        volume_task_space = self.volume_task_space
        assert (len(random_rot_range) == 2)
        assert (random_rot_range[0] <= random_rot_range[-1])

        out_data = dict(data)
        seed = idx if static_epoch_seed else None
        rs = np.random.RandomState(seed=seed)
        # sample random rotation, apply to depths, img_nocs, img_pc
        # get cloth_uv, get y, get pos
        if self.enable_augumentation:
            if self.input_type == 'depth':
                conv_input = self.conv_augmentation(data['depth'], data['img_nocs'], data['img_pc'],
                                                    rs, data['rgb'])
                out_data.update(conv_input)

                rot_angle = -conv_input['rot_angle']
            else:
                rot_angle = rs.uniform(*random_rot_range)
        else:
            rot_angle = 0
        cloth_uv = torch.nonzero(out_data['depth'].squeeze()).long()
        out_data.update({
            'y': out_data['img_nocs'][:, cloth_uv[:, 0], cloth_uv[:, 1]].T,
            'pos': out_data['img_pc'][:, cloth_uv[:, 0], cloth_uv[:, 1]].T,
            'cloth_uv': cloth_uv,
            'rgb': out_data['rgb']
        })
        out_data.pop('img_pc')
        rot_mat = Rotation.from_euler(
            'y', rot_angle, degrees=True
        ).as_matrix().astype(np.float32)
        if not volume_task_space:
            sim_point_keys = ['pos', 'gt_sim_points', 'cloth_sim_verts']

            for key in sim_point_keys:
                if key in out_data:
                    out_data[key] = (out_data[key] @ rot_mat.T)
        # record augumentation matrix for eval
        out_data['input_aug_rot_mat'] = rot_mat.reshape((1,) + rot_mat.shape)
        return out_data

    def noise_augumentation(self, idx: int, data: dict) -> dict:
        pc_noise_std = self.pc_noise_std
        static_epoch_seed = self.static_epoch_seed

        if self.input_type == "depth":
            noise = torch.randn((1, 200, 200)) * pc_noise_std
            mask = data['depth'] > 0
            data['depth'] += mask * noise
            data['img_pc'][1:2] -= mask * noise
        elif self.input_type == "pc":
            noise = np.random.randn(*data['x'].shape) * pc_noise_std
            data['x'][:, 1] += noise[:, 1]
            data['pos'][:, 1] += noise[:, 1]
        return data

    def reshape_for_batching(self, data: dict) -> dict:
        out_data = dict()
        for key, value in data.items():
            if key != 'edges':
                out_data[key] = value.reshape((1,) + value.shape)
        return out_data

    # def __getitem__(self, idx: int) -> Data:
    #     include_volume = self.include_volume
    #     num_volume_sample = self.num_volume_sample
    #     num_surface_sample = self.num_surface_sample
    #     data_in = self.data_io(idx)
    #     data = self.get_base_data(idx, data_in=data_in)
    #     if num_volume_sample > 0:
    #         volume_sample_data = self.get_volume_sample(idx, data_in=data_in)
    #         data.update(volume_sample_data)
    #     if num_surface_sample > 0:
    #         surface_sample_data = self.get_surface_sample(idx, data_in=data_in)
    #         data.update(surface_sample_data)
    #     data['input_aug_rot_mat'] = np.expand_dims(np.eye(3, dtype=np.float32), axis=0)
    #
    #     data = self.noise_augumentation(idx, data=data)
    #     data = self.rotation_augumentation(idx, data=data)
    #
    #     if include_volume:
    #         data['volume'] = data_in['volume']
    #
    #     data_torch = dict(
    #         (x[0], torch.from_numpy(x[1]) if isinstance(x[1], np.ndarray) else x[1]) for x in
    #         data.items())
    #
    #     pg_data = MyData(**data_torch)
    #     return pg_data

    def __getitem__(self, idx: int) -> Data:
        while 1:
            include_volume = self.include_volume
            num_volume_sample = self.num_volume_sample
            num_surface_sample = self.num_surface_sample
            data_in = self.data_io(idx)
            data = self.get_base_data(idx, data_in=data_in)
            if num_volume_sample > 0:
                volume_sample_data = self.get_volume_sample(idx, data_in=data_in)
                data.update(volume_sample_data)
            if num_surface_sample > 0:
                surface_sample_data = self.get_surface_sample(idx, data_in=data_in)
                data.update(surface_sample_data)
            data['input_aug_rot_mat'] = np.expand_dims(np.eye(3, dtype=np.float32), axis=0)

            data = self.noise_augumentation(idx, data=data)
            data = self.rotation_augumentation(idx, data=data)
            if 'pos' in data:
                if data['pos'].shape[0] < 1000:
                    idx += 1
                    continue
                rand_ind = torch.randperm(data['pos'].shape[0])[:1000]
                data['pack_pc'] = data['pos'][rand_ind]
            break

        if include_volume:
            data['volume'] = data_in['volume']

        data_torch = dict(
            (x[0], torch.from_numpy(x[1]) if isinstance(x[1], np.ndarray) else x[1]) for x in
            data.items())

        pg_data = MyData(**data_torch)
        return pg_data


# data modules
# ============
class ConvImplicitWNFDataModule(pl.LightningDataModule):
    def __init__(self, cfg, **kwargs):
        """
        dataset_split: tuple of (train, val, test)
        """
        super().__init__()
        assert (len(kwargs['dataset_split']) == 3)
        self.kwargs = kwargs
        self.cfg = cfg
        self.train_path = os.path.join('dataset', cfg.ds, 'train')
        self.valid_path = os.path.join('dataset', cfg.ds, 'val')
        self.test_path = os.path.join('dataset', cfg.ds, 'test')
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dl_cls = DataLoader_g
        self.follow_batch = ['cloth_tri', 'cloth_nocs_verts', 'edges']
        self.cloth_nocs_aabb = None
        self.initialize_nocs_aabb()

    def initialize_nocs_aabb(self):
        meta_path = os.path.join(self.train_path, "summary_new.h5")
        if os.path.exists(meta_path):
            summary = read_h5_dict(meta_path)
            self.cloth_nocs_aabb = summary['nocs_aabb'].astype(np.float32)

    def prepare_data(self, test_only=False):
        if self.cloth_nocs_aabb is None:
            self.initialize_nocs_aabb()
        # TODO: change split method
        kwargs = self.kwargs
        split_seed = kwargs['split_seed']
        dataset_split = kwargs['dataset_split']

        train_args = dict(kwargs)
        train_args['static_epoch_seed'] = False
        if not test_only:
            self.train_dataset = ConvImplicitWNFDataset(self.cfg.input_type, self.train_path, **train_args)

            train_args['enable_augumentation'] = False
            train_args['is_test'] = True
            self.val_dataset = ConvImplicitWNFDataset(self.cfg.input_type, self.valid_path, **train_args)
            self.val_dataset.static_epoch_seed = True
        train_args['enable_augumentation'] = False
        train_args['is_test'] = True
        self.test_dataset = ConvImplicitWNFDataset(self.cfg.input_type, self.test_path, **train_args)
        self.test_dataset.static_epoch_seed = True

    def train_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = self.cfg.num_workers
        prefetch_factor = 5 if num_workers > 0 else 2
        collate_func = my_collate if self.cfg.input_type != 'pc' else None
        dataloader = self.dl_cls(self.train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 prefetch_factor=prefetch_factor,
                                 collate_fn=collate_func,
                                 follow_batch=self.follow_batch
                                 )
        return dataloader

    def val_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = self.cfg.num_workers
        prefetch_factor = 5 if num_workers > 0 else 2
        collate_func = my_collate if self.cfg.input_type != 'pc' else None
        dataloader = self.dl_cls(self.val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 prefetch_factor=prefetch_factor,
                                 collate_fn=collate_func,
                                 follow_batch=self.follow_batch
                                 )
        return dataloader

    def test_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = self.cfg.num_workers
        prefetch_factor = 5 if num_workers > 0 else 2
        collate_func = my_collate if self.cfg.input_type != 'pc' else None
        dataloader = self.dl_cls(self.test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 prefetch_factor=prefetch_factor,
                                 collate_fn=collate_func,
                                 follow_batch=self.follow_batch
                                 )
        return dataloader


class GarmentnetsDataloader(object):
    def __init__(self, cloth_type):
        zarr_path = f"/data/zixuanhu/garmentnets_dataset.zarr/{cloth_type}"
        metadata_cache_dir = "~/local/.cache/metadata_cache_dir2"
        rawpath = '/home/zixuanhu/occlusion_reasoning/dataset/cloth3d'

        path = pathlib.Path(os.path.expanduser(zarr_path))
        if os.path.exists(path):
            self.id2sample = \
                pickle.load(
                    open("/home/zixuanhu/occlusion_reasoning/cache/sample_id_mapping.pkl", "rb"))[
                    cloth_type]
            root = zarr.open(str(path.absolute()), mode='r')
            samples_group = root['samples']
            groups_df = file_attr_cache(zarr_path, cache_dir=metadata_cache_dir)(_get_groups_df)(
                samples_group)
            groups_df['idx'] = np.arange(len(groups_df))
            self.cloth_cano_aabb = root['summary/cloth_canonical_aabb_union'][:].astype(np.float32)
            self.nocs_aabb = root['summary/cloth_canonical_aabb_union'][:]
            # self.sim_aabb = root['summary/cloth_aabb_union'][:]
            self.samples_group = samples_group
            self.groups_df = groups_df
        else:
            print('Warning zarr path doesn\'t exist')
        # TODO: clean up this part. It is used in conv_wnf
        self.raw_base = os.path.join(rawpath, "nocs", cloth_type)
        if os.path.exists(os.path.join(rawpath, "mesh", cloth_type, 'meta.yaml')):
            meta_info = OmegaConf.load(os.path.join(rawpath, "mesh", cloth_type, 'meta.yaml'))
            self.rescale = meta_info['scale']
            self.nocs_aabb = np.array(meta_info['nocs_aabb'])
            # self.ori_aabb = np.array(meta_info['ori_aabb'])
            # self.to_canon = meta_info['to_canon2']
            print('nocs aabb ', self.nocs_aabb)
            # print('to canon ', self.to_canon)

        # _, self.flatten_states = pickle.load(open(f"{self.raw_base}/flat_states.pkl", "rb"))

    def get_sample(self, garment_id):
        # obj_path = f"{self.raw_base}/{garment_id:04d}.obj"
        # TODO: NOCS for flatten
        # V = readOBJ(obj_path)[0]
        # # When loaded into the simulator, mesh is rotated
        # V2 = to_nocs(V, self.sim_aabb)
        # V = rotate_particles([180, 0, 0], V)
        # print(garment_id)
        dataset_idx = self.id2sample[garment_id]
        row = self.groups_df.iloc[dataset_idx]
        group = self.samples_group[row.group_key]
        mesh_group = group['mesh']
        V = mesh_group['cloth_nocs_verts'][:]
        wnf = group['volume']['nocs_winding_number_field'][str(128)][:]
        return V, wnf
