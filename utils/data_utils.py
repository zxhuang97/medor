import collections
import os.path
import pdb
import time

import cv2
import h5py
import glob
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate_err_msg_format
from torch_geometric.data import Data
import torch_geometric

from utils.camera_utils import get_pointcloud
from utils.geometry_utils import readOBJ, quads2tris, get_barycentric_pc


def process_any_cloth(rgb, depth, matrix_world_to_camera,
                      input_type='depth',
                      coords=None,
                      cloth_id=None,
                      info_path='',
                      real_world=False,
                      normalize_func=None,
                      padding=0,
                      cloth_type="Trousers",
                      ):
    """
    Prepare the input for model inference and privileged information for evaluation

    Returns:
        depth
        pc_sim
        pc_nocs
        cloth_sim_verts
        cloth_nocs_verts
        cloth_faces_tri
        img_nocs
        img_pc
        wnf

        Maybe?
        visibility
    """
    rgb = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
    if real_world and normalize_func is not None:
        depth = normalize_func(depth)
    # todo: check if we need this
    # depth = depth * (depth > 0.5) * (depth < 0.6415)
    if padding > 0:
        new_depth = np.pad(depth, padding)
        depth = cv2.resize(new_depth, (200, 200), interpolation=cv2.INTER_NEAREST)

    pc_sim = get_pointcloud(depth, matrix_world_to_camera=matrix_world_to_camera)
    scale = (200 + 2 * padding) / 200
    if input_type == 'pc':
        all_idxs = np.arange(len(pc_sim))
        selected_idxs = np.random.choice(all_idxs, size=6000, replace=False)
        # aabb = gdloader.cloth_cano_aabb
        pc_sim_s = pc_sim[selected_idxs].astype(np.float32)
        # cloth_sim_aabb = aabb.reshape((1,) + aabb.shape)
        depth_s = cv2.resize(depth, (200, 200), interpolation=cv2.INTER_NEAREST)
        depth_s = torch.tensor(depth_s, dtype=torch.float32)
        data = {
            'rgb': rgb,
            'x': np.copy(pc_sim_s),
            'depth': depth_s.unsqueeze(0),
            'pos': pc_sim_s,
            # 'cloth_sim_aabb': cloth_sim_aabb
        }
    else:
        depth_s = cv2.resize(depth, (200, 200), interpolation=cv2.INTER_NEAREST)
        T_vec_pred = None
        ds_pc = get_pointcloud(depth_s, matrix_world_to_camera).astype(np.float32)
        depth_s = torch.tensor(depth_s, dtype=torch.float32)
        cloth_uv = torch.nonzero(depth_s).long()
        data = {
            'rgb': rgb,
            'depth': depth_s.unsqueeze(0),
            'pos': ds_pc,
            'cloth_uv': cloth_uv,
            'pc_sim': pc_sim,
            'scale': scale,
            'T_vec_pred': T_vec_pred
        }

    if not real_world:
        # nocs, wnf = gdloader.get_sample(cloth_id)
        # import pdb
        # pdb.set_trace()
        # info = read_h5_dict(info_path)
        # nocs, wnf = info['cloth_nocs_verts'], info['wnf']
        V, F = readOBJ(f"dataset/cloth3d/mesh/{cloth_type}/{cloth_id:04d}.obj")[:2]
        F = quads2tris(F)
        # TODO: visibility
        assert coords.shape[0] == V.shape[0], f"mesh mismatch {coords.shape[0]}  {V.shape[0]}"
        # bary_results = get_barycentric_pc(pc_sim, coords, F, {'nocs': nocs})
        # if input_type == 'pc':
        #     pc_nocs = bary_results['nocs'][selected_idxs]
        # else:
        #     all_uvs = np.array(depth.nonzero()).T
        #     canon_img = np.zeros((720, 720, 3), dtype=np.float32)
        #     canon_img[all_uvs[:, 0], all_uvs[:, 1]] = bary_results['nocs']
        #     canon_img = torch.tensor(cv2.resize(canon_img, (200, 200), interpolation=cv2.INTER_NEAREST))
        #     pc_nocs = canon_img[data['cloth_uv'][:, 0], data['cloth_uv'][:, 1]]

        data.update({
            # privileged information only for evaluation purpose
            # 'y': pc_nocs,
            'cloth_sim_verts': coords,
            # 'cloth_nocs_verts': nocs,
            'cloth_tri': F,
            # 'wnf': wnf
        })
    data_torch = dict(
        (x[0], torch.from_numpy(x[1]) if isinstance(x[1], np.ndarray) else x[1]) for x in data.items())

    return MyData(**data_torch)


class PrivilData(Data):
    """
    Encapsulation of multi-graphs for multi-step training
    ind: 0-(hor-1), type: vsbl or full
    Each graph contain:
        edge_index_{type}_{ind},
        x_{type}_{ind},
        edge_attr_{type}_{ind},
        gt_rwd_{type}_{ind}
        gt_accel_{type}_{ind}
        mesh_mapping_{type}_{ind}
    """

    def __init__(self, has_part=False, has_full=False, **kwargs):
        super(PrivilData, self).__init__(**kwargs)
        self.has_part = has_part
        self.has_full = has_full

    def __inc__(self, key, value, *args, **kwargs):
        if 'edge_index' in key:
            x = key.replace('edge_index', 'x')
            return self[x].size(0)
        elif 'mesh_mapping' in key:
            # add index of mesh matching by
            x = key.replace('partial_pc_mapped_idx', 'x')
            return self[x].size(0)
        else:
            return super().__inc__(key, value)


class AggDict(dict):
    def __init__(self, is_detach=True):
        """
        Aggregate numpy arrays or pytorch tensors
        :param is_detach: Whether to save numpy arrays in stead of torch tensors
        """
        super(AggDict).__init__()
        self.is_detach = is_detach

    def __getitem__(self, item):
        return self.get(item, 0)

    def add_item(self, key, value):
        if self.is_detach and torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if not isinstance(value, torch.Tensor):
            if isinstance(value, np.ndarray) or isinstance(value, np.number):
                assert value.size == 1
            else:
                assert isinstance(value, int) or isinstance(value, float)
        if key not in self.keys():
            self[key] = value
        else:
            self[key] += value

    def update_by_add(self, src_dict):
        for key, value in src_dict.items():
            self.add_item(key, value)

    def get_mean(self, prefix, count=1):
        avg_dict = {}
        for k, v in self.items():
            avg_dict[prefix + k] = v / count
        return avg_dict


def updateDictByAdd(dict1, dict2):
    '''
    update dict1 by dict2
    '''
    for k1, v1 in dict2.items():
        for k2, v2 in v1.items():
            dict1[k1][k2] += v2.cpu().item()
    return dict1


def get_index_before_padding(graph_sizes):
    ins_len = graph_sizes.max()
    pad_len = ins_len * graph_sizes.size(0)
    valid_len = graph_sizes.sum()
    accum = torch.zeros(1).cuda()
    out = []
    for gs in graph_sizes:
        new_ind = torch.range(0, gs - 1).cuda() + accum
        out.append(new_ind)
        accum += ins_len
    final_ind = torch.cat(out, dim=0)
    return final_ind.long()


class MyDataParallel(torch_geometric.nn.DataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name == 'module':
            return self._modules['module']
        else:
            return getattr(self.module, name)


def retrieve_data(data, key):
    """
    vsbl: [vsbl], full: [full], dual :[vsbl, full]
    """
    if isinstance(data, dict):
        identifier = '_{}'.format(key)
        out_data = {k.replace(identifier, ''): v for k, v in data.items() if identifier in k}
    return out_data


def read_h5_dict(path, data_names=None):
    for i in range(3):
        try:
            hf = h5py.File(path, 'r')
            data = {}
            data_names = hf.keys() if data_names is None else data_names
            for name in data_names:
                d = np.array(hf.get(name))
                data[name] = d
            hf.close()
            return data
        except:
            print('read h5 error ', path)
            time.sleep(0.2)
    return False


def store_h5_dict(path, data_dict):
    for i in range(5):
        try:
            hf = h5py.File(path, 'w')
            for k, v in data_dict.items():
                hf.create_dataset(k, data=v)
            hf.close()
            return
        except:
            time.sleep(0.1)
            if os.path.exists(path):
                print(path, " being used by other process")
                return
            print('store h5 error ', path)


class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['depth', 'rgb', 'img_nocs', 'pack_pc', 'T_vec_pred']:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


def my_collate(batch, key=None):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if key in ['cloth_uv', 'y', 'pos']:
            # if key in ['cloth_uv_n', 'uv_o']:
            return torch.cat(batch, 0, out=out)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, collections.abc.Mapping):
        result = {key: my_collate([d[key] for d in batch], key) for key in elem}
        batch_id = []
        if 'cloth_uv' in elem:
            out = None
            elem = elem['cloth_uv']
            # if torch.utils.data.get_worker_info() is not None:
            #     # If we're in a background process, concatenate directly into a
            #     # shared memory tensor to avoid an extra copy
            #     numel = sum([x['cloth_uv_n'].shape[0] for x in batch])
            #     storage = elem.storage()._new_shared(numel)
            #     out = elem.new(storage)
            for i, b in enumerate(batch):
                batch_id.append(torch.ones(b['cloth_uv'].shape[0], dtype=torch.long) * i)
            result['batch'] = torch.cat(batch_id, 0, out=out)

        return result

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def update_config(default_cfg, extra_params):
    for k, v in extra_params.items():
        OmegaConf.update(default_cfg, k, v)
    return default_cfg


def find_best_checkpoint(model_path):
    if 'ckpt' not in model_path:
        ckpt_pattern = f'{model_path}/*.ckpt'
    else:
        ckpt_pattern = model_path
    all_ckpts = glob.glob(ckpt_pattern)
    best_ckpt = sorted(all_ckpts)[-1]

    for ck in all_ckpts:
        if 'early' in ck:
            # pdb.set_trace()
            best_ckpt = ck
            print('early')
    ckpt = torch.load(best_ckpt)

    print('The best checkpoint of current model is ' + best_ckpt + ' with epoch ' + str(ckpt['epoch']))
    return best_ckpt
