# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by RainbowSecret (yhyuan@pku.edu.cn)
# ------------------------------------------------------------------------------

import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from garmentnets.components.hrnet_config import MODEL_CONFIGS
# from flow.backbones.sampling_head import GaussianBottleneck, GaussianMixBottleneck, MixtureDetHead, \
#     make_normal2, make_normal

logger = logging.getLogger('hrnet_backbone')
import torch

__all__ = ['hrnet18', 'hrnet32', 'hrnet48']
ALIGN_CORNERS = False


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class TriBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(TriBlock, self).__init__()
        self.block = BasicBlock(inplanes, inplanes)
        self.out_layer = conv1x1(inplanes, planes)

    def forward(self, x):
        x = self.block(x)
        x = self.out_layer(x)
        return x

#
# class CVAEBottleneck(nn.Module):
#     """Encoder of CVAE with privileged information(gt_image)"""
#
#     def __init__(self, cfg, last_inp_channels, num_sample, latent_size, arch='hr18', normal_param='std', alpha=0.5):
#         super(CVAEBottleneck, self).__init__()
#         net = hrnet18 if arch == 'hr18' else hrnet32
#         self.enc = net(cfg, head_mode=1, in_dim=3, role='enc', particle_filter=None)
#         self.num_sample = num_sample
#         self.latent_size = latent_size
#         self.to_dist_obs = TriBlock(last_inp_channels, latent_size * 2)
#         self.to_dist_gt = conv1x1(self.enc.last_inp_channels, latent_size * 2)
#         self.mix = TriBlock(inplanes=last_inp_channels + latent_size, planes=last_inp_channels)
#         self.alpha = 0.5
#         if normal_param == 'std':
#             self.normal = make_normal
#         else:
#             self.normal = make_normal2
#
#     def forward(self, in_x, batch):
#         x_shape = list(in_x.size())
#         sample_shape = [x_shape[0] * self.num_sample, self.latent_size] + x_shape[2:]
#         alpha = self.alpha
#         if 'img_nocs' in batch:
#             gt_image = batch['img_nocs']
#             # gt_image = gt_image.view(-1, *gt_image.size()[2:])  # (B*A)x4xHxW
#             y = self.enc(gt_image)['feat']
#             # print("dist ", y.mean())
#             gt_dist, gt_dist_stat = self.normal(self.to_dist_gt(y), min_std=1e-4, chunk_dim=1)
#         else:
#             gt_dist_stat = None
#         obs_dist, obs_dist_stat = self.normal(self.to_dist_obs(in_x), min_std=1e-4, chunk_dim=1)
#
#         # Sample latent vector so first dimension becomes Bxnum_sample
#         if self.training:
#             # Transpose so that samples of the same observation become adjacent
#             # z = gt_dist.rsample([self.num_sample]).transpose(0, 1).reshape(sample_shape)
#             z1 = gt_dist.rsample([self.num_sample])
#             z2 = obs_dist.rsample([self.num_sample])
#             k = torch.rand((1, x_shape[0], 1, 1, 1), dtype=z1.dtype, device=z1.device)
#             z = torch.where(k > alpha, z1, z2).reshape(sample_shape)
#         else:
#             z = obs_dist.rsample([self.num_sample]).reshape(sample_shape)
#
#         # Expand deterministic feature so it matches the sample
#         in_x = in_x.unsqueeze(0).expand(self.num_sample, -1, -1, -1, -1).reshape(-1, *x_shape[1:])
#         x = torch.cat([in_x, z], 1)
#         x = self.mix(x)
#
#         return x, {'gt_dist': gt_dist_stat, 'obs_dist': obs_dist_stat}
#

class HighResolutionNet(nn.Module):

    def __init__(self,
                 cfg,
                 hr_cfg,
                 norm_layer=None,
                 in_dim=2,
                 out_dim=192,
                 # role='canon',
                 # head_mode=1,
                 # particle_filter=None,
                 # num_sample=1,
                 # latent_size=32,
                 # sample_arch='hr18',
                 # normal_param='std',
                 ):
        super(HighResolutionNet, self).__init__()
        self.cfg = cfg
        # self.num_sample = num_sample
        # self.role = role
        # self.head_mode = head_mode
        # self.particle_filter = particle_filter
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        # stem net
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = self.norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = self.norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = hr_cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # stage 2
        self.stage2_cfg = hr_cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = hr_cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = hr_cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.flow_head = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.last_inp_channels,
                    out_channels=self.last_inp_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                self.norm_layer(self.last_inp_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=self.last_inp_channels,
                    out_channels=out_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.UpsamplingBilinear2d(scale_factor=4)
            )

        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        self.norm_layer(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, batch=None, mode='all'):
        B = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        x = torch.cat([x[0], x1, x2, x3], 1)
        # if self.role == 'canon3d' and self.particle_filter == 'cvae':
        #     x, latent = self.latent_sampler(x, batch)
        # if self.role == 'enc':
        #     return {'feat': x}
        # 18:  270 * 50 * 50
        # 32:  480 * 50 * 50
        # if 'canon' in self.role:
        pred = self.flow_head(x)
            # if self.particle_filter == 'multimodal':
            #     out = pred
            #     # if mode == 'uniform':
            #     #     select = torch.randint(self.num_sample, size=(B,), device=x3.device)
            #     #     select = select * B + torch.arange(B, device=x3.device)
            #     #     out['feat'] = out['feat'][select]
            #     #     out['flow_inv'] = out['flow_inv'][select]
            #     #     out['density'] = out['density'][select]
            #     out['feat'] = self.upsample1(out['feat'])
            # else:
        out = {'flow_inv': pred,
               'feat': self.upsample1(x)}
        # elif self.role == 'flow':
        #     flow = self.flow_head(x)
        #     flow = flow.view(-1, 2, *flow.size()[2:])
        #     out = {'flow': flow}
        #     if self.cfg.occlu_depth != 'const':
        #         out['depth'] = self.depth_head(x).view(-1, 1, *flow.size()[2:])
        #     if self.cfg.pred_vis:
        #         out['vis'] = torch.sigmoid(self.vis_head(x)).view(-1, 1, *flow.size()[2:])
        # if self.particle_filter == 'cvae':
        #     out['latent'] = latent
        return out


def _hrnet(arch, cfg, **kwargs):
    model = HighResolutionNet(cfg, MODEL_CONFIGS[arch], **kwargs)
    return model


def hrnet18(cfg, **kwargs):
    r"""HRNet-18 model
    """
    return _hrnet('hrnet18', cfg, **kwargs)


def hrnet32(cfg, **kwargs):
    r"""HRNet-32 model
    """
    return _hrnet('hrnet32', cfg, **kwargs)


def hrnet48(cfg, pretrained=True, progress=True, **kwargs):
    r"""HRNet-48 model
    """
    return _hrnet('hrnet48', cfg, pretrained, progress,
                  **kwargs)
