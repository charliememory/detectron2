from typing import Dict
import math

import torch
from torch import nn

from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import ShapeSpec

# from adet.layers import conv_with_kaiming_uniform
# from adet.utils.comm import aligned_bilinear
from densepose.layers import conv_with_kaiming_uniform
from densepose.utils.comm import compute_locations, aligned_bilinear
# from .. import (
#     build_densepose_data_filter,
#     build_densepose_head,
#     build_densepose_losses,
#     build_densepose_predictor,
#     densepose_inference,
# )
import pdb

INF = 100000000


def build_iuv_head(cfg):
    # return GlobalIUVHeadAfterMaskBranch(cfg)
    return CoordGlobalIUVHeadAfterMaskBranch(cfg)


class CoordGlobalIUVHeadAfterMaskBranch(nn.Module):
    def __init__(self, cfg, disable_rel_coords=False):
        super().__init__()
        self.num_outputs = cfg.MODEL.CONDINST.IUVHead.OUT_CHANNELS
        norm = cfg.MODEL.CONDINST.IUVHead.NORM
        num_convs = cfg.MODEL.CONDINST.IUVHead.NUM_CONVS
        channels = cfg.MODEL.CONDINST.IUVHead.CHANNELS
        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))
        self.in_channels = channels + 2
        self.iuv_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = disable_rel_coords

        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        tower = []
        tower.append(conv_block(
            self.in_channels, channels, 3, 1
        ))
        for i in range(1,num_convs):
            tower.append(conv_block(
                channels, channels, 3, 1
            ))
        tower.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        self.add_module('tower', nn.Sequential(*tower))

        # self.densepose_losses = build_densepose_losses(cfg)

    def forward(self, s_logits, iuv_feats, iuv_feat_stride, instances):

        locations = compute_locations(
            iuv_feats.size(2), iuv_feats.size(3),
            stride=iuv_feat_stride, device=iuv_feats.device
        )
        # n_inst = len(instances)

        im_inds = instances.im_inds

        N, _, H, W = iuv_feats.size()
        rel_coord = torch.zeros([N,2,H,W], device=iuv_feats.device).to(dtype=iuv_feats.dtype)

        if not self.disable_rel_coords: 
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=iuv_feats.dtype)
            # rel_coord_list = []
            for idx in range(N):
                if idx in im_inds:
                    cc = relative_coords[im_inds==idx,].reshape(-1, 2, H, W)
                    # assert s_logits.shape[1]==1
                    ss = s_logits[im_inds==idx,-1:]
                    # coord = torch.sum(cc*ss, dim=0, keepdim=True) \
                    #       / (torch.sum(ss, dim=0, keepdim=True)+1e-7)
                    coord = torch.mean(cc*ss, dim=0, keepdim=True) 
                    rel_coord[idx:idx+1] = coord #.reshape(1, 2, H, W)
                    # pdb.set_trace()
                    # import imageio
                    # imageio.imwrite("tmp/cc.png",cc[0,0].detach().cpu().numpy())
                    # imageio.imwrite("tmp/ss.png",ss[0,0].detach().cpu().numpy())
                    # imageio.imwrite("tmp/cc_ss.png",(cc*ss)[0,0].detach().cpu().numpy())
                    # imageio.imwrite("tmp/ss_sum.png",torch.sum(ss, dim=0, keepdim=True)[0,0].detach().cpu().numpy())
                    # imageio.imwrite("tmp/coord_mean.png",coord[0,0].detach().cpu().numpy())
                # rel_coord_list.append(rel_coord)
            # pdb.set_trace()
            iuv_head_inputs = torch.cat([rel_coord, iuv_feats], dim=1) 
        else:
            iuv_head_inputs = iuv_feats





        iuv_logit = self.tower(iuv_head_inputs)

        assert iuv_feat_stride >= self.iuv_out_stride
        assert iuv_feat_stride % self.iuv_out_stride == 0
        iuv_logit = aligned_bilinear(iuv_logit, int(iuv_feat_stride / self.iuv_out_stride))

        return iuv_logit


class GlobalIUVHeadAfterMaskBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_outputs = cfg.MODEL.CONDINST.IUVHead.OUT_CHANNELS
        norm = cfg.MODEL.CONDINST.IUVHead.NORM
        num_convs = cfg.MODEL.CONDINST.IUVHead.NUM_CONVS
        channels = cfg.MODEL.CONDINST.IUVHead.CHANNELS
        self.iuv_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE

        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        tower = []
        for i in range(num_convs):
            tower.append(conv_block(
                channels, channels, 3, 1
            ))
        tower.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        self.add_module('tower', nn.Sequential(*tower))

        # self.densepose_losses = build_densepose_losses(cfg)

    def forward(self, iuv_feats, iuv_feat_stride, instances=None):
        iuv_logit = self.tower(iuv_feats)

        assert iuv_feat_stride >= self.iuv_out_stride
        assert iuv_feat_stride % self.iuv_out_stride == 0
        iuv_logit = aligned_bilinear(iuv_logit, int(iuv_feat_stride / self.iuv_out_stride))

        return iuv_logit
