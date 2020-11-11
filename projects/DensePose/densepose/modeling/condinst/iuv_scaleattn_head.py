from typing import Dict
import math

import torch
from torch import nn
import torch.nn.functional as F

from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import ShapeSpec, Conv2d
from detectron2.layers.batch_norm import get_norm

# from adet.layers import conv_with_kaiming_uniform
# from adet.utils.comm import aligned_bilinear
from densepose.layers import conv_with_kaiming_uniform, PartialConv2d
from densepose.utils.comm import compute_locations, compute_grid, aligned_bilinear
# from .. import (
#     build_densepose_data_filter,
#     build_densepose_head,
#     build_densepose_losses,
#     build_densepose_predictor,
#     densepose_inference,
# )
from lambda_networks import LambdaLayer
from .iuv_head import get_embedder
import pdb

INF = 100000000

def build_iuv_scaleattn_head(cfg):
    return CoordGlobalIUVScaleAttnHead(cfg)

## Inspired by HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation
class CoordGlobalIUVScaleAttnHead(nn.Module):
    def __init__(self, cfg, use_rel_coords=True):
        super().__init__()
        self.num_outputs = cfg.MODEL.CONDINST.IUVHead.OUT_CHANNELS
        norm = cfg.MODEL.CONDINST.IUVHead.NORM
        num_convs = cfg.MODEL.CONDINST.IUVHead.NUM_CONVS
        num_lambda_layer = cfg.MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER
        lambda_layer_r = cfg.MODEL.CONDINST.IUVHead.LAMBDA_LAYER_R
        assert num_lambda_layer<=num_convs
        
        agg_channels = cfg.MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS
        channels = cfg.MODEL.CONDINST.IUVHead.CHANNELS
        self.norm_feat = cfg.MODEL.CONDINST.IUVHead.NORM_FEATURES
        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))
        self.iuv_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.use_rel_coords = cfg.MODEL.CONDINST.IUVHead.REL_COORDS
        self.use_abs_coords = cfg.MODEL.CONDINST.IUVHead.ABS_COORDS
        self.use_partial_conv = cfg.MODEL.CONDINST.IUVHead.PARTIAL_CONV
        self.use_partial_norm = cfg.MODEL.CONDINST.IUVHead.PARTIAL_NORM
        # pdb.set_trace()
        # if self.use_rel_coords:
        #     self.in_channels = channels + 2
        # else:
        self.pos_emb_num_freqs = cfg.MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS
        self.use_pos_emb = self.pos_emb_num_freqs>0

        if self.use_pos_emb:
            self.position_embedder, self.position_emb_dim = get_embedder(multires=self.pos_emb_num_freqs, input_dims=2)
            self.in_channels = agg_channels + self.position_emb_dim
        else:
            self.in_channels = agg_channels + 2

        if self.use_abs_coords:
            if self.use_pos_emb:
                self.in_channels += self.position_emb_dim
            else:
                self.in_channels += 2

        if self.use_partial_conv:
            conv_block = conv_with_kaiming_uniform(norm, activation=True, use_partial_conv=True)
        else:
            conv_block = conv_with_kaiming_uniform(norm, activation=True)
            # pdb.set_trace()
        conv_block_bn = conv_with_kaiming_uniform("BN", activation=True)

        # tower_attn = []
        # tower_attn.append(conv_block_bn(
        #     self.position_emb_dim, 32, 3, 1
        # ))
        # tower_attn.append(nn.Conv2d(
        #     32, 3, 3, stride=1, padding=1
        # ))
        # self.add_module('tower_attn', nn.Sequential(*tower_attn))

        num_layer = 3

        tower0 = []
        if num_lambda_layer>0:
            layer = LambdaLayer(
                dim = self.in_channels,
                dim_out = channels,
                r = lambda_layer_r,         # the receptive field for relative positional encoding (23 x 23)
                dim_k = 8,
                heads = 4,
                dim_u = 4
            )
            tower0.append(layer)
        else:
            tower0.append(conv_block(
                self.in_channels, channels, 3, 1
            ))
        for i in range(num_layer):
            tower0.append(conv_block(
                channels, channels, 3, 1
            ))
        self.add_module('tower0', nn.Sequential(*tower0))

        tower1 = []
        if num_lambda_layer>0:
            layer = LambdaLayer(
                dim = self.in_channels,
                dim_out = channels,
                r = lambda_layer_r,         # the receptive field for relative positional encoding (23 x 23)
                dim_k = 8,
                heads = 4,
                dim_u = 4
            )
            tower1.append(layer)
        else:
            tower1.append(conv_block(
                self.in_channels, channels, 3, 1
            ))
        for i in range(num_layer):
            tower1.append(conv_block(
                channels, channels, 3, 1
            ))
        self.add_module('tower1', nn.Sequential(*tower1))

        tower2 = []
        if num_lambda_layer>0:
            layer = LambdaLayer(
                dim = self.in_channels,
                dim_out = channels,
                r = lambda_layer_r,         # the receptive field for relative positional encoding (23 x 23)
                dim_k = 8,
                heads = 4,
                dim_u = 4
            )
            tower2.append(layer)
        else:
            tower2.append(conv_block(
                self.in_channels, channels, 3, 1
            ))
        for i in range(num_layer):
            tower2.append(conv_block(
                channels, channels, 3, 1
            ))
        self.add_module('tower2', nn.Sequential(*tower2))

        tower_out = []
        for i in range(num_convs-num_layer-1):
            if i==0:
                tower_out.append(conv_block(
                    channels*3, channels, 1, 1
                ))
            else:
                tower_out.append(conv_block(
                    channels, channels, 3, 1
                ))
        self.add_module('tower_out', nn.Sequential(*tower_out))


    def forward(self, fpn_features, s_logits, iuv_feats, iuv_feat_stride, rel_coord, instances, fg_mask=None, gt_instances=None):
        N, _, H, W = iuv_feats.size()

        if self.use_rel_coords: 
            if self.use_pos_emb:
                rel_coord = self.position_embedder(rel_coord)
        else:
            rel_coord = torch.zeros([N,2,H,W], device=iuv_feats.device).to(dtype=iuv_feats.dtype)
        iuv_head_inputs = torch.cat([rel_coord, iuv_feats], dim=1) 

        if self.use_abs_coords: 
            abs_coord = compute_grid(H, W, device=iuv_feats.device)[None,...].repeat(N,1,1,1)
            if self.use_pos_emb:
                abs_coord = self.position_embedder(abs_coord)
        else:
            abs_coord = torch.zeros([N,2,H,W], device=iuv_feats.device).to(dtype=iuv_feats.dtype)
        iuv_head_inputs = torch.cat([abs_coord, iuv_head_inputs], dim=1)

        iuv_head_inputs0 = iuv_head_inputs
        iuv_logit0 = self.tower0(iuv_head_inputs0)
        iuv_head_inputs1 = F.avg_pool2d(iuv_head_inputs0,kernel_size=3,stride=2)
        iuv_logit1 = self.tower1(iuv_head_inputs1)
        iuv_logit1 = F.interpolate(iuv_logit1, size=iuv_logit0.shape[-2:])
        iuv_head_inputs2 = F.avg_pool2d(iuv_head_inputs1,kernel_size=3,stride=2)
        iuv_logit2 = self.tower2(iuv_head_inputs2)
        iuv_logit2 = F.interpolate(iuv_logit2, size=iuv_logit0.shape[-2:])

        # attn = F.softmax(self.tower_attn(rel_coord), dim=1)
        # pdb.set_trace()
        # iuv_logit = iuv_logit0*attn[:,0:1] + iuv_logit1*attn[:,1:2] + iuv_logit2*attn[:,2:3]
        iuv_logit = torch.cat([iuv_logit0,iuv_logit1,iuv_logit2], dim=1)

        iuv_logit = self.tower_out(iuv_logit)

        assert iuv_feat_stride >= self.iuv_out_stride
        assert iuv_feat_stride % self.iuv_out_stride == 0
        iuv_logit = aligned_bilinear(iuv_logit, int(iuv_feat_stride / self.iuv_out_stride))

        return iuv_logit
