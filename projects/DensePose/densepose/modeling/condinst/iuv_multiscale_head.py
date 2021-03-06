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
from lambda_networks import LambdaLayer
from .iuv_head import get_embedder
import pdb

INF = 100000000

def build_iuv_multiscale_head(cfg):
    return CoordGlobalIUVMultiscaleHead(cfg)

## Inspired by HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation
class CoordGlobalIUVMultiscaleHead(nn.Module):
    def __init__(self, cfg, use_rel_coords=True):
        super().__init__()
        self.num_outputs = cfg.MODEL.CONDINST.IUVHead.OUT_CHANNELS
        norm = cfg.MODEL.CONDINST.IUVHead.NORM
        num_convs = cfg.MODEL.CONDINST.IUVHead.NUM_CONVS
        num_lambda_layer = cfg.MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER
        assert num_lambda_layer<=num_convs
        channels = cfg.MODEL.CONDINST.IUVHead.CHANNELS
        self.norm_feat = cfg.MODEL.CONDINST.IUVHead.NORM_FEATURES
        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))
        self.iuv_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.use_rel_coords = cfg.MODEL.CONDINST.IUVHead.REL_COORDS
        self.use_abs_coords = cfg.MODEL.CONDINST.IUVHead.ABS_COORDS
        # pdb.set_trace()
        # if self.use_rel_coords:
        #     self.in_channels = channels + 2
        # else:
        self.pos_emb_num_freqs = cfg.MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS
        self.use_pos_emb = self.pos_emb_num_freqs>0
        if self.use_pos_emb:
            self.position_embedder, self.position_emb_dim = get_embedder(multires=self.pos_emb_num_freqs, input_dims=2)
            self.in_channels = channels + self.position_emb_dim
        else:
            self.in_channels = channels + 2

        if self.use_abs_coords:
            if self.use_pos_emb:
                self.in_channels += self.position_emb_dim
            else:
                self.in_channels += 2


        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        tower = []
        if num_lambda_layer>0:
            layer = LambdaLayer(
                dim = self.in_channels,
                dim_out = channels,
                r = 23,         # the receptive field for relative positional encoding (23 x 23)
                dim_k = 16,
                heads = 4,
                dim_u = 4
            )
            tower.append(layer)
        else:
            tower.append(conv_block(
                self.in_channels, channels, 3, 1
            ))

        for i in range(1,num_convs-1):
            if i<num_lambda_layer:
                layer = LambdaLayer(
                    dim = channels,
                    dim_out = channels,
                    r = 23,         # the receptive field for relative positional encoding (23 x 23)
                    dim_k = 16,
                    heads = 4,
                    dim_u = 4
                )
                tower.append(layer)
            else:
                tower.append(conv_block(
                    channels, channels, 3, 1
                ))

        self.add_module('tower', nn.Sequential(*tower))

        self.mid_res_conv = conv_block(channels, channels, 3, 1)
        self.mid_res_out = nn.Conv2d(channels, self.num_outputs, 1)

        self.low_res_conv = conv_block(channels, channels, 3, 2)
        self.low_res_out = nn.Conv2d(channels, self.num_outputs, 1)

        deconv_block = conv_with_kaiming_uniform(norm, activation=True, use_deconv=True)
        self.high_res_conv = deconv_block(channels, channels, 3, 2)
        self.high_res_out = nn.Conv2d(channels, self.num_outputs, 1)

        # tower.append(nn.Conv2d(
        #     channels, max(self.num_outputs, 1), 1
        # ))

        # self.densepose_losses = build_densepose_losses(cfg)

    # def forward(self, s_logits, iuv_feats, iuv_feat_stride, relative_coords, instances):


    #     N, _, H, W = iuv_feats.size()
    #     rel_coord = torch.zeros([N,2,H,W], device=iuv_feats.device).to(dtype=iuv_feats.dtype)
    #     abs_coord = torch.zeros([N,2,H,W], device=iuv_feats.device).to(dtype=iuv_feats.dtype)

    #     if self.use_rel_coords: 
    #         # locations = compute_locations(
    #         #     iuv_feats.size(2), iuv_feats.size(3),
    #         #     stride=iuv_feat_stride, device=iuv_feats.device
    #         # )
    #         # # n_inst = len(instances)

    #         im_inds = instances.im_inds


    #         # instance_locations = instances.locations
    #         # relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
    #         # relative_coords = relative_coords.permute(0, 2, 1).float()
    #         # soi = self.sizes_of_interest.float()[instances.fpn_levels]
    #         # relative_coords = relative_coords / soi.reshape(-1, 1, 1)
    #         # relative_coords = relative_coords.to(dtype=iuv_feats.dtype)
    #         # rel_coord_list = []
    #         for idx in range(N):
    #             if idx in im_inds:
    #                 cc = relative_coords[im_inds==idx,].reshape(-1, 2, H, W)
    #                 # assert s_logits.shape[1]==1
    #                 ss = s_logits[im_inds==idx,-1:]
    #                 # coord = torch.sum(cc*ss, dim=0, keepdim=True) \
    #                 #       / (torch.sum(ss, dim=0, keepdim=True)+1e-7)
    #                 coord = torch.mean(cc*ss, dim=0, keepdim=True) 
    #                 rel_coord[idx:idx+1] = coord #.reshape(1, 2, H, W)
    #                 # pdb.set_trace()
    #                 # import imageio
    #                 # imageio.imwrite("tmp/cc.png",cc[0,0].detach().cpu().numpy())
    #                 # imageio.imwrite("tmp/ss.png",ss[0,0].detach().cpu().numpy())
    #                 # imageio.imwrite("tmp/cc_ss.png",(cc*ss)[0,0].detach().cpu().numpy())
    #                 # imageio.imwrite("tmp/ss_sum.png",torch.sum(ss, dim=0, keepdim=True)[0,0].detach().cpu().numpy())
    #                 # imageio.imwrite("tmp/coord_mean.png",coord[0,0].detach().cpu().numpy())
    #             # rel_coord_list.append(rel_coord)
    #         # assert self.norm_feat
    #         if self.norm_feat:
    #             # iuv_feats = iuv_feats/iuv_feats.max()*2.0 - 1.0
    #             iuv_feats = iuv_feats/20.0

    #         if self.use_pos_emb:
    #             rel_coord = self.position_embedder(rel_coord)


    #     iuv_head_inputs = torch.cat([rel_coord, iuv_feats], dim=1) 
    #     # else:
    #     #     iuv_head_inputs = iuv_feats

    #     if self.use_abs_coords: 
    #         abs_coord = compute_grid(H, W, device=iuv_feats.device)
    #         iuv_head_inputs = torch.cat([abs_coord, iuv_head_inputs], dim=1) 


    def forward(self, s_logits, iuv_feats, iuv_feat_stride, rel_coord, instances):
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

        iuv_logit_mid_res = self.tower(iuv_head_inputs)
        iuv_logit_mid_res = self.mid_res_conv(iuv_logit_mid_res)
        iuv_logit_mid_out = self.mid_res_out(iuv_logit_mid_res)

        iuv_logit_low_res = self.low_res_conv(iuv_logit_mid_res)
        iuv_logit_low_out = self.low_res_out(iuv_logit_low_res)

        iuv_logit_high_res = self.high_res_conv(iuv_logit_mid_res)
        iuv_logit_high_out = self.high_res_out(iuv_logit_high_res)

        # assert iuv_feat_stride >= self.iuv_out_stride
        # assert iuv_feat_stride % self.iuv_out_stride == 0
        # iuv_logit = aligned_bilinear(iuv_logit, int(iuv_feat_stride / self.iuv_out_stride))

        return [iuv_logit_high_out, iuv_logit_mid_out, iuv_logit_low_out]





