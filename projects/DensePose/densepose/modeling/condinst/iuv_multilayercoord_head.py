from typing import Dict
import math

import torch
from torch import nn
import torch.nn.functional as F

from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import ShapeSpec

# from adet.layers import conv_with_kaiming_uniform
# from adet.utils.comm import aligned_bilinear
from densepose.layers import conv_with_kaiming_uniform
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

def build_iuv_multilayercoord_head(cfg):
    return CoordGlobalIUVMultiLayerCoordHead(cfg)

## Inspired by HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation
class CoordGlobalIUVMultiLayerCoordHead(nn.Module):
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
        # pdb.set_trace()
        # if self.use_rel_coords:
        #     self.in_channels = channels + 2
        # else:
        self.pos_emb_num_freqs = cfg.MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS
        self.use_pos_emb = self.pos_emb_num_freqs>0
        extra_channels = 0
        if self.use_pos_emb:
            self.position_embedder, self.position_emb_dim = get_embedder(multires=self.pos_emb_num_freqs, input_dims=2)
            extra_channels += self.position_emb_dim
        else:
            extra_channels += 2

        if self.use_abs_coords:
            if self.use_pos_emb:
                extra_channels += self.position_emb_dim
            else:
                extra_channels += 2

        # pdb.set_trace()
        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        cnt = 0
        self.layers = []
        if num_lambda_layer>0:
            layer = LambdaLayer(
                dim = agg_channels + extra_channels,
                dim_out = channels,
                r = lambda_layer_r,         # the receptive field for relative positional encoding (23 x 23)
                dim_k = 16,
                heads = 4,
                dim_u = 4
            )
        else:
            layer = conv_block(channels+extra_channels, channels, 3, 1)
        setattr(self, 'layer_{}'.format(cnt), layer)
        self.layers.append(layer)
        cnt += 1

        for i in range(1,num_convs):
            if i<num_lambda_layer:
                layer = LambdaLayer(
                    dim = channels+extra_channels,
                    dim_out = channels,
                    r = lambda_layer_r,         # the receptive field for relative positional encoding (23 x 23)
                    dim_k = 16,
                    heads = 4,
                    dim_u = 4
                )
            else:
                layer = conv_block(channels+extra_channels, channels, 3, 1)
            setattr(self, 'layer_{}'.format(cnt), layer)
            self.layers.append(layer)
            cnt += 1

        layer = nn.Conv2d(channels+extra_channels, max(self.num_outputs, 1), 1)
        setattr(self, 'layer_{}'.format(cnt), layer)
        self.layers.append(layer)

    def _torch_dilate(self, binary_img, kernel_size=3, mode='nearest'):
        if not hasattr(self, 'dilate_kernel'):
            # self.dilate_kernel = torch.Tensor(torch.ones([kernel_size,kernel_size]), device=binary_img.device)[None,None,...]
            self.dilate_kernel = torch.ones([1,1,kernel_size,kernel_size], device=binary_img.device)
        # pdb.set_trace()
        pad = nn.ReflectionPad2d(int(kernel_size//2))
        out = torch.clamp(torch.nn.functional.conv2d(pad(binary_img), self.dilate_kernel, padding=0), 0, 1)
        out = F.interpolate(out, size=binary_img.shape[2:], mode=mode)
        return out

    def forward(self, s_logits, iuv_feats, iuv_feat_stride, instances, mask_out_bg=False):
        N, _, H, W = iuv_feats.size()

        if mask_out_bg:
            fg_mask = s_logits.detach()
            fg_mask_list = []
            for i in range(N):
                fg_mask_list.append(torch.max(fg_mask[instances.im_inds==i], dim=0, keepdim=True)[0])
            fg_mask = torch.cat(fg_mask_list, dim=0).detach()
            # if mask_out_bg_feats=="hard":
            fg_mask = (fg_mask>0.05).float()
            fg_mask = self._torch_dilate(fg_mask, kernel_size=3)
        else:
            fg_mask = torch.ones([N,1,H,W], device=s_logits.device)

        rel_coord = torch.zeros([N,2,H,W], device=iuv_feats.device).to(dtype=iuv_feats.dtype)
        abs_coord = torch.zeros([N,2,H,W], device=iuv_feats.device).to(dtype=iuv_feats.dtype)

        if self.use_rel_coords: 
            locations = compute_locations(
                iuv_feats.size(2), iuv_feats.size(3),
                stride=iuv_feat_stride, device=iuv_feats.device
            )
            # n_inst = len(instances)

            im_inds = instances.im_inds


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
            # assert self.norm_feat
            if self.norm_feat:
                # iuv_feats = iuv_feats/iuv_feats.max()*2.0 - 1.0
                iuv_feats = iuv_feats/20.0

            if self.use_pos_emb:
                rel_coord = self.position_embedder(rel_coord)
            coord = rel_coord


        # iuv_head_inputs = torch.cat([rel_coord, iuv_feats], dim=1) 
        # else:
        #     iuv_head_inputs = iuv_feats

        if self.use_abs_coords: 
            abs_coord = compute_grid(H, W, device=iuv_feats.device)[None,...].repeat(N,1,1,1)
            if self.use_pos_emb:
                abs_coord = self.position_embedder(abs_coord)
            coord = torch.cat([abs_coord, coord], dim=1) 

        x = iuv_feats
        for layer in self.layers:
            # pdb.set_trace()
            x = layer(torch.cat([coord, x], dim=1) *fg_mask)
        iuv_logit = x

        # iuv_logit = self.tower(iuv_head_inputs)

        assert iuv_feat_stride >= self.iuv_out_stride
        assert iuv_feat_stride % self.iuv_out_stride == 0
        iuv_logit = aligned_bilinear(iuv_logit, int(iuv_feat_stride / self.iuv_out_stride))

        return iuv_logit

