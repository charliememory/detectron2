from typing import Dict
import math

import torch
from torch import nn

from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import ShapeSpec

# from adet.layers import conv_with_kaiming_uniform
# from adet.utils.comm import aligned_bilinear
from detectron2.layers import ConvTranspose2d
from densepose.layers import conv_with_kaiming_uniform, deform_conv
from densepose.utils.comm import compute_locations, compute_grid, aligned_bilinear
# from .. import (
#     build_densepose_data_filter,
#     build_densepose_head,
#     build_densepose_losses,
#     build_densepose_predictor,
#     densepose_inference,
# )
from lambda_networks import LambdaLayer
import pdb

INF = 100000000


# Positional encoding (section 5.1)
# Ref: NeRF
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], 1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


def build_iuv_head(cfg):
    # return GlobalIUVHead(cfg)
    return CoordGlobalIUVHead(cfg)


class CoordGlobalIUVHead(nn.Module):
    def __init__(self, cfg, use_rel_coords=True):
        super().__init__()
        self.num_outputs = cfg.MODEL.CONDINST.IUVHead.OUT_CHANNELS
        norm = cfg.MODEL.CONDINST.IUVHead.NORM
        num_convs = cfg.MODEL.CONDINST.IUVHead.NUM_CONVS
        num_lambda_layer = cfg.MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER
        lambda_layer_r = cfg.MODEL.CONDINST.IUVHead.LAMBDA_LAYER_R
        num_dcn_layer = cfg.MODEL.CONDINST.IUVHead.NUM_DCN_LAYER
        assert num_lambda_layer<=num_convs

        agg_channels = cfg.MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS
        channels = cfg.MODEL.CONDINST.IUVHead.CHANNELS
        self.norm_feat = cfg.MODEL.CONDINST.IUVHead.NORM_FEATURES
        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))
        self.iuv_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.use_rel_coords = cfg.MODEL.CONDINST.IUVHead.REL_COORDS
        self.use_abs_coords = cfg.MODEL.CONDINST.IUVHead.ABS_COORDS
        self.use_down_up_sampling = cfg.MODEL.CONDINST.IUVHead.DOWN_UP_SAMPLING
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


        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        partial_conv_block = conv_with_kaiming_uniform(norm, activation=True, use_partial_conv=True)
        deform_conv_block = conv_with_kaiming_uniform(norm, activation=True, use_deformable=True)

        tower = []
        if self.use_partial_conv:
            # pdb.set_trace()
            layer = partial_conv_block(self.in_channels, channels, 3, 1)
            tower.append(layer)
            self.in_channels = channels

        if num_lambda_layer>0:
            layer = LambdaLayer(
                dim = self.in_channels,
                dim_out = channels,
                r = lambda_layer_r,         # the receptive field for relative positional encoding (23 x 23)
                dim_k = 16,
                heads = 4,
                dim_u = 4
            )
            tower.append(layer)
        else:
            tower.append(conv_block(
                self.in_channels, channels, 3, 1
            ))
        if num_dcn_layer>0:
            tower.append(deform_conv_block(
                    channels, channels, 3, 1
            ))

        if self.use_down_up_sampling:
            for i in range(1,num_convs):
                if i==1:
                    tower.append(conv_block(
                        channels, channels*2, 3, 2
                    ))
                else:
                    tower.append(conv_block(
                        channels*2, channels*2, 3, 1
                    ))

            tower.append(ConvTranspose2d(
                channels*2, self.num_outputs, 4, stride=2, padding=int(4 / 2 - 1)
            ))
        else:
            for i in range(1,num_convs):
                tower.append(conv_block(
                    channels, channels, 3, 1
                ))
            tower.append(nn.Conv2d(
                channels, max(self.num_outputs, 1), 1
            ))

        self.add_module('tower', nn.Sequential(*tower))

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
        iuv_logit = self.tower(iuv_head_inputs)

        assert iuv_feat_stride >= self.iuv_out_stride
        assert iuv_feat_stride % self.iuv_out_stride == 0
        iuv_logit = aligned_bilinear(iuv_logit, int(iuv_feat_stride / self.iuv_out_stride))

        return iuv_logit



class GlobalIUVHead(nn.Module):
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
