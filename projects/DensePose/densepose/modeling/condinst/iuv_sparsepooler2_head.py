import numpy as np
from typing import Dict, List, Optional, Any

import torch, pdb, os, pickle
import torch.nn as nn
from torch.nn import functional as F

from fvcore.nn import sigmoid_focal_loss_jit
import fvcore.nn.weight_init as weight_init

# from adet.layers import conv_with_kaiming_uniform
# from adet.utils.comm import aligned_bilinear
from detectron2.layers import Conv2d, ShapeSpec, get_norm, ConvTranspose2d
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import ImageList, Instances, Boxes

from densepose.layers import conv_with_kaiming_uniform, deform_conv, sparse_conv_with_kaiming_uniform
from densepose.utils.comm import compute_locations, compute_grid, aligned_bilinear
import spconv
from ..roi_heads import DensePoseDeepLabHead
from .. import (
    build_densepose_data_filter,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    densepose_inference,
)
from lambda_networks import LambdaLayer
from .iuv_head import get_embedder
from .iuv_pooler2_head import CoordGlobalIUVPooler2Head
from ..utils import initialize_module_params
import pdb

INF = 100000000

def build_iuv_sparsepooler2_head(cfg, input_shape):
    # return GlobalIUVHead(cfg)
    return CoordGlobalIUVSparsePooler2Head(cfg, input_shape=input_shape)

class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3), # just like nn.Conv3d but don't support group and all([d > 1, s > 1])
            nn.BatchNorm1d(64), # non-spatial layers can be used directly in SparseSequential.
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # when use submanifold convolutions, their indices can be shared to save indices generation time.
            spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConvTranspose3d(64, 64, 3, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.ToDense(), # convert spconv tensor to dense and convert it to NCHW format.
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int() # unlike torch, this library only accept int coordinates.
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)# .dense()


class DecoderSparse(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], in_features: torch.Tensor, pe_dim=0):
        super(DecoderSparse, self).__init__()

        # fmt: off
        self.in_features      = in_features
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        # num_classes           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NUM_CLASSES
        num_classes = 75
        conv_dims             = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS
        self.common_stride    = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE
        norm                  = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM
        num_lambda_layer = cfg.MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER
        lambda_layer_r = cfg.MODEL.CONDINST.IUVHead.LAMBDA_LAYER_R
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

        # self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        

        if num_lambda_layer>0:
            self.comb_pe_conv = LambdaLayer(
                dim = conv_dims+pe_dim,
                dim_out = conv_dims,
                r = lambda_layer_r,         # the receptive field for relative positional encoding (23 x 23)
                dim_k = 16,
                heads = 4,
                dim_u = 4
            )
        else:
            self.comb_pe_conv = Conv2d(
                conv_dims+pe_dim,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dims),
                activation=F.relu,
            )
        # weight_init.c2_msra_fill(self.comb_pe_conv)

        self.densepose_head = build_densepose_head(cfg, conv_dims)

        self.predictor = Conv2d(
            cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM, num_classes, 1, stride=1, padding=0
        )
        initialize_module_params(self.predictor)
        # weight_init.c2_msra_fill(self.predictor)

    def forward(self, features: List[torch.Tensor], rel_coord: Any, abs_coord: Any, fg_mask: Any):
        assert fg_mask.min()==0, "the fg_mask is all 1"
        for i, _ in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[i])
            else:
                x = x + self.scale_heads[i](features[i])
        if rel_coord is not None:
            x = torch.cat([x,rel_coord], dim=1)
        if abs_coord is not None:
            x = torch.cat([x,abs_coord], dim=1)

        # pdb.set_trace()
        if rel_coord is not None or abs_coord is not None:
            if isinstance(self.comb_pe_conv, LambdaLayer):
                x = self.comb_pe_conv(x)
            else:
                x = self.comb_pe_conv(x)



        # pdb.set_trace()
        ## dense to sparse
        # x = spconv.SparseConvTensor.from_dense((x*fg_mask).permute([0,2,3,1])) # must be NHWC tensor

        ## dense to sparse
        N, C, H, W = x.shape
        coord = compute_grid(H, W, device=x.device, norm=False)
        sparse_coord_batch = []
        sparse_feat_batch = []
        for n in range(N):
            m = fg_mask[n:n+1]
            x_indices = coord[0][m[0,0]>0]
            y_indices = coord[1][m[0,0]>0]
            b_indices = torch.ones_like(x_indices)*n
            sparse_coord = torch.stack([b_indices,y_indices,x_indices],dim=-1).int()
            sparse_coord_batch.append(sparse_coord)
            sparse_feat = x[n:n+1]
            sparse_feat = sparse_feat[m.expand_as(sparse_feat)>0].reshape([C,-1]).permute([1,0])
            sparse_feat_batch.append(sparse_feat)
        sparse_coord_batch = torch.cat(sparse_coord_batch,dim=0)
        sparse_feat_batch = torch.cat(sparse_feat_batch,dim=0)
        # pdb.set_trace()
        x = spconv.SparseConvTensor(sparse_feat_batch, sparse_coord_batch, (H,W), N)



        # x = x * fg_mask
        x = self.densepose_head(x)
        x = x.dense()
        x = self.predictor(x)
        return x


class CoordGlobalIUVSparsePooler2Head(CoordGlobalIUVPooler2Head):
    def _init_densepose_head(self, cfg, input_shape):
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.use_rel_coords = cfg.MODEL.CONDINST.IUVHead.REL_COORDS
        self.use_abs_coords = cfg.MODEL.CONDINST.IUVHead.ABS_COORDS
        self.pos_emb_num_freqs = cfg.MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS
        self.use_pos_emb = self.pos_emb_num_freqs>0
        if self.use_pos_emb:
            self.position_embedder, self.position_emb_dim = get_embedder(multires=self.pos_emb_num_freqs, input_dims=2)
        self.pe_dim_all = 0
        if self.use_rel_coords:
            self.pe_dim_all += self.position_emb_dim
        if self.use_abs_coords:
            self.pe_dim_all += self.position_emb_dim
        self.decoder = DecoderSparse(cfg, input_shape, self.in_features, self.pe_dim_all)


        dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        dp_pooler_sampling_ratio   = 0
        dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        # in_channels = [input_shape[f].channels for f in self.in_features][0]
        self.densepose_pooler = ROIPooler(
            output_size=dp_pooler_resolution,
            scales=dp_pooler_scales,
            sampling_ratio=dp_pooler_sampling_ratio,
            pooler_type=dp_pooler_type,
        )
        # self.densepose_head = build_densepose_head(cfg, in_channels)
        # self.densepose_predictor = build_densepose_predictor(
        #     cfg, self.densepose_head.n_out_channels
        # )
        self.densepose_losses = build_densepose_losses(cfg)





