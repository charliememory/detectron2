import numpy as np
from typing import Dict, List, Optional, Any

import torch, pdb, os, pickle
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

from fvcore.nn import sigmoid_focal_loss_jit
import fvcore.nn.weight_init as weight_init

# from adet.layers import conv_with_kaiming_uniform
# from adet.utils.comm import aligned_bilinear
from detectron2.layers import Conv2d, ShapeSpec, get_norm, ConvTranspose2d
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import ImageList, Instances, Boxes

from densepose.layers import conv_with_kaiming_uniform, deform_conv
from densepose.utils.comm import compute_locations, compute_grid, aligned_bilinear
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
from ..utils import initialize_module_params
import pdb

INF = 100000000

def build_iuv_pooler2_head(cfg, input_shape):
    # return GlobalIUVHead(cfg)
    return CoordGlobalIUVPooler2Head(cfg, input_shape=input_shape)


class Decoder(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], in_features: torch.Tensor, pe_dim=0):
        super(Decoder, self).__init__()

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

    def forward(self, features: List[torch.Tensor], iuv_feats: torch.Tensor, rel_coord: Any, abs_coord: Any, fg_mask: Any, ins_mask_list=None):
        for i, _ in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[i])
            else:
                x = x + self.scale_heads[i](features[i])
        if rel_coord is not None:
            x = torch.cat([x,rel_coord], dim=1)
        if abs_coord is not None:
            x = torch.cat([x,abs_coord], dim=1)
        # if skeleton_feats is not None:
        #     x = torch.cat([x,skeleton_feats], dim=1)

        # pdb.set_trace()
        if rel_coord is not None or abs_coord is not None:
            x = self.comb_pe_conv(x)
        x = x * fg_mask
        x = self.densepose_head(x)
        x = self.predictor(x)
        return x


class CoordGlobalIUVPooler2Head(nn.Module):
    def __init__(self, cfg, input_shape=None):
        super().__init__()

        self._init_densepose_head(cfg, input_shape)

    def _init_densepose_head(self, cfg, input_shape):
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.use_rel_coords = cfg.MODEL.CONDINST.IUVHead.REL_COORDS
        self.use_abs_coords = cfg.MODEL.CONDINST.IUVHead.ABS_COORDS
        self.pos_emb_num_freqs = cfg.MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS
        self.use_gt_ins = cfg.MODEL.CONDINST.IUVHead.GT_INSTANCES
        self.inference_global_siuv = cfg.MODEL.CONDINST.INFERENCE_GLOBAL_SIUV
        self.add_skeleton_feat = cfg.MODEL.CONDINST.IUVHead.SKELETON_FEATURES
        self.use_pos_emb = self.pos_emb_num_freqs>0
        if self.use_pos_emb:
            self.position_embedder, self.position_emb_dim = get_embedder(multires=self.pos_emb_num_freqs, input_dims=2)
        self.pe_dim_all = 0
        if self.use_rel_coords:
            self.pe_dim_all += self.position_emb_dim
        if self.use_abs_coords:
            self.pe_dim_all += self.position_emb_dim
        if self.add_skeleton_feat:
            self.pe_dim_all += 55
        self.decoder = Decoder(cfg, input_shape, self.in_features, self.pe_dim_all)

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

    def forward(self, fpn_features, s_logits, iuv_feats, iuv_feat_stride, rel_coord, instances, fg_mask, gt_instances=None, ins_mask_list=None):
        # assert not self.use_abs_coords

        fea0 = fpn_features[self.in_features[0]]
        N, _, H, W = fea0.shape

        if self.use_rel_coords: 
            if self.use_pos_emb:
                rel_coord = self.position_embedder(rel_coord)
        else:
            rel_coord = None

        if self.use_abs_coords: 
            abs_coord = compute_grid(H, W, device=fea0.device)[None,...].repeat(N,1,1,1)
            if self.use_pos_emb:
                abs_coord = self.position_embedder(abs_coord)
        else:
            abs_coord = None

        features = [fpn_features[f] for f in self.in_features]

        if self.inference_global_siuv:
            assert not self.training

        if self.training:
            features = [self.decoder(features, iuv_feats, rel_coord, abs_coord, fg_mask, ins_mask_list)]
            proposal_boxes = [x.gt_boxes for x in gt_instances]
            features_dp = self.densepose_pooler(features, proposal_boxes)
            iuv_logits = features_dp
            # iuv_logit_global = features[0]
            return None, iuv_logits
        else:
            features = [self.decoder(features, iuv_feats, rel_coord, abs_coord, fg_mask, ins_mask_list)]
            # pdb.set_trace()

            if self.inference_global_siuv:
                iuv_logits = features[0]
                coarse_segm = s_logits
            else:
                # if isinstance(instances,Instances):
                # if self.use_gt_ins:
                #     proposal_boxes = [x.gt_boxes for x in gt_instances]
                # else:
                proposal_boxes = [instances.pred_boxes]
                # else:
                #     proposal_boxes = [x.pred_boxes for x in instances]
                features_dp = self.densepose_pooler(features, proposal_boxes)
                # pdb.set_trace()
                s_logit_list = []
                for idx in range(s_logits.shape[0]):
                    s_logit = self.densepose_pooler([s_logits[idx:idx+1]], [proposal_boxes[0][idx:idx+1]])
                    s_logit_list.append(s_logit)
                coarse_segm = torch.cat(s_logit_list,dim=0)
                # iuv_logit = torch.cat([torch.cat(s_logit_list,dim=0), features_dp], dim=1)
                # iuv_logit_global = features[0]
                iuv_logits = features_dp
                # print(instances.pred_boxes)
        # else:
        #     features = [self.decoder(features, iuv_feats, rel_coord, abs_coord, fg_mask, ins_mask_list)]
        #     proposal_boxes = [instances.pred_boxes]
        #     features_dp = self.densepose_pooler(features, proposal_boxes)
        #     iuv_logit = features_dp
        #     iuv_logit_global = features[0]


            return coarse_segm, iuv_logits






