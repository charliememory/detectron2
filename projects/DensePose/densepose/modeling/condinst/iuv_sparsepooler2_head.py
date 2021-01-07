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
from densepose.layers import sparse_conv_with_kaiming_uniform, SAN_BottleneckGN, SAN_BottleneckGN_GatedEarly, SAN_BottleneckGN_Gated
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
        self.use_agg_feat    = cfg.MODEL.CONDINST.IUVHead.USE_AGG_FEATURES
        self.use_ins_gn = cfg.MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN
        self.checkpoint_grad_num = cfg.MODEL.CONDINST.CHECKPOINT_GRAD_NUM
        agg_channels = cfg.MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS
        self.use_aux_global_s = cfg.MODEL.CONDINST.AUX_SUPERVISION_GLOBAL_S
        self.use_aux_global_skeleton = cfg.MODEL.CONDINST.AUX_SUPERVISION_GLOBAL_SKELETON
        if self.use_aux_global_s:
            num_classes += 1
        if self.use_aux_global_skeleton:
            "to check"
            num_classes += 55
        self.predictor_conv_type = cfg.MODEL.CONDINST.IUVHead.PREDICTOR_TYPE
        self.use_dropout = cfg.MODEL.CONDINST.IUVHead.DROPOUT
        self.use_san = cfg.MODEL.CONDINST.IUVHead.USE_SAN
        self.san_type = cfg.MODEL.CONDINST.SAN_TYPE
        # fmt: on

        # if not self.use_agg_feat:
        #     self.scale_heads = []
        #     for in_feature in self.in_features:
        #         head_ops = []
        #         head_length = max(
        #             1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
        #         )
        #         for k in range(head_length):
        #             conv = Conv2d(
        #                 feature_channels[in_feature] if k == 0 else conv_dims,
        #                 conv_dims,
        #                 kernel_size=3,
        #                 stride=1,
        #                 padding=1,
        #                 bias=not norm,
        #                 norm=get_norm(norm, conv_dims),
        #                 activation=F.relu,
        #             )
        #             weight_init.c2_msra_fill(conv)
        #             head_ops.append(conv)
        #             if feature_strides[in_feature] != self.common_stride:
        #                 head_ops.append(
        #                     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        #                 )
        #         self.scale_heads.append(nn.Sequential(*head_ops))
        #         self.add_module(in_feature, self.scale_heads[-1])

        # self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        

        if num_lambda_layer>0:
            self.comb_pe_conv = LambdaLayer(
                dim = agg_channels+pe_dim,
                dim_out = agg_channels,
                r = lambda_layer_r,         # the receptive field for relative positional encoding (23 x 23)
                dim_k = 16,
                heads = 4,
                dim_u = 4
            )
        else:
            self.comb_pe_conv = Conv2d(
                agg_channels+pe_dim,
                agg_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, agg_channels),
                activation=F.relu,
            )

        if self.use_san:
            # sa_type = 1 ## 0: pairwise; 1: patchwise
            sa_type = 1
            if self.san_type=="SAN_BottleneckGN":
                san_func = SAN_BottleneckGN
            elif self.san_type=="SAN_BottleneckGN_GatedEarly":
                san_func = SAN_BottleneckGN_GatedEarly
            elif self.san_type=="SAN_BottleneckGN_Gated":
                san_func = SAN_BottleneckGN_Gated
            self.san_blk_1 = san_func(sa_type, agg_channels, agg_channels // 16, agg_channels // 4, agg_channels, 8, kernel_size=7, stride=1)

        # weight_init.c2_msra_fill(self.comb_pe_conv)
        if self.use_dropout:
            self.dropout_layer = nn.Dropout2d(0.25)

        self.densepose_head = build_densepose_head(cfg, agg_channels)

        if self.predictor_conv_type=="conv":
            self.predictor = Conv2d(
                cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM, num_classes, 1, stride=1, padding=0
            )
            initialize_module_params(self.predictor)
        elif self.predictor_conv_type=="dcnv1":
            self.predictor = deform_conv.DFConv2d(
                cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM, num_classes,
                with_modulated_dcn=False, kernel_size=3
            )
        elif self.predictor_conv_type=="dcnv2":
            self.predictor = deform_conv.DFConv2d(
                cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM, num_classes,
                with_modulated_dcn=True, kernel_size=3
            )
        elif self.predictor_conv_type=="dcnv2Conv":
            self.predictor = []
            self.predictor.append(deform_conv.DFConv2d(
                cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM, cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM,
                with_modulated_dcn=True, kernel_size=3
            ))
            self.predictor.append(Conv2d(
                cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM, num_classes, 1, stride=1, padding=0
            ))
            initialize_module_params(self.predictor[-1])
            self.predictor = nn.Sequential(*self.predictor)
        elif self.predictor_conv_type=="dcnv2ResConv":
            self.predictor = []
            self.predictor.append(deform_conv.DeformBottleneckBlock(
                cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM, cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM,
                bottleneck_channels=cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM,
                deform_modulated=True
            ))
            self.predictor.append(Conv2d(
                cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM, num_classes, 1, stride=1, padding=0
            ))
            initialize_module_params(self.predictor[-1])
            self.predictor = nn.Sequential(*self.predictor)
        elif self.predictor_conv_type=="sparse":
            # self.predictor = nn.Identity()
            conv = sparse_conv_with_kaiming_uniform(norm=None, activation=None, use_sep=False, 
                                use_submconv=True, use_deconv=False)
            self.predictor = conv(
                        cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM,
                        num_classes,
                        kernel_size=3,
                        stride=1,
                        dilation=1,
                        indice_key="subm0",
                    )

        # self.amp_enable =  cfg.SOLVER.AMP.ENABLED
        # self.amp_enable = False
        # if self.amp_enable:
        #     self = self.half()

        # weight_init.c2_msra_fill(self.predictor)

        ## debug
        # pdb.set_trace()
        # if self.amp_enable:
        # [p[1].data.dtype for p in self.named_parameters()]
        # for p in self.named_parameters():
        #     if p[1].data.dtype!=torch.float16:
        #         print(p[1].data.dtype)
        #         pdb.set_trace()

    ## Ref: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, features: List[torch.Tensor], iuv_feats: torch.Tensor, rel_coord: torch.Tensor, 
                abs_coord: torch.Tensor, fg_mask: torch.Tensor, ins_mask_list: List[torch.Tensor]):
        # assert fg_mask.min()==0, "the fg_mask is all 1"
        if not self.use_agg_feat:
            for i, _ in enumerate(self.in_features):
                if i == 0:
                    x = self.scale_heads[i](features[i])
                else:
                    x = x + self.scale_heads[i](features[i])
        else:
            x = iuv_feats
        if rel_coord is not None:
            x = torch.cat([x,rel_coord], dim=1)
        if abs_coord is not None:
            x = torch.cat([x,abs_coord], dim=1)
        # pdb.set_trace()
        # if skeleton_feats is not None:
        #     x = torch.cat([x,skeleton_feats], dim=1)

        if rel_coord is not None or abs_coord is not None:
            # if isinstance(self.comb_pe_conv, LambdaLayer):
            #     x = self.comb_pe_conv(x)
            # else:
            x = self.comb_pe_conv(x)

        if self.use_dropout:
            x = self.dropout_layer(x)

        if self.use_san:
            # x = self.san_blk_1(x*fg_mask)
            x = self.san_blk_1(x) 

        ## dense to sparse
        N, C, H, W = x.shape
        coord = compute_grid(H, W, device=x.device, norm=False)
        sparse_coord_batch = []
        sparse_feat_batch = []
        ins_indices_batch = []
        ins_indices_len = []
        ins_cnt = 0
        for n in range(N):
            m = fg_mask[n:n+1]
            x_indices = coord[0][m[0,0]>0]
            y_indices = coord[1][m[0,0]>0]
            if self.use_ins_gn:
                # pdb.set_trace()
                # bg_and_ins = torch.cat([m[0],ins_mask_list[n].float()], dim=0)
                # ins_indices = torch.argmax(bg_and_ins, dim=0)[m[0,0]>0] + ins_cnt
                # try:
                ins_indices = torch.argmax(ins_mask_list[n].float(), dim=0)[m[0,0]>0] + ins_cnt
                # except:
                #     pdb.set_trace()
                ins_indices_batch.append(ins_indices)
                ins_cnt += ins_mask_list[n].shape[0]

                ins_indices_len.append(torch.sum(ins_mask_list[n],dim=[1,2]))

                # import imageio
                # masks = ins_mask_list[n]
                # pdb.set_trace()
                # for ii in range(masks.shape[0]):
                #     imageio.imwrite('tmp/masks_{}.png'.format(ii), masks[ii].detach().cpu().numpy())
                # tmp = torch.argmax(ins_mask_list[n].float(), dim=0).float() + 1
                # imageio.imwrite('tmp/ins_masks.png', (tmp/tmp.max() * m[0,0]).detach().cpu().numpy())
                # imageio.imwrite('tmp/m.png', m[0,0].detach().cpu().numpy())
                # pdb.set_trace()


            b_indices = torch.ones_like(x_indices)*n
            sparse_coord = torch.stack([b_indices,y_indices,x_indices],dim=-1).int()
            sparse_coord_batch.append(sparse_coord)
            sparse_feat = x[n:n+1]
            sparse_feat = sparse_feat[m.expand_as(sparse_feat)>0].reshape([C,-1]).permute([1,0])
            sparse_feat_batch.append(sparse_feat)
        sparse_coord_batch = torch.cat(sparse_coord_batch,dim=0)
        sparse_feat_batch = torch.cat(sparse_feat_batch,dim=0)
        # pdb.set_trace()
        # if self.use_ins_gn:
        #     x = spconv.SparseConvTensor(sparse_feat_batch, sparse_coord_batch, (H,W), ins_cnt)
        # else:
        x = spconv.SparseConvTensor(sparse_feat_batch, sparse_coord_batch, (H,W), N)
        # pdb.set_trace()
        if self.use_ins_gn:
            ins_indices_batch = torch.cat(ins_indices_batch,dim=0)
            ins_indices_len = torch.cat(ins_indices_len,dim=0)
            x = self.densepose_head(x, ins_indices_batch, ins_indices_len)
        else:
            x = self.densepose_head(x)

        # x = x * fg_mask
        # x = self.densepose_head(x, ins_mask_list)
        if self.predictor_conv_type=="sparse":
            x = self.predictor(x).dense()
        else:
            # pdb.set_trace()
            x = x.dense()

            # if self.checkpoint_grad_num>0 and len(self.bbox_tower)>0:
            #     modules = [module for k, module in self.bbox_tower._modules.items()]
            #     bbox_tower = checkpoint.checkpoint_sequential(modules,1,feature)
            # else:
            #     bbox_tower = self.bbox_tower(feature)

            if self.checkpoint_grad_num>0:
                x = checkpoint.checkpoint(self.custom(self.predictor), x)
            else:
                x = self.predictor(x)

        return x


class CoordGlobalIUVSparsePooler2Head(CoordGlobalIUVPooler2Head):
    def _init_densepose_head(self, cfg, input_shape):
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.use_rel_coords = cfg.MODEL.CONDINST.IUVHead.REL_COORDS
        self.use_abs_coords = cfg.MODEL.CONDINST.IUVHead.ABS_COORDS
        self.pos_emb_num_freqs = cfg.MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS
        self.checkpoint_grad_num = cfg.MODEL.CONDINST.CHECKPOINT_GRAD_NUM
        self.use_gt_ins = cfg.MODEL.CONDINST.IUVHead.GT_INSTANCES
        self.inference_global_siuv = cfg.MODEL.CONDINST.INFERENCE_GLOBAL_SIUV
        self.add_skeleton_feat = cfg.MODEL.CONDINST.IUVHead.SKELETON_FEATURES
        # if self.inference_global_siuv:
        #     assert not self.training
        self.use_pos_emb = self.pos_emb_num_freqs>0
        if self.use_pos_emb:
            self.position_embedder, self.position_emb_dim = get_embedder(multires=self.pos_emb_num_freqs, input_dims=2)
        self.pe_dim_all = 0
        if self.use_rel_coords:
            self.pe_dim_all += self.position_emb_dim
        if self.use_abs_coords:
            self.pe_dim_all += self.position_emb_dim
        # if self.add_skeleton_feat:
        #     self.pe_dim_all += 55

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





