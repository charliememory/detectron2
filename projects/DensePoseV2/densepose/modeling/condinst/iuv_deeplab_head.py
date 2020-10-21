from typing import Dict
import math

import torch
from torch import nn

from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import ShapeSpec

# from adet.layers import conv_with_kaiming_uniform
# from adet.utils.comm import aligned_bilinear
from densepose.layers import conv_with_kaiming_uniform
from densepose.utils.comm import aligned_bilinear
from ..roi_heads import DensePoseDeepLabHead
# from .. import (
#     build_densepose_data_filter,
#     build_densepose_head,
#     build_densepose_losses,
#     build_densepose_predictor,
#     densepose_inference,
# )
import pdb

INF = 100000000


def build_iuv_deeplab_head(cfg):
    return GlobalIUVDeepLabHeadAfterMaskBranch(cfg)


class GlobalIUVDeepLabHeadAfterMaskBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_outputs = cfg.MODEL.CONDINST.IUVHead.OUT_CHANNELS
        norm = cfg.MODEL.CONDINST.IUVHead.NORM
        num_convs = cfg.MODEL.CONDINST.IUVHead.NUM_CONVS
        channels = cfg.MODEL.CONDINST.IUVHead.CHANNELS
        self.iuv_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE

        # conv_block = conv_with_kaiming_uniform(norm, activation=True)

        tower = []

        # cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM = channels
        tower.append(
            DensePoseDeepLabHead(cfg, input_channels=channels, hidden_dim=channels)
        )
        # for i in range(num_convs):
        #     tower.append(conv_block(
        #         channels, channels, 3, 1
        #     ))
        tower.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        self.add_module('tower', nn.Sequential(*tower))

        # self.densepose_losses = build_densepose_losses(cfg)

    def forward(self, fea, feat_stride, gt_instances=None):
        iuv_logit = self.tower(fea)

        assert feat_stride >= self.iuv_out_stride
        assert feat_stride % self.iuv_out_stride == 0
        iuv_logit = aligned_bilinear(iuv_logit, int(feat_stride / self.iuv_out_stride))

        return iuv_logit
