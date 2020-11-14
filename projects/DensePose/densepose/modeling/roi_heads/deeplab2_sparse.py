# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import fvcore.nn.weight_init as weight_init
import torch, pdb
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d
from densepose.layers import sparse_conv_with_kaiming_uniform
import spconv

from .registry import ROI_DENSEPOSE_HEAD_REGISTRY


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseDeepLab2SparseHead(nn.Module):
    """
    DensePose head using DeepLabV3 model from
    "Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>.
    """

    def __init__(self, cfg: CfgNode, input_channels: int, indice_key="subm0"):
        super(DensePoseDeepLab2SparseHead, self).__init__()
        # fmt: off
        hidden_dim           = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL
        norm                 = cfg.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS
        self.use_nonlocal    = cfg.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NONLOCAL_ON
        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels
        conv = sparse_conv_with_kaiming_uniform(norm, activation=True, use_sep=False, 
                            use_submconv=True, use_deconv=False)
        deconv = sparse_conv_with_kaiming_uniform(norm, activation=True, use_sep=False, 
                            use_submconv=True, use_deconv=True)

        #### different part ####
        self.ASPP = ASPP(input_channels, [6, 12, 56, 112], n_channels, conv, deconv, indice_key)  # 6, 12, 56
        ###########

        self.add_module("ASPP", self.ASPP)

        if self.use_nonlocal:
            raise NotImplementedError
            # self.NLBlock = NONLocalBlock2D(input_channels, bn_layer=True)
            # self.add_module("NLBlock", self.NLBlock)
        # weight_init.c2_msra_fill(self.ASPP)

        for i in range(self.n_stacked_convs):
            layer = conv(
                n_channels,
                hidden_dim,
                kernel_size,
                stride=1,
                dilation=1,
                indice_key=indice_key,
            )
            # norm_module = nn.GroupNorm(32, hidden_dim) if norm == "GN" else None
            # layer = Conv2d(
            #     n_channels,
            #     hidden_dim,
            #     kernel_size,
            #     stride=1,
            #     padding=pad_size,
            #     bias=not norm,
            #     norm=norm_module,
            # )
            # weight_init.c2_msra_fill(layer)
            n_channels = hidden_dim
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
        self.n_out_channels = hidden_dim
        # initialize_module_params(self)

    def forward(self, features):
        x0 = features
        x = self.ASPP(x0)
        # print(x0.shape, x.shape)
        if self.use_nonlocal:
            x = self.NLBlock(x)
        output = x
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            # x = F.relu(x)
            output = x
        # print(x.shape)
        # pdb.set_trace()
        return output

    def _get_layer_name(self, i: int):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name


# Copied from
# https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
# See https://arxiv.org/pdf/1706.05587.pdf for details
# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):

#         conv = sparse_conv_with_kaiming_uniform(norm, activation=True, use_sep=False, 
#                             use_submconv=True, use_deconv=False)
#         modules = [
#             conv(
#                 in_channels, out_channels, 3, dilation=dilation, 
#             ),
#             nn.GroupNorm(32, out_channels),
#             nn.ReLU(),
#         ]
#         super(ASPPConv, self).__init__(*modules)


# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels, out_channels, conv, deconv, indice_key):
#             # nn.AdaptiveAvgPool2d(1),
#         super(ASPPPooling, self).__init__(
#             spconv.SparseMaxPool2d(1),
#             conv(in_channels, out_channels, 1, indice_key=indice_key),
#             deconv(out_channels, out_channels, 1, indice_key=indice_key),
#         )

#     def forward(self, x):
#         x = super(ASPPPooling, self).forward(x)
#         return x


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels, conv, deconv, indice_key):
        super(ASPP, self).__init__()
        layers = []
        layers.append(conv(in_channels, out_channels, kernel_size=3, indice_key=indice_key))

        rate1, rate2, rate3, rate4 = tuple(atrous_rates)
        layers.append(conv(in_channels, out_channels, kernel_size=3, dilation=rate1, indice_key=indice_key))
        layers.append(conv(in_channels, out_channels, kernel_size=3, dilation=rate2, indice_key=indice_key))
        layers.append(conv(in_channels, out_channels, kernel_size=3, dilation=rate3, indice_key=indice_key))
        layers.append(conv(in_channels, out_channels, kernel_size=3, dilation=rate4, indice_key=indice_key))
        # layers.append(ASPPPooling(in_channels, out_channels, conv, deconv, indice_key))

        self.convs = nn.Sequential(*layers)

        self.project = conv(len(layers) * out_channels, out_channels, kernel_size=1, indice_key=indice_key)

        self.concat_sparse = spconv.JoinTable()
        # nn.Sequential(
        #     nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
        #     # nn.BatchNorm2d(out_channels),
        #     nn.ReLU()
        #     # nn.Dropout(0.5)
        # )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        # res = torch.cat(res, dim=1)
        # pdb.set_trace()
        res = self.concat_sparse(res)
        return self.project(res)


