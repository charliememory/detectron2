# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d
from densepose.layers import sparse_conv_with_kaiming_uniform
import spconv

from ..utils import initialize_module_params
from .registry import ROI_DENSEPOSE_HEAD_REGISTRY


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseV1ConvXGNSparseHead(nn.Module):
    """
    Fully convolutional DensePose head.
    """

    def __init__(self, cfg: CfgNode, input_channels: int):
        """
        Initialize DensePose fully convolutional head

        Args:
            cfg (CfgNode): configuration options
            input_channels (int): number of input channels
        """
        super(DensePoseV1ConvXGNSparseHead, self).__init__()
        # fmt: off
        hidden_dim           = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL
        norm                 = cfg.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS
        # fmt: on
        # pad_size = kernel_size // 2
        # pad_size = 0
        n_channels = input_channels
        conv = sparse_conv_with_kaiming_uniform(norm, activation=True, use_sep=False, 
                            use_submconv=True, use_deconv=False)
        for i in range(self.n_stacked_convs):
            layer = conv(
                n_channels,
                hidden_dim,
                kernel_size,
                stride=1,
                dilation=1,
                indice_key="subm0",
            )
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self.n_out_channels = n_channels
        # initialize_module_params(self)

    def forward(self, features: spconv.SparseConvTensor):
        """
        Apply DensePose fully convolutional head to the input features

        Args:
            features (tensor): input features
        Result:
            A tensor of DensePose head outputs
        """
        x = features
        output = x
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            # x = F.relu(x)
            output = x
        return output

    def _get_layer_name(self, i: int):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name
