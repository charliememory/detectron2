# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional, Any
import torch, pdb, math
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d
from densepose.layers import sparse_conv_with_kaiming_uniform
import spconv

from ..utils import initialize_module_params
from .registry import ROI_DENSEPOSE_HEAD_REGISTRY
from . v1convxgn_sparsegn import *


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseV1ConvXGNSparseGNESPHead(nn.Module):
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
        super(DensePoseV1ConvXGNSparseGNESPHead, self).__init__()
        # fmt: off
        hidden_dim           = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL
        norm                 = cfg.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS
        self.use_ins_gn = cfg.MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN
        assert self.use_ins_gn
        # fmt: on
        # pad_size = kernel_size // 2
        # pad_size = 0
        N = self.n_stacked_convs
        self.n_branch = n_branch = 4
        n_convs = self.n_stacked_convs/n_branch


        conv = sparse_conv_with_kaiming_uniform(norm=None, activation=None, use_sep=False, 
                            use_submconv=True, use_deconv=False)

        self.start_layers = []
        cnt_layer = 0
        n_channels = input_channels
        layer = conv(
            n_channels,
            hidden_dim//n_branch,
            kernel_size=1,
            stride=1,
            dilation=1,
            indice_key="subm0",
        )
        self.add_module("start_layer{}".format(cnt_layer), layer)
        self.start_layers.append(layer)
        cnt_layer += 1
        #
        if norm in ["GN","BN"]:
            layer = SparseGNBN(32, n_channels, norm)
        elif norm in ["InsGN","InsBN","InsIN"]:
            layer = SparseInsGNBNIN(32, n_channels, norm)
        self.add_module("start_layer{}".format(cnt_layer), layer)
        self.start_layers.append(layer)
        cnt_layer += 1
        #
        layer = SparseReLU(inplace=True)
        self.add_module("start_layer{}".format(cnt_layer), layer)
        self.start_layers.append(layer)
        cnt_layer += 1

        self.branchs = []
        for k in range(n_branch):
            layers = []
            cnt_layer = 0
            # n_channels = input_channels
            for i in range(self.n_stacked_convs):
                layer = conv(
                    hidden_dim//n_branch,
                    hidden_dim//n_branch,
                    kernel_size=3,
                    stride=1,
                    dilation=int(2**k+1),
                    indice_key="subm0",
                ) 
                self.add_module("branch{}_layer{}".format(k,cnt_layer), layer)
                layers.append(layer)
                cnt_layer += 1
                #
                if norm in ["GN","BN"]:
                    layer = SparseGNBN(32, hidden_dim//n_branch, norm)
                elif norm in ["InsGN","InsBN","InsIN"]:
                    layer = SparseInsGNBNIN(32, hidden_dim//n_branch, norm)
                self.add_module("branch{}_layer{}".format(k,cnt_layer), layer)
                layers.append(layer)
                cnt_layer += 1
                #
                layer = SparseReLU(inplace=True)
                self.add_module("branch{}_layer{}".format(k,cnt_layer), layer)
                layers.append(layer)
                cnt_layer += 1
                #
                # n_channels = hidden_dim/n_branch
            self.branchs.append(layers)
            # self.n_out_channels = n_channels
        # initialize_module_params(self)
        self.concat_sparse = spconv.JoinTable()
        self.add_sparse = spconv.AddTable()

    def forward(self, features: spconv.SparseConvTensor, ins_indices_batch: List[torch.Tensor], ins_indices_len):
        """
        Apply DensePose fully convolutional head to the input features

        Args:
            features (tensor): input features
        Result:
            A tensor of DensePose head outputs
        """
        x = features
        # output = x
        "TODO: change ins_indices_batch to start-end slice to save GPU Memory"
        ins_ids = torch.unique(ins_indices_batch)
        # ins_indices_batch = ins_indices_batch[...,None].expand_as(x.features)
        # pdb.set_trace()
        for layer in self.start_layers:
            # print(type(layer))
            if isinstance(layer,SparseInsGNBNIN):
                x = layer(x, ins_indices_batch, ins_ids, ins_indices_len)
            else:
                x = layer(x)

        mid_feat = x

        out_feat = []
        for k in range(self.n_branch):
            feat = mid_feat
            for layer in self.branchs[k]:
                if isinstance(layer,SparseInsGNBNIN):
                    feat = layer(feat, ins_indices_batch, ins_ids, ins_indices_len)
                else:
                    feat = layer(feat)
            out_feat.append(feat)
        for k in range(1,self.n_branch):
            out_feat[k] = self.add_sparse([out_feat[k], out_feat[k-1]])

        output = self.add_sparse([self.concat_sparse(out_feat), features])

        return output

