# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch, pdb
from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.config import CfgNode
from detectron2.layers import Conv2d

from densepose.utils.comm import compute_locations, compute_grid, aligned_bilinear
from ..utils import initialize_module_params
from .registry import ROI_DENSEPOSE_HEAD_REGISTRY


class InsGNBNIN(nn.Module):
    def __init__(self, num_groups, out_channels, norm):
        super(InsGNBNIN, self).__init__()
        if norm=="InsGN":
            self.norm = nn.GroupNorm(num_groups, out_channels)
        elif norm=="InsBN":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm=="InsIN":
            self.norm = nn.InstanceNorm1d(out_channels)

    # def forward(self, x: spconv.SparseConvTensor, ins_indices_batch: torch.Tensor, ins_ids: Any, ins_indices_len):
    def forward(self, x: torch.Tensor, ins_indices_batch: torch.Tensor, ins_ids_list):
        
        N, C, H, W = x.shape
        # out_batch = []
        x = x.reshape([N,C,H*W])
        ins_indices_batch = ins_indices_batch.reshape([N,H*W])
        # for i in ins_ids:
        #     out = self.norm(x.features[ins_indices_batch==i].reshape([-1,1,C]).permute([1,2,0])) ## HWxBxC -> BxCxHW
        #     out_batch.append(out.permute([2,0,1]).reshape([-1,C]))
        # return spconv.SparseConvTensor(torch.cat(out_batch, dim=0), 
        #                 x.indices, x.spatial_shape, x.batch_size)

        # cnt = 0
        # for ind_len in ins_indices_len.int():
        #     out = self.norm(x.features[cnt:cnt+ind_len].reshape([-1,1,C]).permute([1,2,0])) ## HWxBxC -> BxCxHW
        #     x.features[cnt:cnt+ind_len] = out.permute([2,0,1]).reshape([-1,C]) 
        #     cnt += ind_len
        # assert cnt==N
        # return x
        # pdb.set_trace()
        for n,ins_ids in enumerate(ins_ids_list):
            for i in ins_ids:
                if i==-1:
                    continue
                
                try:
                    fea = x[n,:,(ins_indices_batch[n]==i).nonzero()].permute([1,0,2])
                    x[n,:,(ins_indices_batch[n]==i).nonzero()] = self.norm(fea).permute([1,0,2])
                    # out = self.norm(x.features[(ins_indices_batch==i).nonzero(),].reshape([-1,1,C]).permute([1,2,0])) ## HWxBxC -> BxCxHW
                    # x.features[(ins_indices_batch==i).nonzero(),] = out.permute([2,0,1]) #.reshape(-1)
                except Exception as e:
                    print(e)
                    # print((ins_indices_batch==i).nonzero())
                    # pdb.set_trace()
        return x.reshape([N,C,H, W])


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseV1ConvXGNInsHead(nn.Module):
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
        super(DensePoseV1ConvXGNInsHead, self).__init__()
        # fmt: off
        hidden_dim           = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL
        norm                 = cfg.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS
        self.use_res_input    = cfg.MODEL.CONDINST.IUVHead.RESIDUAL_INPUT
        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels
        cnt = 0
        self.layers = []
        for i in range(self.n_stacked_convs):
            # norm_module = nn.GroupNorm(32, hidden_dim) if norm == "GN" else None
            layer = Conv2d(
                n_channels,
                hidden_dim,
                kernel_size,
                stride=1,
                padding=pad_size,
                bias=not norm,
            )
            weight_init.c2_msra_fill(layer)
            self.add_module("layer{}".format(cnt), layer)
            self.layers.append(layer)
            cnt += 1

            if norm=="GN":
                layer = nn.GroupNorm(32, hidden_dim)
            elif norm=="BN":
                layer = nn.BatchNorm2d(hidden_dim)
            elif norm=="IN":
                layer = nn.InstanceNorm2d(hidden_dim)
            elif norm in ["InsGN","InsBN","InsIN"]:
                layer = InsGNBNIN(32, hidden_dim, norm)
            self.add_module("layer{}".format(cnt), layer)
            self.layers.append(layer)
            cnt += 1

            layer = nn.ReLU(inplace=True)
            # layer_name = self._get_layer_name(cnt)
            self.add_module("layer{}".format(cnt), layer)
            self.layers.append(layer)
            cnt += 1

            # layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            # layer_name = self._get_layer_name(i)
            # self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self.n_out_channels = n_channels
        # initialize_module_params(self)

    def forward(self, features: torch.Tensor, ins_indices_batch):
        """
        Apply DensePose fully convolutional head to the input features

        Args:
            features (tensor): input features
        Result:
            A tensor of DensePose head outputs
        """
        # x = features
        # output = x
        # for i in range(self.n_stacked_convs):
        #     layer_name = self._get_layer_name(i)
        #     x = getattr(self, layer_name)(x)
        #     x = F.relu(x)
        #     output = x
        # return output

        N = ins_indices_batch.shape[0]
        ins_ids_list = [torch.unique(ins_indices_batch[n]) for n in range(N)]

        x = features
        res = None
        for idx, layer in enumerate(self.layers):

            if self.use_res_input:
                if idx in [0,9,18,27,36]:
                    res = x

            if isinstance(layer,InsGNBNIN):
                x = layer(x, ins_indices_batch, ins_ids_list)
            else:
                x = layer(x)

            if self.use_res_input:
                if idx in [5,14,23,32,41]:
                    x = x + res
        output = x
        return output



    def _get_layer_name(self, i: int):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name
