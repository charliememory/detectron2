# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional, Any
import torch, pdb
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d
from densepose.layers import sparse_conv_with_kaiming_uniform
import spconv

from ..utils import initialize_module_params
from .registry import ROI_DENSEPOSE_HEAD_REGISTRY

# class SparseGN(nn.Module):
#     def __init__(self, num_groups, out_channels):
#         super(SparseGN, self).__init__()
#         self.gn = nn.GroupNorm(num_groups, out_channels)

#     def forward(self, x: torch.tensor, batch_size: int, H: int, W: int, 
#         indices: torch.tensor)->spconv.SparseConvTensor:
#         dim = x.shape[1]
#         batch_indices = indices[:,:1].expand_as(x)
#         out_batch = []
#         for i in range(self.batch_size):
#             pdb.set_trace()
#             out = self.gn(x[batch_indices==i].reshape([1,dim,-1]))
#             out_batch.append(out.reshape([-1,dim]))

#         return spconv.SparseConvTensor(torch.cat(out_batch, dim=0), indices, (H,W), batch_size)

class SparseInsGNBNIN(nn.Module):
    def __init__(self, num_groups, out_channels, norm):
        super(SparseInsGNBNIN, self).__init__()
        if norm=="InsGN":
            self.norm = nn.GroupNorm(num_groups, out_channels)
        elif norm=="InsBN":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm=="InsIN":
            self.norm = nn.InstanceNorm1d(out_channels)

    def forward(self, x: spconv.SparseConvTensor, ins_indices_batch: torch.Tensor, ins_ids: Any, ins_indices_len):
        N, C = x.features.shape
        out_batch = []
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

        for i in ins_ids:
            # pdb.set_trace()
            try:
                out = self.norm(x.features[(ins_indices_batch==i).nonzero(),].reshape([-1,1,C]).permute([1,2,0])) ## HWxBxC -> BxCxHW
                x.features[(ins_indices_batch==i).nonzero(),] = out.permute([2,0,1]) #.reshape(-1)
            except Exception as e:
                print(e)
                print((ins_indices_batch==i).nonzero())
                # pdb.set_trace()
        return x


        # for i in ins_ids:
        #     out = self.norm(x.features[ins_indices_batch==i].reshape([-1,1,C]).permute([1,2,0])) ## HWxBxC -> BxCxHW
        #     x.features[ins_indices_batch==i] = out.permute([2,0,1]).reshape(-1)
        # return x

        # N, C = x.features.shape
        # out_batch = self.bn(x.features.reshape([-1,1,C]).permute([1,2,0]))
        # return spconv.SparseConvTensor(out_batch.permute([2,0,1]).reshape([-1,C]), 
        #                 x.indices, x.spatial_shape, x.batch_size)



        # batch_indices = x.indices[:,:1].expand_as(x.features)
        # for i in range(x.batch_size):
        #     # pdb.set_trace()
        #     out = self.gn(x.features[batch_indices==i].reshape([-1,1,C]).permute([1,2,0])) ## HWxBxC -> BxCxHW
        #     out_batch.append(out.permute([2,0,1]).reshape([-1,C]))


class SparseGNBN(nn.Module):
    def __init__(self, num_groups, out_channels, norm):
        super(SparseGNBN, self).__init__()
        if norm=="GN":
            self.norm = nn.GroupNorm(num_groups, out_channels)
        elif norm=="BN":
            self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: spconv.SparseConvTensor):
        # pdb.set_trace()
        N, C = x.features.shape
        batch_indices = x.indices[:,:1].expand_as(x.features)
        out_batch = []
        for i in range(x.batch_size):
            # pdb.set_trace()
            out = self.norm(x.features[batch_indices==i].reshape([-1,1,C]).permute([1,2,0])) ## HWxBxC -> BxCxHW
            x.features[ins_indices_batch==i] = out.permute([2,0,1]).reshape(-1)
        #     out_batch.append(out.permute([2,0,1]).reshape([-1,C]))
        # return spconv.SparseConvTensor(torch.cat(out_batch, dim=0), 
        #                 x.indices, x.spatial_shape, x.batch_size)
        return x

class SparseReLU(nn.Module):
    def __init__(self, inplace=True):
        super(SparseReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x: spconv.SparseConvTensor):
        return spconv.SparseConvTensor(F.relu(x.features, inplace=self.inplace), x.indices, x.spatial_shape, x.batch_size)


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseV1ConvXGNSparseGNHead(nn.Module):
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
        super(DensePoseV1ConvXGNSparseGNHead, self).__init__()
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
        n_channels = input_channels
        conv = sparse_conv_with_kaiming_uniform(norm=None, activation=None, use_sep=False, 
                            use_submconv=True, use_deconv=False)
        cnt = 0
        self.layers = []
        for i in range(self.n_stacked_convs):
            layer = conv(
                n_channels,
                hidden_dim,
                kernel_size,
                stride=1,
                dilation=1,
                indice_key="subm0",
            )
            # layer_name = self._get_layer_name(cnt)
            self.add_module("layer{}".format(cnt), layer)
            self.layers.append(layer)
            cnt += 1

            # if self.use_ins_gn:
            #     layer = SparseInsGNBNIN(32, n_channels)
            if norm in ["GN","BN"]:
                layer = SparseGNBN(32, n_channels, norm)
            elif norm in ["InsGN","InsBN","InsIN"]:
                layer = SparseInsGNBNIN(32, n_channels, norm)
            # layer_name = self._get_layer_name(cnt)
            self.add_module("layer{}".format(cnt), layer)
            self.layers.append(layer)
            cnt += 1

            layer = SparseReLU(inplace=True)
            # layer_name = self._get_layer_name(cnt)
            self.add_module("layer{}".format(cnt), layer)
            self.layers.append(layer)
            cnt += 1

            n_channels = hidden_dim
        self.n_out_channels = n_channels
        # initialize_module_params(self)

    def forward(self, features: spconv.SparseConvTensor, ins_indices_batch: List[torch.Tensor], ins_indices_len):
        """
        Apply DensePose fully convolutional head to the input features

        Args:
            features (tensor): input features
        Result:
            A tensor of DensePose head outputs
        """
        # pdb.set_trace()
        # features.batch_size
        # x = spconv.SparseConvTensor(sparse_feat_batch, sparse_coord_batch, (H,W), N)

        # batch_indices = features.indices[:,0:1].expand_as(features.features)
        x = features
        # output = x
        "TODO: change ins_indices_batch to start-end slice to save GPU Memory"
        ins_ids = torch.unique(ins_indices_batch)
        # ins_indices_batch = ins_indices_batch[...,None].expand_as(x.features)
        # pdb.set_trace()
        for layer in self.layers:
            # print(type(layer))
            if isinstance(layer,SparseInsGNBNIN):
                x = layer(x, ins_indices_batch, ins_ids, ins_indices_len)
            else:
                x = layer(x)

        output = x
        return output

    # def _get_layer_name(self, i: int):
    #     layer_name = "body_conv_fcn{}".format(i + 1)
    #     layer_name = "layer".format(i)
    #     return layer_name
