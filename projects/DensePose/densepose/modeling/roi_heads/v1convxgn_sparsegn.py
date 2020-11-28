# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional, Any
import torch, pdb
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

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

    # def forward(self, x: spconv.SparseConvTensor, ins_indices_batch: torch.Tensor, ins_ids: Any, ins_indices_len):
    def forward(self, x: spconv.SparseConvTensor, ins_indices_batch: torch.Tensor, ins_ids: Any):
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
                # print((ins_indices_batch==i).nonzero())
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


class Conv1dWS(nn.Conv1d):
    def forward(self, input):
        ## Weight Standardization
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)

        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SparseECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3, use_weight_std=False):
        super(SparseECA, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if use_weight_std:
            self.conv = Conv1dWS(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        else:
            self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        "TODO: change sigmoid to hard-swish?"
        self.activation = nn.Sigmoid()

    def forward(self, x: spconv.SparseConvTensor):
        N, C = x.features.shape
        batch_indices = x.indices[:,:1].expand_as(x.features)
        out_batch = []
        for i in range(x.batch_size):
            # pdb.set_trace()
            out = x.features[batch_indices==i].reshape([-1,1,C]).mean(dim=0).unsqueeze(dim=1) ## HWxBxC -> Bx1xC
            out_batch.append(out)

        out_batch = self.activation(self.conv(torch.cat(out_batch, dim=0))).squeeze(dim=1) ## Bx1xC -> BxC
        for i in range(x.batch_size):
            # pdb.set_trace()
            x.features[batch_indices==i] = (x.features[batch_indices==i].reshape([-1,C]) * out_batch[i:i+1]).reshape(-1)
        #     out_batch.append(out.permute([2,0,1]).reshape([-1,C]))
        # return spconv.SparseConvTensor(torch.cat(out_batch, dim=0), 
        #                 x.indices, x.spatial_shape, x.batch_size)
        return x

"TODO"
class SparseInsECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3, use_weight_std=False):
        super(SparseInsECA, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if use_weight_std:
            self.conv = Conv1dWS(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        else:
            self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        "TODO: change sigmoid to hard-swish?"
        self.activation = nn.Sigmoid()

    def forward(self, x: spconv.SparseConvTensor, ins_indices_batch: torch.Tensor, ins_ids: Any):
        N, C = x.features.shape
        batch_indices = x.indices[:,:1].expand_as(x.features)
        out_batch = []
        # for i in range(x.batch_size):
        #     out = x.features[batch_indices==i].reshape([-1,1,C]).mean(dim=0).unsqueeze(dim=1) ## HWxBxC -> Bx1xC
        #     out_batch.append(out)

        # out_batch = self.activation(self.conv(torch.cat(out_batch, dim=0))).squeeze(dim=1) ## Bx1xC -> BxC
        # for i in range(x.batch_size):
        #     x.features[batch_indices==i] = (x.features[batch_indices==i].reshape([-1,C]) * out_batch[i:i+1]).reshape(-1)
        # return x


        for i in ins_ids:
            out = x.features[(ins_indices_batch==i).nonzero(),].reshape([-1,1,C]).mean(dim=0).unsqueeze(dim=1) ## HWxBxC -> Bx1xC
            out_batch.append(out)
        # pdb.set_trace()
        out_batch = self.activation(self.conv(torch.cat(out_batch, dim=0)))#.squeeze(dim=1) ## Bx1xC -> BxC
        for i in ins_ids:
            try:
                x.features[(ins_indices_batch==i).nonzero(),] = x.features[(ins_indices_batch==i).nonzero(),].reshape([-1,1,C]) * out_batch[i:i+1]
            except Exception as e:
                pass
                # print(e)
        return x

# for i in ins_ids:
#     out = self.norm(x.features[(ins_indices_batch==i).nonzero(),].reshape([-1,1,C]).permute([1,2,0])) ## HWxBxC -> BxCxHW
#     x.features[(ins_indices_batch==i).nonzero(),] = out.permute([2,0,1]) #.reshape(-1)
# return x


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
        self.use_ins_gn      = cfg.MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN
        self.use_weight_std  = cfg.MODEL.CONDINST.IUVHead.WEIGHT_STANDARDIZATION
        # self.use_eca = cfg.MODEL.CONDINST.IUVHead.Efficient_Channel_Attention
        self.use_eca = False
        self.use_ins_eca = cfg.MODEL.CONDINST.IUVHead.INSTANCE_EFFICIENT_CHANNEL_ATTENTION
        self.use_res_input    = cfg.MODEL.CONDINST.IUVHead.RESIDUAL_INPUT
        self.use_res_after_relu    = cfg.MODEL.CONDINST.IUVHead.RESIDUAL_SKIP_AFTER_RELU
        self.use_res_later   = cfg.MODEL.CONDINST.IUVHead.RESIDUAL_SKIP_LATER
        self.use_dilated_conv = cfg.MODEL.CONDINST.IUVHead.DILATION_CONV
        self.add_sparse = spconv.tables.AddTable()
        self.checkpoint_grad_num = cfg.MODEL.CONDINST.CHECKPOINT_GRAD_NUM
        assert self.use_ins_gn
        # fmt: on
        # pad_size = kernel_size // 2
        # pad_size = 0
        n_channels = input_channels
        from torch.cuda.amp import autocast
        with autocast():
            conv = sparse_conv_with_kaiming_uniform(norm=None, activation=None, use_sep=False, 
                                use_submconv=True, use_deconv=False, use_weight_std=self.use_weight_std)
            cnt = 0
            self.layers = []
            for i in range(self.n_stacked_convs):
                if self.use_dilated_conv:
                    if i==3:
                        r = 2
                    elif i==4:
                        r = 4
                    elif i==5:
                        r = 8
                    else:
                        r = 1
                    layer = conv(
                        n_channels,
                        hidden_dim,
                        kernel_size,
                        stride=1,
                        dilation=r,
                        indice_key="subm0",
                    )
                else:
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
                # pdb.set_trace()
                # [p[1].data.dtype for p in layer.half().named_parameters()]
                cnt += 1

                if self.use_eca:
                    layer = SparseECA(channel=hidden_dim)
                    self.add_module("layer{}".format(cnt), layer)
                    self.layers.append(layer)
                    cnt += 1

                if self.use_ins_eca=="AfterConv":
                    layer = SparseInsECA(channel=hidden_dim)
                    self.add_module("layer{}".format(cnt), layer)
                    self.layers.append(layer)
                    cnt += 1

                # if self.use_ins_gn:
                #     layer = SparseInsGNBNIN(32, n_channels)
                if norm in ["GN","BN"]:
                    layer = SparseGNBN(32, hidden_dim, norm)
                elif norm in ["InsGN","InsBN","InsIN"]:
                    layer = SparseInsGNBNIN(32, hidden_dim, norm)
                # layer_name = self._get_layer_name(cnt)
                self.add_module("layer{}".format(cnt), layer)
                self.layers.append(layer)
                cnt += 1

                if self.use_ins_eca=="AfterNorm":
                    layer = SparseInsECA(channel=hidden_dim)
                    self.add_module("layer{}".format(cnt), layer)
                    self.layers.append(layer)
                    cnt += 1


                if self.use_ins_eca=="AfterRelu":
                    layer = SparseReLU(inplace=False)
                    # layer_name = self._get_layer_name(cnt)
                    self.add_module("layer{}".format(cnt), layer)
                    self.layers.append(layer)
                    cnt += 1

                    layer = SparseInsECA(channel=hidden_dim)
                    self.add_module("layer{}".format(cnt), layer)
                    self.layers.append(layer)
                    cnt += 1
                else:
                    layer = SparseReLU(inplace=True)
                    # layer_name = self._get_layer_name(cnt)
                    self.add_module("layer{}".format(cnt), layer)
                    self.layers.append(layer)
                    cnt += 1

                n_channels = hidden_dim
            self.n_out_channels = n_channels
        # initialize_module_params(self)

        # if self.amp_enable:
        #     self = self.half()
            # [p[1].data.dtype for p in self.named_parameters()]
            # for layer in self.layers:
            # for p in self.named_parameters():
            #     if p[1].data.dtype!=torch.float16:
            #         print(p[1].data.dtype)
            #         pdb.set_trace()

    # ## Ref: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
    # def custom(self, module):
    #     def custom_forward(*inputs):
    #         inputs = module(inputs[0])
    #         return inputs
    #     return custom_forward

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

        # pdb.set_trace()
        # if self.amp_enable:
        #     x.features = x.features.half()

        res = None
        for idx, layer in enumerate(self.layers):
            # if self.use_res_later:
            #     if idx==3:
            #         res = x
            #     elif idx==8:
            #         x = self.add_sparse([x,res])
            #     if idx==12:
            #         res = x
            #     elif idx==17:
            #         x = self.add_sparse([x,res])
            # if self.use_res_after_relu:
            #     if idx==3:
            #         res = x
            #     elif idx==9:
            #         x = self.add_sparse([x,res])
            #     if idx==12:
            #         res = x
            #     elif idx==18:
            #         x = self.add_sparse([x,res])

            if self.use_res_input:
                if self.use_ins_eca!="none":
                    # pdb.set_trace()
                    if self.use_ins_eca=="AfterRelu":
                        if idx in [0,9+3-1,18+6-1,27+9-1,36+12-1]:
                            res = x
                    else:
                        if idx in [0,9+3,18+6,27+9,36+12]:
                            res = x
                else:
                    if idx in [0,9,18,27,36]:
                        res = x

            # print(type(layer))
            # if isinstance(layer, spconv.SubMConv2d):
            #     if self.checkpoint_grad_num>0:
            #         x = checkpoint.checkpoint(self.custom(layer), x)
            #     else:
            #         x = layer(x)
            if isinstance(layer,SparseInsGNBNIN) or isinstance(layer,SparseInsECA):
                # x = layer(x, ins_indices_batch, ins_ids, ins_indices_len)
                x = layer(x, ins_indices_batch, ins_ids)
            else:
                # pdb.set_trace()
                x = layer(x)

            if self.use_res_input:
                if self.use_ins_eca!="none":
                    if self.use_ins_eca=="AfterRelu":
                        if idx in [5+3-1,14+6-1,23+9-1,32+12-1,41+15-1]:
                            x = self.add_sparse([x,res])
                    else:
                        if idx in [5+3,14+6,23+9,32+12,41+15]:
                            x = self.add_sparse([x,res])
                else:
                    if idx in [5,14,23,32,41]:
                        x = self.add_sparse([x,res])



        x.features = x.features #.float()
        output = x
        return output

    # def _get_layer_name(self, i: int):
    #     layer_name = "body_conv_fcn{}".format(i + 1)
    #     layer_name = "layer".format(i)
    #     return layer_name
