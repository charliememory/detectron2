# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional, Any
import torch, pdb, math
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





"TODO"
import spconv
class SubMConv2dInsDilate(spconv.conv.SubMConv2d):
    # def __init__(self, num_groups, out_channels):
    #     super(SubMConv2dInsDilate, self).__init__()

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation_max=1,
                 groups=1,
                 bias=True,
                 algo=spconv.ops.ConvAlgo.Native):

        super(SubMConv2dInsDilate, self).__init__(
                                                 in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 dilation=1,
                                                 groups=groups,
                                                 bias=bias,
                                                 indice_key=None,
                                                 use_hash=False,
                                                 algo=algo)
        self.dilation_max = dilation_max

    def forward(self, input, ins_indices_batch, ins_ids):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        out_spatial_shape = spatial_shape
        # input.update_grid(out_spatial_shape)
        # t = time.time()

        features_list = []
        outids_list =[]
        for i in ins_ids:
            indice_key = "ins{}_dilatemax{}".format(i, self.dilation_max)
            # datas = input.find_indice_pair(indice_key)
            # if self.indice_key is not None and datas is not None:
            #     outids, _, indice_pairs, indice_pair_num, _ = datas
            # else:
            if indice_key not in input.indice_dict:
                "Dilation depends on instance size"
                ins_indices = input.indices[(ins_indices_batch==i).nonzero()].squeeze(1)
                h_ratio = (ins_indices[:,1].max() - ins_indices[:,1].min()).float()/input.spatial_shape[0]
                w_ratio = (ins_indices[:,2].max() - ins_indices[:,2].min()).float()/input.spatial_shape[1]
                d = max(1, math.ceil(max(h_ratio, w_ratio)*self.dilation_max))
                
                outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
                    ins_indices,
                    input.batch_size,
                    input.spatial_shape,
                    ksize=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=d,
                    out_padding=self.output_padding,
                    subm=self.subm,
                    transpose=self.transposed,
                    grid=input.grid,
                    use_hash=self.use_hash
                    )
                input.indice_dict[indice_key] = (outids, input.indices,
                                                      indice_pairs,
                                                      indice_pair_num,
                                                      input.spatial_shape)
            else:
                datas = input.find_indice_pair(indice_key)
                outids, _, indice_pairs, indice_pair_num, _ = datas

            feat = features[(ins_indices_batch==i).nonzero()].squeeze(1)
            feat = spconv.functional.indice_subm_conv(feat, self.weight,
                                                indice_pairs.to(device),
                                                indice_pair_num,
                                                outids.shape[0], self.algo)
            features_list.append(feat)
            outids_list.append(outids)

        out_features = torch.cat(features_list, dim=0)
        outids = torch.cat(outids_list, dim=0)
        if self.bias is not None:
            out_features += self.bias
        # datas = input.find_indice_pair(indice_key)
        # if self.indice_key is not None and datas is not None:
        #     outids, _, indice_pairs, indice_pair_num, _ = datas
        # else:
        #     outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
        #         indices,
        #         batch_size,
        #         spatial_shape,
        #         self.kernel_size,
        #         self.stride,
        #         self.padding,
        #         self.dilation,
        #         self.output_padding,
        #         self.subm,
        #         self.transposed,
        #         grid=input.grid,
        #         use_hash=self.use_hash)
        #     input.indice_dict[self.indice_key] = (outids, indices,
        #                                           indice_pairs,
        #                                           indice_pair_num,
        #                                           spatial_shape)

        # if self.subm:
        #     out_features = Fsp.indice_subm_conv(features, self.weight,
        #                                         indice_pairs.to(device),
        #                                         indice_pair_num,
        #                                         outids.shape[0], self.algo)


        # if self.bias is not None:
        #     out_features += self.bias
        out_tensor = spconv.SparseConvTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


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
        for i in ins_ids:
            # pdb.set_trace()
            try:
                out = self.norm(x.features[(ins_indices_batch==i).nonzero(),].reshape([-1,1,C]).permute([1,2,0])) ## HWxBxC -> BxCxHW
                x.features[(ins_indices_batch==i).nonzero(),] = out.permute([2,0,1]) #.reshape(-1)
            except Exception as e:
                # print(e)
                pass
                # print((ins_indices_batch==i).nonzero())
                # pdb.set_trace()
        return x


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
        x.features = F.relu(x.features, inplace=self.inplace)
        return x
        # return spconv.SparseConvTensor(F.relu(x.features, inplace=self.inplace), x.indices, x.spatial_shape, x.batch_size)


# class ASPP(nn.Module):
#     def __init__(self, in_channels, atrous_rates, out_channels):
#         super(ASPP, self).__init__()
#         modules = []
#         modules.append(
#             nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                 nn.GroupNorm(32, out_channels),
#                 nn.ReLU(),
#             )
#         )

#         rate1, rate2, rate3 = tuple(atrous_rates)
#         modules.append(ASPPConv(in_channels, out_channels, rate1))
#         modules.append(ASPPConv(in_channels, out_channels, rate2))
#         modules.append(ASPPConv(in_channels, out_channels, rate3))
#         modules.append(ASPPPooling(in_channels, out_channels))

#         self.convs = nn.ModuleList(modules)

#         self.project = nn.Sequential(
#             nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
#             # nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#             # nn.Dropout(0.5)
#         )

#     def forward(self, x):
#         res = []
#         for conv in self.convs:
#             res.append(conv(x))
#         res = torch.cat(res, dim=1)
#         return self.project(res)

class SubMConv2dASPP(nn.Module):
    # def __init__(self, num_groups, out_channels):
    #     super(SubMConv2dInsDilate, self).__init__()

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 norm="InsIN",
                 stride=1,
                 padding=0,
                 atrous_rates=[1,3,5],
                 groups=1,
                 bias=True,
                 algo=spconv.ops.ConvAlgo.Native):
        super(SubMConv2dASPP, self).__init__()

        self.fuse_conv = spconv.conv.SubMConv2d(
                             in_channels+out_channels*3,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=1,
                             bias=bias,
                             algo=algo,
                             indice_key="subm0",)
        self.aspp_conv1 = spconv.conv.SubMConv2d(
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=atrous_rates[0],
                             groups=groups,
                             bias=bias,
                             algo=algo,
                             indice_key="subm0_aspp1",)
        self.aspp_norm1 = SparseInsGNBNIN(32, out_channels, norm)
        self.aspp_relu1 = SparseReLU()
        self.aspp_conv2 = spconv.conv.SubMConv2d(
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=atrous_rates[1],
                             groups=groups,
                             bias=bias,
                             algo=algo,
                             indice_key="subm0_aspp2")
        self.aspp_norm2 = SparseInsGNBNIN(32, out_channels, norm)
        self.aspp_relu2 = SparseReLU()
        self.aspp_conv3 = spconv.conv.SubMConv2d(
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=atrous_rates[2],
                             groups=groups,
                             bias=bias,
                             algo=algo,
                             indice_key="subm0_aspp3")
        self.aspp_norm3 = SparseInsGNBNIN(32, out_channels, norm)
        self.aspp_relu3 = SparseReLU()
        # self.aspp_conv1 = SubMConv2dInsDilate(
        #                      in_channels,
        #                      out_channels,
        #                      kernel_size,
        #                      stride=stride,
        #                      padding=padding,
        #                      dilation_max=dilation_max,
        #                      groups=groups,
        #                      bias=bias,
        #                      algo=algo)
        # self.aspp_conv2 = SubMConv2dInsDilate(
        #                      in_channels,
        #                      out_channels,
        #                      kernel_size,
        #                      stride=stride,
        #                      padding=padding,
        #                      dilation_max=dilation_max*2,
        #                      groups=groups,
        #                      bias=bias,
        #                      algo=algo)
        # self.aspp_conv3 = SubMConv2dInsDilate(
        #                      in_channels,
        #                      out_channels,
        #                      kernel_size,
        #                      stride=stride,
        #                      padding=padding,
        #                      dilation_max=dilation_max*4,
        #                      groups=groups,
        #                      bias=bias,
        #                      algo=algo)
        self.join_sparse = spconv.tables.JoinTable()

    def forward(self, input, ins_indices_batch, ins_ids):
        assert isinstance(input, spconv.SparseConvTensor)

        aspp1 = self.aspp_relu1(self.aspp_norm1(self.aspp_conv1(input), ins_indices_batch, ins_ids))
        aspp2 = self.aspp_relu2(self.aspp_norm2(self.aspp_conv2(input), ins_indices_batch, ins_ids))
        aspp3 = self.aspp_relu3(self.aspp_norm3(self.aspp_conv3(input), ins_indices_batch, ins_ids))
        # aspp1 = self.aspp_conv1(input, ins_indices_batch, ins_ids)
        # aspp2 = self.aspp_conv2(input, ins_indices_batch, ins_ids)
        # aspp3 = self.aspp_conv3(input, ins_indices_batch, ins_ids)
        return self.fuse_conv(self.join_sparse([input,aspp1,aspp2,aspp3]))


class SubMConv2dInsDilateASPP(nn.Module):
    # def __init__(self, num_groups, out_channels):
    #     super(SubMConv2dInsDilate, self).__init__()

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 norm="InsIN",
                 stride=1,
                 padding=0,
                 atrous_rates=[1,3,5],
                 groups=1,
                 bias=True,
                 algo=spconv.ops.ConvAlgo.Native):
        super(SubMConv2dInsDilateASPP, self).__init__()

        self.fuse_conv = spconv.conv.SubMConv2d(
                             in_channels+out_channels*3,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=1,
                             bias=bias,
                             algo=algo,
                             indice_key="subm0",)
        self.aspp_conv1 = SubMConv2dInsDilate(
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation_max=atrous_rates[0],
                             groups=groups,
                             bias=bias,
                             algo=algo)
        self.aspp_norm1 = SparseInsGNBNIN(32, out_channels, norm)
        self.aspp_relu1 = SparseReLU()
        self.aspp_conv2 = SubMConv2dInsDilate(
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation_max=atrous_rates[1],
                             groups=groups,
                             bias=bias,
                             algo=algo)
        self.aspp_norm2 = SparseInsGNBNIN(32, out_channels, norm)
        self.aspp_relu2 = SparseReLU()
        self.aspp_conv3 = SubMConv2dInsDilate(
                             in_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation_max=atrous_rates[2],
                             groups=groups,
                             bias=bias,
                             algo=algo)
        self.aspp_norm3 = SparseInsGNBNIN(32, out_channels, norm)
        self.aspp_relu3 = SparseReLU()
        self.join_sparse = spconv.tables.JoinTable()

    def forward(self, input, ins_indices_batch, ins_ids):
        assert isinstance(input, spconv.SparseConvTensor)

        aspp1 = self.aspp_relu1(self.aspp_norm1(self.aspp_conv1(input, ins_indices_batch, ins_ids), ins_indices_batch, ins_ids))
        aspp2 = self.aspp_relu2(self.aspp_norm2(self.aspp_conv2(input, ins_indices_batch, ins_ids), ins_indices_batch, ins_ids))
        aspp3 = self.aspp_relu3(self.aspp_norm3(self.aspp_conv3(input, ins_indices_batch, ins_ids), ins_indices_batch, ins_ids))
        # aspp1 = self.aspp_conv1(input, ins_indices_batch, ins_ids)
        # aspp2 = self.aspp_conv2(input, ins_indices_batch, ins_ids)
        # aspp3 = self.aspp_conv3(input, ins_indices_batch, ins_ids)
        return self.fuse_conv(self.join_sparse([input,aspp1,aspp2,aspp3]))


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
        self.use_ins_conv    = cfg.MODEL.CONDINST.IUVHead.INSTANCE_AWARE_CONV
        self.use_weight_std  = cfg.MODEL.CONDINST.IUVHead.WEIGHT_STANDARDIZATION
        self.use_submconv  = cfg.MODEL.CONDINST.IUVHead.SUBM_CONV
        # self.use_eca = cfg.MODEL.CONDINST.IUVHead.Efficient_Channel_Attention
        self.use_eca = False
        self.use_ins_eca = cfg.MODEL.CONDINST.IUVHead.INSTANCE_EFFICIENT_CHANNEL_ATTENTION
        self.use_res_input    = cfg.MODEL.CONDINST.IUVHead.RESIDUAL_INPUT
        self.use_res_after_relu    = cfg.MODEL.CONDINST.IUVHead.RESIDUAL_SKIP_AFTER_RELU
        self.use_res_later   = cfg.MODEL.CONDINST.IUVHead.RESIDUAL_SKIP_LATER
        self.use_res_skip_conv    = cfg.MODEL.CONDINST.IUVHead.RESIDUAL_SKIP_CONV
        self.dilated_conv_type = cfg.MODEL.CONDINST.IUVHead.DILATION_CONV
        self.dilated_conv_r_max = cfg.MODEL.CONDINST.IUVHead.DILATION_CONV_R_MAX
        self.add_sparse = spconv.tables.AddTable()
        self.checkpoint_grad_num = cfg.MODEL.CONDINST.CHECKPOINT_GRAD_NUM
        # self.replace_minus_one = cfg.MODEL.CONDINST.IUVHead.REPLACE_MINUS_ONE
        assert self.use_ins_gn
        # fmt: on
        # pad_size = kernel_size // 2
        # pad_size = 0
        n_channels = input_channels
        from torch.cuda.amp import autocast
        with autocast():
            conv = sparse_conv_with_kaiming_uniform(norm=None, activation=None, use_sep=False, 
                                use_submconv=self.use_submconv, use_deconv=False, use_weight_std=self.use_weight_std)
            cnt = 0
            self.layers = []
            self.pad = 0
            if self.use_res_skip_conv:
                self.res_skip_convs = []
                for ii in range(3):
                    layer_list = []
                    layer = conv(
                        n_channels,
                        hidden_dim,
                        kernel_size,
                        stride=1,
                        padding=self.pad,
                        dilation=1,
                        indice_key="subm0",
                    )
                    layer_list.append(layer)
                    self.add_module("res_skip_conv{}".format(ii), layer)
                    if norm in ["GN","BN"]:
                        layer = SparseGNBN(32, hidden_dim, norm)
                    elif norm in ["InsGN","InsBN","InsIN"]:
                        layer = SparseInsGNBNIN(32, hidden_dim, norm)
                    layer_list.append(layer)
                    self.add_module("res_skip_norm{}".format(ii), layer)
                    layer = SparseReLU(inplace=True)
                    layer_list.append(layer)
                    self.add_module("res_skip_relu{}".format(ii), layer)
                    self.res_skip_convs.append(layer_list)

            for i in range(self.n_stacked_convs):
                # pdb.set_trace()
                if self.dilated_conv_type=="none":
                    layer = conv(
                        n_channels,
                        hidden_dim,
                        kernel_size,
                        stride=1,
                        padding=self.pad,
                        dilation=1,
                        indice_key="subm0",
                    )
                # if self.dilated_conv_type=="one_layer_ori":
                #     if cnt<12:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                #     elif cnt in [12]:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=self.dilated_conv_r_max,
                #             indice_key="subm0_insconv_adapt",
                #         )
                #     else:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0_insconv",
                #         )
                # elif self.dilated_conv_type=="progressive_ori":
                #     if cnt<12:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                #     elif cnt==12:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0_insconv_adapt",
                #         )
                #     elif cnt==15:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm1_insconv_adapt",
                #         )
                #     elif cnt==18:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm2_insconv_adapt",
                #         )
                #     else:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0_insconv",
                #         )
                # elif self.dilated_conv_type=="one_layer":
                #     if cnt<12:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                #     elif cnt in [12]:
                #         layer = SubMConv2dInsDilate(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              stride=1,
                #              padding=self.pad,
                #              dilation_max=self.dilated_conv_r_max
                #         )
                #         # layer = conv(
                #         #     n_channels,
                #         #     hidden_dim,
                #         #     kernel_size,
                #         #     stride=1,
                #         #     padding=self.pad,
                #         #     dilation=self.dilated_conv_r_max,
                #         #     indice_key="subm0_insconv_adapt",
                #         # )
                #     elif self.use_ins_conv:
                #         layer = SubMConv2dInsDilate(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              stride=1,
                #              padding=self.pad,
                #              dilation_max=1
                #         )
                #     else:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                #         # layer = conv(
                #         #     n_channels,
                #         #     hidden_dim,
                #         #     kernel_size,
                #         #     stride=1,
                #         #     padding=self.pad,
                #         #     dilation=1,
                #         #     indice_key="subm0_insconv",
                #         # )
                # elif self.dilated_conv_type=="aspp":
                #     if cnt<6:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                #     elif cnt in [6]:
                #         d = self.dilated_conv_r_max
                #         layer = SubMConv2dASPP(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              norm=norm,
                #              stride=1,
                #              padding=self.pad,
                #              atrous_rates=[d,d+2,d+4]
                #         )
                #     else:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                # elif self.dilated_conv_type=="aspp_large":
                #     if cnt<6:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                #     elif cnt in [6]:
                #         d = self.dilated_conv_r_max
                #         layer = SubMConv2dASPP(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              norm=norm,
                #              stride=1,
                #              padding=self.pad,
                #              atrous_rates=[d,d*2,d*4]
                #         )
                #     else:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                # elif self.dilated_conv_type=="aspp_large":
                #     if cnt<6:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                #     elif cnt in [6]:
                #         d = self.dilated_conv_r_max
                #         layer = SubMConv2dASPP(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              norm=norm,
                #              stride=1,
                #              padding=self.pad,
                #              atrous_rates=[d,d*2,d*4]
                #         )
                #     else:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                # elif self.dilated_conv_type=="aspp_larger":
                #     if cnt<6:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                #     elif cnt in [6]:
                #         d = self.dilated_conv_r_max
                #         layer = SubMConv2dASPP(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              norm=norm,
                #              stride=1,
                #              padding=self.pad,
                #              atrous_rates=[d,d*2,d*8]
                #         )
                #     else:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                # elif self.dilated_conv_type=="ins_dilate_aspp":
                #     if cnt<6:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                #     elif cnt in [6]:
                #         d = self.dilated_conv_r_max
                #         layer = SubMConv2dInsDilateASPP(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              norm=norm,
                #              stride=1,
                #              padding=self.pad,
                #              atrous_rates=[d,d+2,d+4]
                #         )
                #     else:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                # elif self.dilated_conv_type=="progressive":
                #     if cnt==0:
                #         layer = SubMConv2dInsDilate(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              stride=1,
                #              padding=self.pad,
                #              dilation_max=self.dilated_conv_r_max
                #         )
                #         # layer = conv(
                #         #     n_channels,
                #         #     hidden_dim,
                #         #     kernel_size,
                #         #     stride=1,
                #         #     padding=self.pad,
                #         #     dilation=1,
                #         #     indice_key="subm0_insconv_adapt",
                #         # )
                #     elif cnt==9:
                #         layer = SubMConv2dInsDilate(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              stride=1,
                #              padding=self.pad,
                #              dilation_max=self.dilated_conv_r_max*2
                #         )
                #         # layer = conv(
                #         #     n_channels,
                #         #     hidden_dim,
                #         #     kernel_size,
                #         #     stride=1,
                #         #     padding=self.pad,
                #         #     dilation=1,
                #         #     indice_key="subm1_insconv_adapt",
                #         # )
                #     elif cnt==18:
                #         layer = SubMConv2dInsDilate(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              stride=1,
                #              padding=self.pad,
                #              dilation_max=self.dilated_conv_r_max*4
                #         )
                #         # layer = conv(
                #         #     n_channels,
                #         #     hidden_dim,
                #         #     kernel_size,
                #         #     stride=1,
                #         #     padding=self.pad,
                #         #     dilation=1,
                #         #     indice_key="subm2_insconv_adapt",
                #         # )
                #     elif self.use_ins_conv:
                #         layer = SubMConv2dInsDilate(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              stride=1,
                #              padding=self.pad,
                #              dilation_max=1
                #         )
                #     else:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )
                # elif self.dilated_conv_type=="two_layers":
                #     if cnt==6:
                #         layer = SubMConv2dInsDilate(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              stride=1,
                #              padding=self.pad,
                #              dilation_max=self.dilated_conv_r_max
                #         )
                #     elif cnt==15:
                #         layer = SubMConv2dInsDilate(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              stride=1,
                #              padding=self.pad,
                #              dilation_max=self.dilated_conv_r_max
                #         )
                #     elif self.use_ins_conv:
                #         layer = SubMConv2dInsDilate(
                #              n_channels,
                #              hidden_dim,
                #              kernel_size,
                #              stride=1,
                #              padding=self.pad,
                #              dilation_max=1
                #         )
                #     else:
                #         layer = conv(
                #             n_channels,
                #             hidden_dim,
                #             kernel_size,
                #             stride=1,
                #             padding=self.pad,
                #             dilation=1,
                #             indice_key="subm0",
                #         )

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
    def rearrange_inputs(self, x, ins_indices_batch, ins_ids):
        x_features_list =[]
        x_indices_list =[]
        ins_indices_list = []
        for i in ins_ids:
            x_features_list.append(x.features[(ins_indices_batch==i).nonzero()].squeeze(1))
            x_indices_list.append(x.indices[(ins_indices_batch==i).nonzero()].squeeze(1))
            ins_indices_list.append(ins_indices_batch[ins_indices_batch==i])
        x.features = torch.cat(x_features_list, dim=0)
        x.indices = torch.cat(x_indices_list, dim=0)
        ins_indices_batch = torch.cat(ins_indices_list,dim=0)
        return x, ins_indices_batch


    def create_dilated_indices(self, x, ins_indices_batch, ins_ids, dilation_max, ksize=3):
        features_list =[]
        ins_indices_list = []
        outids_list =[]
        indice_pairs_list = []
        indice_pair_num_list = []
        cnt = 0
        for i in ins_ids:
            # # pdb.set_trace()
            # try:
            #     out = self.norm(x.features[(ins_indices_batch==i).nonzero(),].reshape([-1,1,C]).permute([1,2,0])) ## HWxBxC -> BxCxHW
            #     x.features[(ins_indices_batch==i).nonzero(),] = out.permute([2,0,1]) #.reshape(-1)
            # except Exception as e:
            #     print(e)
            #     # print((ins_indices_batch==i).nonzero())
            #     # pdb.set_trace()

            "TODO: dilation depends on instance size"
            ins_indices = x.indices[(ins_indices_batch==i).nonzero()].squeeze(1)
            h_ratio = (ins_indices[:,1].max() - ins_indices[:,1].min()).float()/x.spatial_shape[0]
            w_ratio = (ins_indices[:,2].max() - ins_indices[:,2].min()).float()/x.spatial_shape[1]
            d = max(1, math.ceil(max(h_ratio, w_ratio)*dilation_max))
            
            # outids0, indice_pairs0, indice_pair_num0 = spconv.ops.get_indice_pairs(x.indices, x.batch_size, x.spatial_shape, ksize=3, stride=1, padding=0, dilation=1,subm=True)
            # try:
            outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
                ins_indices,
                x.batch_size,
                x.spatial_shape,
                ksize=ksize,
                stride=1,
                padding=self.pad,
                dilation=d,
                out_padding=0,
                subm=True,
                transpose=False,
                grid=None,
                use_hash=False,
                )
            # pdb.set_trace()
            # outids[:,0] = i
            # except:
            indice_pairs[indice_pairs!=-1] += cnt

            """
            Replace -1 with center pixel value to avoid CUDA error: an illegal memory access was encountered.
            Because, for each instance, there are -1 in the end of indice_pairs. When combining more than one
            instance, there will be -1 in the middle which causes CUDA memory error.
            """
            # if i==0:
            #     pdb.set_trace()
            # center = (indice_pairs.shape[1]-1)//2
            center = torch.argmax(indice_pair_num)
            for t in range(indice_pairs.shape[1]):
                if t!=center:
                    replace_idx = indice_pairs[0,t,:]==-1
                    indice_pairs[:,t,replace_idx] = indice_pairs[:,center,replace_idx]
                    # indice_pair_num[t] = indice_pair_num[center]
            valid_idx = indice_pairs[0,center,:]!=-1
            outids = outids[valid_idx]
            indice_pairs = indice_pairs[:,:,valid_idx]
            indice_pair_num = indice_pair_num*0 + valid_idx.int().sum()
            # if i==0:
            #     pdb.set_trace()

            # "Divide according to indice_pair_num"

            # ins_indices_list.append(ins_indices)
            outids_list.append(outids)
            indice_pairs_list.append(indice_pairs)
            indice_pair_num_list.append(indice_pair_num)
            cnt += (ins_indices_batch==i).int().sum()

        # ins_indices = torch.cat(ins_indices_list, dim=0)
        outids = torch.cat(outids_list, dim=0)
        indice_pairs = torch.cat(indice_pairs_list, dim=-1)
        indice_pair_num = torch.stack(indice_pair_num_list).sum(dim=0).int()


        # ins_indices0 = x.indices
        # outids0, indice_pairs0, indice_pair_num0 = spconv.ops.get_indice_pairs(x.indices, x.batch_size, x.spatial_shape, ksize=3, stride=1, padding=0, dilation=1,subm=True)
        

        # (indice_pairs[0,0]!=-1).int().sum()
        # indice_pairs[0,0,10006:]
        # -1 in indice_pairs[0,0,:10006]

        # outids = outids0
        # indice_pairs = indice_pairs0
        # indice_pair_num = indice_pair_num0
        # indice_pairs[indice_pairs==-1] = 0
        # indice_pair_num = indice_pair_num-100

        # if resort_input:
        #     features = torch.cat(features_list, dim=0)
        #     return features, ins_indices, indice_tuple
        # else:
        indice_tuple = (outids, x.indices, indice_pairs, indice_pair_num, x.spatial_shape)
        return indice_tuple


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
        # pdb.set_trace()
        # output = x
        "TODO: change ins_indices_batch to start-end slice to save GPU Memory"
        ins_ids = torch.unique(ins_indices_batch)
        if self.dilated_conv_type!="none":
            print("rearrange_inputs")
            x, ins_indices_batch = self.rearrange_inputs(x, ins_indices_batch, ins_ids)
        # ins_indices_batch = ins_indices_batch[...,None].expand_as(x.features)
        # pdb.set_trace()

        # if self.amp_enable:
        #     x.features = x.features.half()
        # pdb.set_trace()

        if "ori" in self.dilated_conv_type:
            if self.use_ins_conv:
                x.indice_dict["subm0_insconv"] = self.create_dilated_indices(x, ins_indices_batch, ins_ids, 1)
            # if self.dilated_conv_type!="none":
                # assert self.use_ins_conv
            x.indice_dict["subm0_insconv_adapt"] = self.create_dilated_indices(x, ins_indices_batch, ins_ids, self.dilated_conv_r_max)
            if "progressive" in self.dilated_conv_type:
                x.indice_dict["subm1_insconv_adapt"] = self.create_dilated_indices(x, ins_indices_batch, ins_ids, self.dilated_conv_r_max*2)
                x.indice_dict["subm2_insconv_adapt"] = self.create_dilated_indices(x, ins_indices_batch, ins_ids, self.dilated_conv_r_max*4)

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
            # if isinstance(layer, SubMConv2dInsDilate):
            #     pdb.set_trace()
            if isinstance(layer, SparseInsGNBNIN) \
                or isinstance(layer, SparseInsECA) \
                or isinstance(layer, SubMConv2dInsDilate)\
                or isinstance(layer, SubMConv2dInsDilateASPP)\
                or isinstance(layer, SubMConv2dASPP):
                # x = layer(x, ins_indices_batch, ins_ids, ins_indices_len)
                x = layer(x, ins_indices_batch, ins_ids)
            else:
                # print(idx, x.indice_dict.keys(), x.indices.shape)
                try:
                    x = layer(x)
                except Exception as e:
                    print(e)
                    pdb.set_trace()
                # print(idx, x.indice_dict.keys())
                # pdb.set_trace()

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
                        if self.use_res_skip_conv:
                            if idx==5:
                                layer_list = self.res_skip_convs[0]
                            if idx==14:
                                layer_list = self.res_skip_convs[1]
                            if idx==23:
                                layer_list = self.res_skip_convs[2]
                            for l in layer_list:
                                if isinstance(l, SparseInsGNBNIN):
                                    res = l(res, ins_indices_batch, ins_ids)
                                else:
                                    res = l(res)
                        x = self.add_sparse([x,res])



        x.features = x.features #.float()
        output = x
        return output

    # def _get_layer_name(self, i: int):
    #     layer_name = "body_conv_fcn{}".format(i + 1)
    #     layer_name = "layer".format(i)
    #     return layer_name
