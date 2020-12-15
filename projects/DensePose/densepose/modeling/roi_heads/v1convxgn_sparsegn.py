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
class SparseInsDilateConv(nn.Module):
    def __init__(self, num_groups, out_channels):
        super(SparseInsDilateConv, self).__init__()


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
            # # pdb.set_trace()
            # try:
            #     out = self.norm(x.features[(ins_indices_batch==i).nonzero(),].reshape([-1,1,C]).permute([1,2,0])) ## HWxBxC -> BxCxHW
            #     x.features[(ins_indices_batch==i).nonzero(),] = out.permute([2,0,1]) #.reshape(-1)
            # except Exception as e:
            #     print(e)
            #     # print((ins_indices_batch==i).nonzero())
            #     # pdb.set_trace()

            "TODO: dilation depends on instance size"
            pdb.set_trace()
            outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                x.indices[(ins_indices_batch==i).nonzero(),],
                x.batch_size,
                x.spatial_shape,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=d,
                )
            x.indice_dict["ins_dilate"] = (outids, x.indices,
                                                  indice_pairs,
                                                  indice_pair_num,
                                                  x.spatial_shape)




        return x

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
                print(e)
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
        # self.use_eca = cfg.MODEL.CONDINST.IUVHead.Efficient_Channel_Attention
        self.use_eca = False
        self.use_ins_eca = cfg.MODEL.CONDINST.IUVHead.INSTANCE_EFFICIENT_CHANNEL_ATTENTION
        self.use_res_input    = cfg.MODEL.CONDINST.IUVHead.RESIDUAL_INPUT
        self.use_res_after_relu    = cfg.MODEL.CONDINST.IUVHead.RESIDUAL_SKIP_AFTER_RELU
        self.use_res_later   = cfg.MODEL.CONDINST.IUVHead.RESIDUAL_SKIP_LATER
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
                                use_submconv=True, use_deconv=False, use_weight_std=self.use_weight_std)
            cnt = 0
            self.layers = []
            self.pad = 0
            for i in range(self.n_stacked_convs):
                if self.dilated_conv_type=="one_layer_ori":
                    if i in [3]:
                        layer = conv(
                            n_channels,
                            hidden_dim,
                            kernel_size,
                            stride=1,
                            padding=self.pad,
                            dilation=self.dilated_conv_r_max,
                            indice_key="subm0_insconv_adapt",
                        )
                    else:
                        layer = conv(
                            n_channels,
                            hidden_dim,
                            kernel_size,
                            stride=1,
                            padding=self.pad,
                            dilation=1,
                            indice_key="subm0_insconv",
                        )
                elif self.dilated_conv_type=="progressive_ori":
                    if i==3:
                        layer = conv(
                            n_channels,
                            hidden_dim,
                            kernel_size,
                            stride=1,
                            padding=self.pad,
                            dilation=1,
                            indice_key="subm0_insconv_adapt",
                        )
                    elif i==6:
                        layer = conv(
                            n_channels,
                            hidden_dim,
                            kernel_size,
                            stride=1,
                            padding=self.pad,
                            dilation=1,
                            indice_key="subm1_insconv_adapt",
                        )
                    elif i==9:
                        layer = conv(
                            n_channels,
                            hidden_dim,
                            kernel_size,
                            stride=1,
                            padding=self.pad,
                            dilation=1,
                            indice_key="subm2_insconv_adapt",
                        )
                    else:
                        layer = conv(
                            n_channels,
                            hidden_dim,
                            kernel_size,
                            stride=1,
                            padding=self.pad,
                            dilation=1,
                            indice_key="subm0_insconv",
                        )
                else:
                    if i<12:
                        layer = conv(
                            n_channels,
                            hidden_dim,
                            kernel_size,
                            stride=1,
                            padding=self.pad,
                            dilation=1,
                            indice_key="subm0",
                        )
                    else:
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
                        elif self.dilated_conv_type=="one_layer":
                            if i in [12]:
                                layer = conv(
                                    n_channels,
                                    hidden_dim,
                                    kernel_size,
                                    stride=1,
                                    padding=self.pad,
                                    dilation=self.dilated_conv_r_max,
                                    indice_key="subm0_insconv_adapt",
                                )
                            else:
                                layer = conv(
                                    n_channels,
                                    hidden_dim,
                                    kernel_size,
                                    stride=1,
                                    padding=self.pad,
                                    dilation=1,
                                    indice_key="subm0_insconv",
                                )
                        # elif self.dilated_conv_type=="all_layers":
                        #     if i>2:
                        #         layer = conv(
                        #             n_channels,
                        #             hidden_dim,
                        #             kernel_size,
                        #             stride=1,
                        #             padding=self.pad,
                        #             dilation=1,
                        #             indice_key="subm0_adapt",
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
                        elif self.dilated_conv_type=="progressive":
                            if i==12:
                                layer = conv(
                                    n_channels,
                                    hidden_dim,
                                    kernel_size,
                                    stride=1,
                                    padding=self.pad,
                                    dilation=1,
                                    indice_key="subm0_insconv_adapt",
                                )
                            elif i==15:
                                layer = conv(
                                    n_channels,
                                    hidden_dim,
                                    kernel_size,
                                    stride=1,
                                    padding=self.pad,
                                    dilation=1,
                                    indice_key="subm1_insconv_adapt",
                                )
                            elif i==18:
                                layer = conv(
                                    n_channels,
                                    hidden_dim,
                                    kernel_size,
                                    stride=1,
                                    padding=self.pad,
                                    dilation=1,
                                    indice_key="subm2_insconv_adapt",
                                )
                            else:
                                layer = conv(
                                    n_channels,
                                    hidden_dim,
                                    kernel_size,
                                    stride=1,
                                    padding=self.pad,
                                    dilation=1,
                                    indice_key="subm0_insconv",
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
        # output = x
        "TODO: change ins_indices_batch to start-end slice to save GPU Memory"
        ins_ids = torch.unique(ins_indices_batch)
        x, ins_indices_batch = self.rearrange_inputs(x, ins_indices_batch, ins_ids)
        # ins_indices_batch = ins_indices_batch[...,None].expand_as(x.features)
        # pdb.set_trace()

        # if self.amp_enable:
        #     x.features = x.features.half()
        # pdb.set_trace()

        if self.use_ins_conv:
            x.indice_dict["subm0_insconv"] = self.create_dilated_indices(x, ins_indices_batch, ins_ids, 1)
        if self.dilated_conv_type!="none":
            # assert self.use_ins_conv
            x.indice_dict["subm0_insconv_adapt"] = self.create_dilated_indices(x, ins_indices_batch, ins_ids, self.dilated_conv_r_max)
            if "progressive" in self.dilated_conv_type:
                x.indice_dict["subm1_insconv_adapt"] = self.create_dilated_indices(x, ins_indices_batch, ins_ids, self.dilated_conv_r_max*2)
                x.indice_dict["subm2_insconv_adapt"] = self.create_dilated_indices(x, ins_indices_batch, ins_ids, self.dilated_conv_r_max*4)

        res = None
        # pdb.set_trace()
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
                # print(idx, x.indice_dict.keys(), x.indices.shape)
                try:
                    x = layer(x)
                except:
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
                        x = self.add_sparse([x,res])



        x.features = x.features #.float()
        output = x
        return output

    # def _get_layer_name(self, i: int):
    #     layer_name = "body_conv_fcn{}".format(i + 1)
    #     layer_name = "layer".format(i)
    #     return layer_name
