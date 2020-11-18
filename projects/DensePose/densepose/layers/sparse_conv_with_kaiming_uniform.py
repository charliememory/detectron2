from torch import nn

from detectron2.layers import Conv2d
from .partialconv2d import PartialConv2d
from .deform_conv import DFConv2d
from detectron2.layers.batch_norm import get_norm
import spconv
from spconv import ops
import spconv.functional as Fsp
import pdb
# from spconv import ops, SparseConvTensor


class SparseConvolutionWS(spconv.conv.SparseConvolution):
    __constants__ = [
        'stride', 'padding', 'dilation', 'groups', 'bias', 'subm', 'inverse',
        'transposed', 'output_padding', 'fused_bn'
    ]

    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            if self.transposed:
                out_spatial_shape = ops.get_deconv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation, self.output_padding)
            else:
                out_spatial_shape = ops.get_conv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation)
        else:
            out_spatial_shape = spatial_shape
        # input.update_grid(out_spatial_shape)
        # t = time.time()
        
        ## Weight Standardization
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)

        if self.conv1x1:
            features = torch.mm(
                input.features,
                weight.view(self.in_channels, self.out_channels))
            if self.bias is not None:
                features += self.bias
            out_tensor = spconv.SparseConvTensor(features, input.indices,
                                                 input.spatial_shape,
                                                 input.batch_size)
            out_tensor.indice_dict = input.indice_dict
            out_tensor.grid = input.grid
            return out_tensor
        datas = input.find_indice_pair(self.indice_key)
        if self.inverse:
            assert datas is not None and self.indice_key is not None
            _, outids, indice_pairs, indice_pair_num, out_spatial_shape = datas
            assert indice_pair_num.shape[0] == np.prod(
                self.kernel_size
            ), "inverse conv must have same kernel size as its couple conv"
        else:
            if self.indice_key is not None and datas is not None:
                outids, _, indice_pairs, indice_pair_num, _ = datas
            else:
                outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                    indices,
                    batch_size,
                    spatial_shape,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.output_padding,
                    self.subm,
                    self.transposed,
                    grid=input.grid,
                    use_hash=self.use_hash)
                input.indice_dict[self.indice_key] = (outids, indices,
                                                      indice_pairs,
                                                      indice_pair_num,
                                                      spatial_shape)
        if self.fused_bn:
            assert self.bias is not None
            out_features = ops.fused_indice_conv(features, weight,
                                                 self.bias,
                                                 indice_pairs.to(device),
                                                 indice_pair_num,
                                                 outids.shape[0], self.inverse,
                                                 self.subm)
        else:
            if self.subm:
                out_features = Fsp.indice_subm_conv(features, weight,
                                                    indice_pairs.to(device),
                                                    indice_pair_num,
                                                    outids.shape[0], self.algo)
            else:
                if self.inverse:
                    out_features = Fsp.indice_inverse_conv(
                        features, weight, indice_pairs.to(device),
                        indice_pair_num, outids.shape[0], self.algo)
                else:
                    out_features = Fsp.indice_conv(features, weight,
                                                   indice_pairs.to(device),
                                                   indice_pair_num,
                                                   outids.shape[0], self.algo)

            if self.bias is not None:
                out_features += self.bias
        out_tensor = spconv.SparseConvTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor

class SubMConv2dWS(SparseConvolutionWS):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 use_hash=False,
                 algo=ops.ConvAlgo.Native):
        super(SubMConv2dWS, self).__init__(2,
                                         in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding,
                                         dilation,
                                         groups,
                                         bias,
                                         True,
                                         indice_key=indice_key,
                                         use_hash=use_hash,
                                         algo=algo)


# class ConvAlgo(Enum):
#     Native = 0  # small memory cost, faster when number of points is large.
#     Batch = 1  # high memory cost, faster when number of points is small (< 50000)
#     BatchGemmGather = 2  # high memory cost, faster when number of points medium

# from spconv.conv import (SparseConv2d, SparseConv3d, SparseConvTranspose2d,
#                          SparseConvTranspose3d, SparseInverseConv2d,
#                          SparseInverseConv3d, SubMConv2d, SubMConv3d)
def sparse_conv_with_kaiming_uniform(
        norm=None, activation=None, use_sep=False, use_submconv=True, use_deconv=False, use_weight_std=False):
    def make_conv(
        in_channels, out_channels, kernel_size, 
        stride=1, dilation=1, indice_key="subm0"
    ):
        if use_weight_std:
            assert use_submconv and not use_deconv, "WS are not added to others spconv layers yet"

        if use_submconv:
            if use_deconv:
                conv_func = spconv.SparseConvTranspose2d
                # conv_func = spconv.SparseInverseConv2d
            else:
                if use_weight_std:
                    conv_func = SubMConv2dWS
                else:
                    conv_func = spconv.SubMConv2d
        else:
            if use_deconv:
                conv_func = spconv.SparseConvTranspose2d
            else:
                conv_func = spconv.SparseConv2d

        if use_sep:
            assert in_channels == out_channels
            groups = in_channels
        else:
            groups = 1

        try:
            conv = conv_func(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=(norm is None), 
                indice_key=indice_key,
                algo=spconv.ops.ConvAlgo.Native
            )
        except:
            conv = conv_func(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=(norm is None), 
                indice_key=indice_key,
                algo=spconv.ops.ConvAlgo.Native
            )

        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if norm is None:
            nn.init.constant_(conv.bias, 0)

        module = [conv,]
        if norm is not None and len(norm) > 0:
            if norm == "GN":
                raise NotImplementedError
                print("GN")
                norm_module = nn.GroupNorm(32, out_channels)
            # elif norm == "SparseGN":
            #     # raise NotImplementedError
            #     norm_module = SparseGN(32, out_channels) 
            elif norm == "BN":
                norm_module = nn.BatchNorm1d(out_channels)
            else:
                raise NotImplementedError
            module.append(norm_module)
        if activation is not None:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return spconv.SparseSequential(*module)
        return conv

    return make_conv
