from torch import nn

from detectron2.layers import Conv2d
from .partialconv2d import PartialConv2d
from .deform_conv import DFConv2d
from detectron2.layers.batch_norm import get_norm
import spconv
import pdb
# from spconv import ops, SparseConvTensor



# class SparseConvolution(SparseModule):
#     __constants__ = [
#         'stride', 'padding', 'dilation', 'groups', 'bias', 'subm', 'inverse',
#         'transposed', 'output_padding', 'fused_bn'
#     ]

#     def __init__(self,
#                  ndim,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  bias=True,
#                  subm=False,
#                  output_padding=0,
#                  transposed=False,
#                  inverse=False,
#                  indice_key=None,
#                  fused_bn=False,
#                  use_hash=False,
#                  algo=ops.ConvAlgo.Native):

# class SparseGN(nn.Module):
#     def __init__(self, num_groups, out_channels):
#         super(SparseGN, self).__init__()
#         self.gn = nn.GroupNorm(num_groups, out_channels)

#     def forward(self, x: spconv.SparseConvTensor):
#         pdb.set_trace()
#         N, C = x.features.shape
#         batch_indices = x.indices[:,:1].expand_as(x.features)
#         out_batch = []
#         for i in range(x.batch_size):
#             out = self.gn(x[batch_indices==i].reshape([1,C,-1]))
#             out_batch.append(out.reshape([-1,C]))
#         return torch.cat(out_batch, dim=0)


# class ConvAlgo(Enum):
#     Native = 0  # small memory cost, faster when number of points is large.
#     Batch = 1  # high memory cost, faster when number of points is small (< 50000)
#     BatchGemmGather = 2  # high memory cost, faster when number of points medium

# from spconv.conv import (SparseConv2d, SparseConv3d, SparseConvTranspose2d,
#                          SparseConvTranspose3d, SparseInverseConv2d,
#                          SparseInverseConv3d, SubMConv2d, SubMConv3d)
def sparse_conv_with_kaiming_uniform(
        norm=None, activation=None, use_sep=False, use_submconv=True, use_deconv=False):
    def make_conv(
        in_channels, out_channels, kernel_size, 
        stride=1, dilation=1, indice_key="subm0"
    ):
        if use_submconv:
            if use_deconv:
                conv_func = spconv.SparseConvTranspose2d
                # conv_func = spconv.SparseInverseConv2d
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
