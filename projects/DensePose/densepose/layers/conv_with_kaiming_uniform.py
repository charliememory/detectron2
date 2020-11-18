import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d
from .partialconv2d import PartialConv2d
from .deform_conv import DFConv2d
from detectron2.layers.batch_norm import get_norm


class Conv2dWS(Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """
    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        ## Weight Standardization
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)

        x = F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def conv_with_kaiming_uniform(
        norm=None, activation=None,
        use_deformable=False, use_sep=False, use_deconv=False, use_partial_conv=False, use_weight_std=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):

        if use_weight_std:
            assert not use_deformable and not use_deconv and not use_partial_conv, "WS are not added to others conv layers yet"

        if use_deformable:
            conv_func = DFConv2d
        elif use_deconv:
            conv_func = nn.ConvTranspose2d
        elif use_partial_conv:
            conv_func = PartialConv2d
        else:
            if use_weight_std:
                conv_func = Conv2dWS
            else:
                conv_func = Conv2d
        if use_sep:
            assert in_channels == out_channels
            groups = in_channels
        else:
            groups = 1
        conv = conv_func(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=groups,
            bias=(norm is None)
        )
        if not use_deformable:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(conv.weight, a=1)
            if norm is None:
                nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if norm is not None and len(norm) > 0:
            if norm == "GN":
                norm_module = nn.GroupNorm(32, out_channels)
            else:
                norm_module = get_norm(norm, out_channels)
            module.append(norm_module)
        if activation is not None:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv
