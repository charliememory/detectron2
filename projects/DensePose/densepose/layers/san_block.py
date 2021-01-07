import torch
from torch import nn

import sannet.san
class SAN_BottleneckGN(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(SAN_BottleneckGN, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = torch.nn.GroupNorm(num_groups=min(in_planes,32), num_channels=in_planes)
        self.sam = sannet.san.SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        # self.bn2 = nn.BatchNorm2d(mid_planes)
        self.bn2 = torch.nn.GroupNorm(num_groups=min(mid_planes,32), num_channels=mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out = out + identity
        return out

class SAN_BottleneckGN_noInputNormReLU(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(SAN_BottleneckGN_noInputNormReLU, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = torch.nn.GroupNorm(num_groups=min(in_planes,32), num_channels=in_planes)
        self.sam = sannet.san.SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        # self.bn2 = nn.BatchNorm2d(mid_planes)
        self.bn2 = torch.nn.GroupNorm(num_groups=min(mid_planes,32), num_channels=mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = x
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        # out = self.relu(self.bn1(out))
        out = out + identity
        return out

class SAN_BottleneckGN_Gated(nn.Module): 
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(SAN_BottleneckGN_Gated, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = torch.nn.GroupNorm(num_groups=min(in_planes,32), num_channels=in_planes)
        self.sam = sannet.san.SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        # self.bn2 = nn.BatchNorm2d(mid_planes)
        self.bn2 = torch.nn.GroupNorm(num_groups=min(mid_planes,32), num_channels=mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        ks = 3
        self.conv_g = nn.Conv2d(in_planes, out_planes, kernel_size=ks)
        self.conv_a = nn.Conv2d(in_planes, out_planes, kernel_size=ks)
        self.pad = nn.ReflectionPad2d(ks//2)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out = out + identity
        out = torch.sigmoid(self.conv_g(self.pad(out))) * torch.tanh(self.conv_a(self.pad(out)))
        return out

class SAN_BottleneckGN_GatedEarly(nn.Module): 
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(SAN_BottleneckGN_GatedEarly, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = torch.nn.GroupNorm(num_groups=min(in_planes,32), num_channels=in_planes)
        self.sam = sannet.san.SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        # self.bn2 = nn.BatchNorm2d(mid_planes)
        self.bn2 = torch.nn.GroupNorm(num_groups=min(mid_planes,32), num_channels=mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        ks = 3
        self.conv_g = nn.Conv2d(in_planes, out_planes, kernel_size=ks)
        self.conv_a = nn.Conv2d(in_planes, out_planes, kernel_size=ks)
        self.pad = nn.ReflectionPad2d(ks//2)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out = torch.sigmoid(self.conv_g(self.pad(out))) * torch.tanh(self.conv_a(self.pad(out)))
        out = out + identity
        return out
