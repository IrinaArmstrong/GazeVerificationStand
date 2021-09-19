import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------- Convolutional Block Attention Module (CBAM) -----------


# Channel Attention Module
class Flatten(nn.Module):
    """ Layer for flattening n-dimensional input tensors to [dim_0, -1] shape. """
    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):
    """
    The implementation of the Channel Attention block.
    """
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=('avg', 'max')):
        super(ChannelGate, self).__init__()
        self._gate_channels = gate_channels
        self._mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self._pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self._pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool1d( x, x.size(2), stride=x.size(2))
                channel_att_raw = self._mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool1d( x, x.size(2), stride=x.size(2))
                channel_att_raw = self._mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool1d( x, 2, x.size(2), stride=x.size(2))
                channel_att_raw = self._mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self._mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).expand_as(x)
        return x * scale


# Spatial Attention Module (SAM)
class BasicConv(nn.Module):
    """
    Applies a 1D convolution over an input signal with
    following batch normalization and ReLu over output.
    """
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    """ Make stack of Max Pooling and Average Pooling operations subsequently over input data. """
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    """
    The implementation of the Spatial Attention mechanism
    as 1d convolution over Max+Avg Pooled input with sigmoid activation in the end.
    """
    def __init__(self, kernel_size: int = 3):
        super(SpatialGate, self).__init__()
        self._kernel_size = kernel_size
        self._compress = ChannelPool()
        self._spatial = BasicConv(2, 1, self._kernel_size, stride=1,
                                  padding=(self._kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self._compress(x)
        x_out = self._spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    """
    Full convolutional bloch attention mechanism module (CBAM).
    """
    def __init__(self, gate_channels, reduction_ratio=8,
                 pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels,
                                       reduction_ratio, pool_types)
        self.no_spatial= no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out