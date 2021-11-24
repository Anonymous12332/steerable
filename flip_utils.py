from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from equiv_aux_utils import flip_, flip


class FlipTensorToGroupConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        init_normalization = 2 * in_channels * kernel_size[0] * kernel_size[1]
        self.k = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)
        self.k_bias = Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x):
        k_t = flip(self.k)
        kernel = torch.cat((self.k, k_t), dim=0)
        bias_tensor = torch.cat((self.k_bias, self.k_bias), dim=0) if self.bias else None
        return F.conv2d(x, kernel, bias_tensor,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)


class FlipGroupToGroupConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__()
        if kernel_size in [1, (1, 1)]:
            self.conv_module = PairToPairEfficientConv1d(in_channels=in_channels,
                                                         out_channels=out_channels,
                                                         stride=stride,
                                                         padding=padding,
                                                         dilation=dilation,
                                                         groups=groups,
                                                         bias=bias)
        else:
            self.conv_module = PairToPairEfficientConv(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding,
                                                       dilation=dilation,
                                                       groups=groups,
                                                       bias=bias)

    def forward(self, x):
        return self.conv_module(x)


class PairToPairEfficientConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.bias = bias
        init_normalization = 2 * in_channels
        self.k_a = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)
        self.k_b = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)
        self.k_bias = Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x):
        k1 = torch.cat((self.k_a, self.k_b), dim=1)
        k2 = torch.cat((self.k_b, self.k_a), dim=1)
        kernel = torch.cat((k1, k2), dim=0)
        bias_tensor = torch.cat([self.k_bias] * 2, dim=0) if self.bias else None

        return F.conv2d(x, kernel, bias_tensor, **self.kwargs)


class PairToPairEfficientConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.bias = bias
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        init_normalization = 2 * in_channels * kernel_size[0] * kernel_size[1]
        self.k_a = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)
        self.k_b = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)
        self.k_bias = Parameter(torch.randn(out_channels) * 0.5) if bias else None

    def forward(self, x):
        k_a_t, k_b_t = flip_(self.k_a, self.k_b)

        k1 = torch.cat((self.k_a, self.k_b), dim=1)
        k2 = torch.cat((k_b_t, k_a_t), dim=1)
        kernel = torch.cat((k1, k2), dim=0)
        bias_tensor = torch.cat([self.k_bias] * 2, dim=0) if self.bias else None

        return F.conv2d(x, kernel, bias_tensor, **self.kwargs)
