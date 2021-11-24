import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from equiv_aux_utils import rotate4, rotate_c_, rotate_cc_
from typing import Union, Tuple


class RotationTensorToGroupConv(nn.Module):
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
        k, k_r, k_rr, k_rrr = rotate4(self.k)
        kernel = torch.cat((k, k_r, k_rr, k_rrr), dim=0)
        bias_tensor = torch.cat((self.k_bias, self.k_bias, self.k_bias, self.k_bias), dim=0) if self.bias else None
        return F.conv2d(x, kernel, bias_tensor,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)


class RotationGroupToGroupConv(nn.Module):
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
            self.conv_module = QuadToQuadEfficientConv1d(in_channels=in_channels,
                                                         out_channels=out_channels,
                                                         stride=stride,
                                                         padding=padding,
                                                         dilation=dilation,
                                                         groups=groups,
                                                         bias=bias)
        else:
            self.conv_module = QuadToQuadEfficientConv(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding,
                                                       dilation=dilation,
                                                       groups=groups,
                                                       bias=bias)

    def forward(self, x):
        return self.conv_module(x)





class QuadToQuadEfficientConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.bias = bias
        init_normalization = 4 * in_channels
        self.k_a = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)
        self.k_b = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)
        self.k_c = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)
        self.k_d = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)
        self.k_bias = Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x):
        k_a, k_b, k_c, k_d = self.k_a, self.k_b, self.k_c, self.k_d

        k1 = torch.cat((k_a, k_b, k_c, k_d), dim=1)
        k2 = torch.cat((k_d, k_a, k_b, k_c), dim=1)
        k3 = torch.cat((k_c, k_d, k_a, k_b), dim=1)
        k4 = torch.cat((k_b, k_c, k_d, k_a), dim=1)
        kernel = torch.cat((k1, k2, k3, k4), dim=0)
        bias_tensor = torch.cat([self.k_bias] * 4, dim=0) if self.bias else None

        return F.conv2d(x, kernel, bias_tensor, **self.kwargs)

class QuadToQuadEfficientConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.bias = bias
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        init_normalization = 4 * in_channels * kernel_size[0] * kernel_size[1]
        self.k_a = Parameter(torch.randn(out_channels, in_channels, 3, 3) / init_normalization)
        self.k_b = Parameter(torch.randn(out_channels, in_channels, 3, 3) / init_normalization)
        self.k_c = Parameter(torch.randn(out_channels, in_channels, 3, 3) / init_normalization)
        self.k_d = Parameter(torch.randn(out_channels, in_channels, 3, 3) / init_normalization)
        self.k_bias = Parameter(torch.randn(out_channels) * 0.5) if bias else None

    def forward(self, x):
        k_a, k_b, k_c, k_d = self.k_a, self.k_b, self.k_c, self.k_d
        k_a_r, k_b_r, k_c_r, k_d_r = rotate_c_(self.k_a, self.k_b, self.k_c, self.k_d)
        k_a_rr, k_b_rr, k_c_rr, k_d_rr = rotate_c_(k_a_r, k_b_r, k_c_r, k_d_r)
        k_a_rrr, k_b_rrr, k_c_rrr, k_d_rrr = rotate_cc_(self.k_a, self.k_b, self.k_c, self.k_d)

        k1 = torch.cat((k_a, k_b, k_c, k_d), dim=1)
        k2 = torch.cat((k_d_r, k_a_r, k_b_r, k_c_r), dim=1)
        k3 = torch.cat((k_c_rr, k_d_rr, k_a_rr, k_b_rr), dim=1)
        k4 = torch.cat((k_b_rrr, k_c_rrr, k_d_rrr, k_a_rrr), dim=1)
        kernel = torch.cat((k1, k2, k3, k4), dim=0)
        bias_tensor = torch.cat([self.k_bias] * 4, dim=0) if self.bias else None

        return F.conv2d(x, kernel, bias_tensor, **self.kwargs)
