import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from equiv_aux_utils import rotate4, rotate_c_, flip_, flip, rotate_cc_, rotate_cc, rotate_c, rotate_180
from typing import Union, Tuple


class FlipRotationTensorToGroupConv(nn.Module):
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
        init_normalization = 8 * in_channels * kernel_size[0] * kernel_size[1]
        self.k = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)
        self.k_bias = Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x):
        k_f = flip(self.k)
        k, k_r, k_rr, k_rrr = rotate4(self.k)
        k_f, k_r_f, k_rr_f, k_rrr_f = rotate4(k_f)
        kernel = torch.cat((k, k_f, k_r, k_r_f, k_rr, k_rr_f, k_rrr, k_rrr_f), dim=0)
        bias_tensor = torch.cat([self.k_bias] * 8, dim=0) if self.bias else None
        return F.conv2d(x, kernel, bias_tensor,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)


class FlipRotationGroupToGroupConv(nn.Module):
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
            self.conv_module = OctaveToOctaveEfficientConv1d(in_channels=in_channels,
                                                             out_channels=out_channels,
                                                             stride=stride,
                                                             padding=padding,
                                                             dilation=dilation,
                                                             groups=groups,
                                                             bias=bias)
        else:
            self.conv_module = OctaveToOctaveEfficientConv(in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           kernel_size=kernel_size,
                                                           stride=stride,
                                                           padding=padding,
                                                           dilation=dilation,
                                                           groups=groups,
                                                           bias=bias)

    def forward(self, x):
        return self.conv_module(x)


class OctaveToOctaveEfficientConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.bias = bias
        init_normalization = 8 * in_channels
        self.k_c_0 = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)
        self.k_c_1 = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)

        self.k_r_0 = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)
        self.k_r_1 = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)

        self.k_b_0 = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)
        self.k_b_1 = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)

        self.k_l_0 = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)
        self.k_l_1 = Parameter(torch.randn(out_channels, in_channels, 1, 1) / init_normalization)

        self.k_bias = Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x):
        k_c_0, k_c_1, k_r_0, k_r_1, k_b_0, k_b_1, k_l_0, k_l_1 = self.k_c_0, self.k_c_1, \
                                                                 self.k_r_0, self.k_r_1, \
                                                                 self.k_b_0, self.k_b_1, \
                                                                 self.k_l_0, self.k_l_1

        k_t_n = torch.cat((k_c_0, k_c_1, k_r_0, k_r_1, k_b_0, k_b_1, k_l_0, k_l_1), dim=1)
        k_t_f = torch.cat((k_c_1, k_c_0, k_l_1, k_l_0, k_b_1, k_b_0, k_r_1, k_r_0), dim=1)
        k_r_n = torch.cat((k_l_0, k_l_1, k_c_0, k_c_1, k_r_0, k_r_1, k_b_0, k_b_1), dim=1)
        k_r_f = torch.cat((k_r_1, k_r_0, k_c_1, k_c_0, k_l_1, k_l_0, k_b_1, k_b_0), dim=1)
        k_b_n = torch.cat((k_b_0, k_b_1, k_l_0, k_l_1, k_c_0, k_c_1, k_r_0, k_r_1), dim=1)
        k_b_f = torch.cat((k_b_1, k_b_0, k_r_1, k_r_0, k_c_1, k_c_0, k_l_1, k_l_0), dim=1)
        k_l_n = torch.cat((k_r_0, k_r_1, k_b_0, k_b_1, k_l_0, k_l_1, k_c_0, k_c_1), dim=1)
        k_l_f = torch.cat((k_l_1, k_l_0, k_b_1, k_b_0, k_r_1, k_r_0, k_c_1, k_c_0), dim=1)
        kernel = torch.cat((k_t_n, k_t_f, k_r_n, k_r_f, k_b_n, k_b_f, k_l_n, k_l_f), dim=0)
        bias_tensor = torch.cat([self.k_bias] * 8, dim=0) if self.bias else None

        return F.conv2d(x, kernel, bias_tensor, **self.kwargs)


class OctaveToOctaveEfficientConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.bias = bias
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        init_normalization = 8 * in_channels * kernel_size[0] * kernel_size[1]
        self.k_c_0 = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)
        self.k_c_1 = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)

        self.k_r_0 = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)
        self.k_r_1 = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)

        self.k_b_0 = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)
        self.k_b_1 = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)

        self.k_l_0 = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)
        self.k_l_1 = Parameter(torch.randn(out_channels, in_channels, *kernel_size) / init_normalization)
        self.k_bias = Parameter(torch.randn(out_channels) * 0.5) if bias else None

    def forward(self, x):
        k_c_0, k_c_0_r, k_c_0_rr, k_c_0_rrr = rotate4(self.k_c_0)
        k_r_0, k_r_0_r, k_r_0_rr, k_r_0_rrr = rotate4(self.k_r_0)
        k_b_0, k_b_0_r, k_b_0_rr, k_b_0_rrr = rotate4(self.k_b_0)
        k_l_0, k_l_0_r, k_l_0_rr, k_l_0_rrr = rotate4(self.k_l_0)

        k_c_1, k_c_1_r, k_c_1_rr, k_c_1_rrr = rotate4(self.k_c_1)
        k_r_1, k_r_1_r, k_r_1_rr, k_r_1_rrr = rotate4(self.k_r_1)
        k_b_1, k_b_1_r, k_b_1_rr, k_b_1_rrr = rotate4(self.k_b_1)
        k_l_1, k_l_1_r, k_l_1_rr, k_l_1_rrr = rotate4(self.k_l_1)

        k_c_0_f, k_c_0_r_f, k_c_0_rr_f, k_c_0_rrr_f = flip_(k_c_0, k_c_0_r, k_c_0_rr, k_c_0_rrr)
        k_r_0_f, k_r_0_r_f, k_r_0_rr_f, k_r_0_rrr_f = flip_(k_r_0, k_r_0_r, k_r_0_rr, k_r_0_rrr)
        k_b_0_f, k_b_0_r_f, k_b_0_rr_f, k_b_0_rrr_f = flip_(k_b_0, k_b_0_r, k_b_0_rr, k_b_0_rrr)
        k_l_0_f, k_l_0_r_f, k_l_0_rr_f, k_l_0_rrr_f = flip_(k_l_0, k_l_0_r, k_l_0_rr, k_l_0_rrr)

        k_c_1_f, k_c_1_r_f, k_c_1_rr_f, k_c_1_rrr_f = flip_(k_c_1, k_c_1_r, k_c_1_rr, k_c_1_rrr)
        k_r_1_f, k_r_1_r_f, k_r_1_rr_f, k_r_1_rrr_f = flip_(k_r_1, k_r_1_r, k_r_1_rr, k_r_1_rrr)
        k_b_1_f, k_b_1_r_f, k_b_1_rr_f, k_b_1_rrr_f = flip_(k_b_1, k_b_1_r, k_b_1_rr, k_b_1_rrr)
        k_l_1_f, k_l_1_r_f, k_l_1_rr_f, k_l_1_rrr_f = flip_(k_l_1, k_l_1_r, k_l_1_rr, k_l_1_rrr)

        k_t_n = torch.cat((k_c_0, k_c_1, k_r_0, k_r_1, k_b_0, k_b_1, k_l_0, k_l_1), dim=1)
        k_t_f = torch.cat((k_c_1_f, k_c_0_f, k_l_1_f, k_l_0_f, k_b_1_f, k_b_0_f, k_r_1_f, k_r_0_f), dim=1)
        k_r_n = torch.cat((k_l_0_r, k_l_1_r, k_c_0_r, k_c_1_r, k_r_0_r, k_r_1_r, k_b_0_r, k_b_1_r), dim=1)
        k_r_f = torch.cat(
            (k_r_1_rrr_f, k_r_0_rrr_f, k_c_1_rrr_f, k_c_0_rrr_f, k_l_1_rrr_f, k_l_0_rrr_f, k_b_1_rrr_f, k_b_0_rrr_f),
            dim=1)
        k_b_n = torch.cat((k_b_0_rr, k_b_1_rr, k_l_0_rr, k_l_1_rr, k_c_0_rr, k_c_1_rr, k_r_0_rr, k_r_1_rr), dim=1)
        k_b_f = torch.cat(
            (k_b_1_rr_f, k_b_0_rr_f, k_r_1_rr_f, k_r_0_rr_f, k_c_1_rr_f, k_c_0_rr_f, k_l_1_rr_f, k_l_0_rr_f), dim=1)
        k_l_n = torch.cat((k_r_0_rrr, k_r_1_rrr, k_b_0_rrr, k_b_1_rrr, k_l_0_rrr, k_l_1_rrr, k_c_0_rrr, k_c_1_rrr),
                          dim=1)
        k_l_f = torch.cat((k_l_1_r_f, k_l_0_r_f, k_b_1_r_f, k_b_0_r_f, k_r_1_r_f, k_r_0_r_f, k_c_1_r_f, k_c_0_r_f),
                          dim=1)

        kernel = torch.cat((k_t_n, k_t_f, k_r_n, k_r_f, k_b_n, k_b_f, k_l_n, k_l_f), dim=0)
        bias_tensor = torch.cat([self.k_bias] * 8, dim=0) if self.bias else None
        return F.conv2d(x, kernel, bias_tensor, **self.kwargs)


class SobelGroupConv(nn.Module):
    def __init__(self, in_channels, kernel_size, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        init_normalization = 8 * in_channels * kernel_size[0] * kernel_size[1]
        self.k_x = Parameter(torch.randn(1, in_channels, *kernel_size) / init_normalization)
        self.k_y = Parameter(torch.randn(1, in_channels, *kernel_size) / init_normalization)

    def forward(self, x, num=8):
        assert num == 8
        k_x_2 = rotate_c(self.k_y)
        k_x_4 = -rotate_180(self.k_x)
        k_x_6 = -rotate_cc(self.k_y)

        k_x_1 = -flip(self.k_x)
        k_x_3 = flip(rotate_cc(self.k_y))
        k_x_5 = flip(rotate_180(self.k_x))
        k_x_7 = -flip(rotate_c(self.k_y))

        k_y_2 = -rotate_c(self.k_x)
        k_y_4 = -rotate_180(self.k_y)
        k_y_6 = rotate_cc(self.k_x)

        k_y_1 = flip(self.k_y)
        k_y_3 = flip(rotate_cc(self.k_x))
        k_y_5 = -flip(rotate_180(self.k_y))
        k_y_7 = -flip(rotate_c(self.k_x))

        k_x_kernel = torch.cat((self.k_x, k_x_1, k_x_2, k_x_3, k_x_4, k_x_5, k_x_6, k_x_7), dim=1)
        k_y_kernel = torch.cat((self.k_y, k_y_1, k_y_2, k_y_3, k_y_4, k_y_5, k_y_6, k_y_7), dim=1)

        kernel = torch.cat((k_x_kernel, k_y_kernel), dim=0)
        return F.conv2d(x, kernel, **self.kwargs)


class SquareGroupConv(nn.Module):
    def __init__(self, in_channels, kernel_size, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        init_normalization = 8 * in_channels * kernel_size[0] * kernel_size[1]
        self.k_x = Parameter(torch.randn(1, in_channels, *kernel_size) / init_normalization)
        self.k_y = Parameter(torch.randn(1, in_channels, *kernel_size) / init_normalization)

    def forward(self, x, num=8):
        assert num == 8
        k_x_2 = rotate_c(self.k_x)
        k_x_4 = rotate_180(self.k_x)
        k_x_6 = rotate_cc(self.k_x)

        k_x_1 = -flip(self.k_x)
        k_x_3 = -flip(rotate_cc(self.k_x))
        k_x_5 = -flip(rotate_180(self.k_x))
        k_x_7 = -flip(rotate_c(self.k_x))

        k_y_2 = rotate_c(self.k_y)
        k_y_4 = rotate_180(self.k_y)
        k_y_6 = rotate_cc(self.k_y)

        k_y_1 = flip(self.k_y)
        k_y_3 = flip(rotate_cc(self.k_y))
        k_y_5 = flip(rotate_180(self.k_y))
        k_y_7 = flip(rotate_c(self.k_y))

        k_x_kernel = torch.cat((self.k_x, k_x_1, k_x_2, k_x_3, k_x_4, k_x_5, k_x_6, k_x_7), dim=1)
        k_y_kernel = torch.cat((self.k_y, k_y_1, k_y_2, k_y_3, k_y_4, k_y_5, k_y_6, k_y_7), dim=1)

        kernel = torch.cat((k_x_kernel, k_y_kernel), dim=0)
        return F.conv2d(x, kernel, **self.kwargs)


class SquareGroupConv(nn.Module):
    def __init__(self, in_channels, kernel_size, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        init_normalization = 8 * in_channels * kernel_size[0] * kernel_size[1]
        self.k_x = Parameter(torch.randn(1, in_channels, *kernel_size) / init_normalization)
        self.k_y = Parameter(torch.randn(1, in_channels, *kernel_size) / init_normalization)

    def forward(self, x, num=8):
        assert num == 8
        k_x_2 = rotate_c(self.k_x)
        k_x_4 = rotate_180(self.k_x)
        k_x_6 = rotate_cc(self.k_x)

        k_x_1 = -flip(self.k_x)
        k_x_3 = -flip(rotate_cc(self.k_x))
        k_x_5 = -flip(rotate_180(self.k_x))
        k_x_7 = -flip(rotate_c(self.k_x))

        k_y_2 = rotate_c(self.k_y)
        k_y_4 = rotate_180(self.k_y)
        k_y_6 = rotate_cc(self.k_y)

        k_y_1 = flip(self.k_y)
        k_y_3 = flip(rotate_cc(self.k_y))
        k_y_5 = flip(rotate_180(self.k_y))
        k_y_7 = flip(rotate_c(self.k_y))

        k_x_kernel = torch.cat((self.k_x, k_x_1, k_x_2, k_x_3, k_x_4, k_x_5, k_x_6, k_x_7), dim=1)
        k_y_kernel = torch.cat((self.k_y, k_y_1, k_y_2, k_y_3, k_y_4, k_y_5, k_y_6, k_y_7), dim=1)

        kernel = torch.cat((k_x_kernel, k_y_kernel), dim=0)
        return F.conv2d(x, kernel, **self.kwargs)


class LineGroupConv(nn.Module):
    def __init__(self, in_channels, kernel_size, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        init_normalization = 8 * in_channels * kernel_size[0] * kernel_size[1]
        self.k_x = Parameter(torch.randn(1, in_channels, *kernel_size) / init_normalization)
        self.k_y = Parameter(torch.randn(1, in_channels, *kernel_size) / init_normalization)

    def forward(self, x, num=8):
        assert num == 8
        k_x_2 = -rotate_c(self.k_x)
        k_x_4 = rotate_180(self.k_x)
        k_x_6 = -rotate_cc(self.k_x)

        k_x_1 = -flip(self.k_x)
        k_x_3 = flip(rotate_cc(self.k_x))
        k_x_5 = -flip(rotate_180(self.k_x))
        k_x_7 = flip(rotate_c(self.k_x))

        k_y_2 = -rotate_c(self.k_y)
        k_y_4 = rotate_180(self.k_y)
        k_y_6 = -rotate_cc(self.k_y)

        k_y_1 = flip(self.k_y)
        k_y_3 = -flip(rotate_cc(self.k_y))
        k_y_5 = flip(rotate_180(self.k_y))
        k_y_7 = -flip(rotate_c(self.k_y))

        k_x_kernel = torch.cat((self.k_x, k_x_1, k_x_2, k_x_3, k_x_4, k_x_5, k_x_6, k_x_7), dim=1)
        k_y_kernel = torch.cat((self.k_y, k_y_1, k_y_2, k_y_3, k_y_4, k_y_5, k_y_6, k_y_7), dim=1)

        kernel = torch.cat((k_x_kernel, k_y_kernel), dim=0)
        return F.conv2d(x, kernel, **self.kwargs)
