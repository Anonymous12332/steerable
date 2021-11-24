import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from equiv_aux_utils import rotate4, rotate_c_, flip_, flip, rotate_cc_, vflip, rotate_c, rotate_cc


class SymmetricalKernel(nn.Module):
    def __init__(self, input_depth, k_size):
        super().__init__()
        assert k_size == 3
        self.a = Parameter(torch.randn(1, input_depth))
        self.b = Parameter(torch.randn(1, input_depth))
        self.c = Parameter(torch.randn(1, input_depth))

    def forward(self):
        k1 = torch.stack((self.a, self.b, self.a), dim=2)
        k2 = torch.stack((self.b, self.c, self.b), dim=2)
        kernel = torch.stack((k1, k2, k1), dim=3)
        return kernel


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


# class SobelKernel(nn.Module):
#     def __init__(self, input_depth, k_size):
#         super().__init__()
#         assert k_size == 3
#         self.a = Parameter(torch.randn(1, input_depth))
#         self.b = Parameter(torch.randn(1, input_depth))
#         self.zeros = torch.zeros(1, input_depth, 3)
#         if torch.cuda.is_available():
#             self.zeros = self.zeros.cuda()
#
#     def forward(self):
#         k1 = torch.stack((self.a, self.b, self.a), dim=2)
#         k_x = torch.stack((-k1, self.zeros, k1), dim=3)
#         k_y = -k_x.transpose(-1, -2)
#         kernel = torch.cat((k_x, k_y), dim=0)
#         return kernel


class SobelKernel(nn.Module):
    def __init__(self, input_depth, k_size):
        super().__init__()
        self.k_size = k_size
        half_size = int(k_size // 2)
        self.half_size = half_size
        if k_size % 2 == 0:
            self.block = Parameter(torch.randn(1, input_depth, half_size, half_size))
        else:
            self.block = Parameter(torch.randn(1, input_depth, half_size + 1, half_size))
            self.zeros = torch.zeros(1, input_depth, half_size + 1, 1)
            if torch.cuda.is_available():
                self.zeros = self.zeros.cuda()

        # self.a = Parameter(torch.randn(1, input_depth))
        # self.b = Parameter(torch.randn(1, input_depth))
        # self.zeros = torch.zeros(1, input_depth, 3)
        # if torch.cuda.is_available():
        #     self.zeros = self.zeros.cuda()

    def forward(self):
        if self.k_size % 2 == 0:
            k_x = torch.cat((self.block, -flip(self.block)), dim=3)
            k_x = torch.cat((k_x, vflip(k_x)), dim=2)
        else:
            k_x = torch.cat((self.block, self.zeros, -flip(self.block)), dim=3)
            k_x = torch.cat((k_x, vflip(k_x[:, :, :self.half_size])), dim=2)

        k_y = rotate_cc(k_x)
        kernel = torch.cat((k_x, k_y), dim=0)
        return kernel

        # k1 = torch.stack((self.a, self.b, self.a), dim=2)
        # k_x = torch.stack((-k1, self.zeros, k1), dim=3)
        # k_y = -k_x.transpose(-1, -2)
        # kernel = torch.cat((k_x, k_y), dim=0)
        # return kernel


class SquareKernel(nn.Module):
    def __init__(self, input_depth, k_size):
        super().__init__()
        self.model = {4: SquareKernel4,
                      5: SquareKernel5}[k_size](input_depth)

    def forward(self):
        return self.model()


class SquareKernel5(nn.Module):
    def __init__(self, input_depth):
        super().__init__()
        self.a = Parameter(torch.randn(1, input_depth) / (25 * input_depth))
        self.b = Parameter(torch.randn(1, input_depth) / (25 * input_depth))
        self.c = Parameter(torch.randn(1, input_depth) / (25 * input_depth))
        self.d = Parameter(torch.randn(1, input_depth) / (25 * input_depth))
        self.e = Parameter(torch.randn(1, input_depth) / (25 * input_depth))
        self.f = Parameter(torch.randn(1, input_depth) / (25 * input_depth))
        self.g = Parameter(torch.randn(1, input_depth))
        self.zeros = torch.zeros(1, input_depth)
        self.zeros_5 = torch.zeros(1, input_depth, 5)
        if torch.cuda.is_available():
            self.zeros = self.zeros.cuda()
            self.zeros_5 = self.zeros_5.cuda()

    def forward(self):
        r1 = torch.stack((self.a, self.b, self.c, self.b, self.a), dim=2)
        r2 = torch.stack((self.b, self.d, self.e, self.d, self.b), dim=2)
        r3 = torch.stack((self.c, self.e, self.f, self.e, self.c), dim=2)
        k_y = torch.stack((r1, r2, r3, r2, r1), dim=3)

        r1 = torch.stack((self.zeros, self.g, self.zeros, -self.g, self.zeros), dim=2)
        r2 = torch.stack((-self.g, self.zeros, self.zeros, self.zeros, self.g), dim=2)
        k_x = torch.stack((r1, r2, self.zeros_5, -r2, -r1), dim=3)

        kernel = torch.cat((k_x, k_y), dim=0)
        return kernel


class SquareKernel4(nn.Module):
    def __init__(self, input_depth):
        super().__init__()
        self.a = Parameter(torch.randn(1, input_depth) / (25 * input_depth))
        self.b = Parameter(torch.randn(1, input_depth) / (25 * input_depth))
        self.c = Parameter(torch.randn(1, input_depth) / (25 * input_depth))
        self.d = Parameter(torch.randn(1, input_depth))
        self.zeros = torch.zeros(1, input_depth)
        # self.zeros_4 = torch.zeros(1, input_depth, 4)
        if torch.cuda.is_available():
            self.zeros = self.zeros.cuda()
            # self.zeros_4 = self.zeros_4.cuda()

    def forward(self):
        r1 = torch.stack((self.a, self.b, self.b, self.a), dim=2)
        r2 = torch.stack((self.b, self.c, self.c, self.b), dim=2)
        k_y = torch.stack((r1, r2, r2, r1), dim=3)

        r1 = torch.stack((self.zeros, self.d, -self.d, self.zeros), dim=2)
        r2 = torch.stack((-self.d, self.zeros, self.zeros, self.d), dim=2)
        k_x = torch.stack((r1, r2, -r2, -r1), dim=3)

        kernel = torch.cat((k_x, k_y), dim=0)
        return kernel


class LineKernel(nn.Module):
    def __init__(self, input_depth, k_size):
        super().__init__()
        self.model = {3: LineKernel3,
                      4: LineKernel4,
                      5: LineKernel5}[k_size](input_depth)

    def forward(self):
        return self.model()


class LineKernel3(nn.Module):
    def __init__(self, input_depth):
        super().__init__()
        self.a = Parameter(torch.randn(1, input_depth))
        self.b = Parameter(torch.randn(1, input_depth))
        self.zeros = torch.zeros(1, input_depth)
        self.zeros_3 = torch.zeros(1, input_depth, 3)
        if torch.cuda.is_available():
            self.zeros = self.zeros.cuda()
            self.zeros_3 = self.zeros_3.cuda()

    def forward(self):
        r1 = torch.stack((self.zeros, self.a, self.zeros), dim=2)
        r2 = torch.stack((-self.a, self.zeros, -self.a), dim=2)
        k_y = torch.stack((r1, r2, r1), dim=3)

        r1 = torch.stack((self.b, self.zeros, -self.b), dim=2)
        k_x = torch.stack((r1, self.zeros_3, -r1), dim=3)

        kernel = torch.cat((k_x, k_y), dim=0)
        return kernel


class LineKernel4(nn.Module):
    def __init__(self, input_depth):
        super().__init__()
        self.a = Parameter(torch.randn(1, input_depth))
        self.b = Parameter(torch.randn(1, input_depth))
        self.c = Parameter(torch.randn(1, input_depth))
        self.d = Parameter(torch.randn(1, input_depth))
        self.zeros = torch.zeros(1, input_depth)
        # self.zeros_4 = torch.zeros(1, input_depth, 4)
        if torch.cuda.is_available():
            self.zeros = self.zeros.cuda()
            # self.zeros_4 = self.zeros_4.cuda()

    def forward(self):
        r1 = torch.stack((self.a, self.b, -self.b, -self.a), dim=2)
        r2 = torch.stack((self.b, self.c, -self.c, -self.b), dim=2)
        k_x = torch.stack((r1, r2, -r2, -r1), dim=3)

        r1 = torch.stack((self.zeros, self.d, self.d, self.zeros), dim=2)
        r2 = torch.stack((-self.d, self.zeros, self.zeros, -self.d), dim=2)
        k_y = torch.stack((r1, r2, r2, r1), dim=3)

        kernel = torch.cat((k_x, k_y), dim=0)
        return kernel


class LineKernel5(nn.Module):
    def __init__(self, input_depth):
        super().__init__()
        self.a = Parameter(torch.randn(1, input_depth) / (16 * input_depth))
        self.b = Parameter(torch.randn(1, input_depth) / (16 * input_depth))
        self.c = Parameter(torch.randn(1, input_depth) / (16 * input_depth))
        self.d = Parameter(torch.randn(1, input_depth))
        self.e = Parameter(torch.randn(1, input_depth))
        self.f = Parameter(torch.randn(1, input_depth))
        self.zeros = torch.zeros(1, input_depth)
        self.zeros_5 = torch.zeros(1, input_depth, 5)
        if torch.cuda.is_available():
            self.zeros = self.zeros.cuda()
            self.zeros_5 = self.zeros_5.cuda()

    def forward(self):
        r1 = torch.stack((self.a, self.b, self.zeros, -self.b, -self.a), dim=2)
        r2 = torch.stack((self.b, self.c, self.zeros, -self.c, -self.b), dim=2)
        k_x = torch.stack((r1, r2, self.zeros_5, -r2, -r1), dim=3)

        r1 = torch.stack((self.zeros, self.d, self.e, self.d, self.zeros), dim=2)
        r2 = torch.stack((-self.d, self.zeros, self.f, self.zeros, -self.d), dim=2)
        r3 = torch.stack((-self.e, -self.f, self.zeros, -self.f, -self.e), dim=2)
        k_y = torch.stack((r1, r2, r3, r2, r1), dim=3)

        kernel = torch.cat((k_x, k_y), dim=0)
        return kernel


# class SquareKernel(nn.Module):
#     def __init__(self, input_depth):
#         super().__init__()
#         self.k_x = SymmetricalKernel(input_depth=input_depth)
#         self.k_y = SymmetricalKernel(input_depth=input_depth)
#
#     def forward(self):
#         kernel = torch.cat((self.k_x(), self.k_y()), dim=0)
#         return kernel


class UsualKernel(nn.Module):
    def __init__(self, input_depth, k_size):
        super().__init__()
        self.k = Parameter(torch.randn(2, input_depth, k_size, k_size))

    def forward(self):
        return self.k


# class UsualKernel3(nn.Module):
#     def __init__(self, input_depth):
#         super().__init__()
#         self.k = Parameter(torch.randn(2, input_depth, 3, 3))
#
#     def forward(self):
#         return self.k
#
#
# class UsualKernel5(nn.Module):
#     def __init__(self, input_depth):
#         super().__init__()
#         self.k = Parameter(torch.randn(2, input_depth, 5, 5))
#
#     def forward(self):
#         return self.k
#
#
# class UsualKernel4(nn.Module):
#     def __init__(self, input_depth):
#         super().__init__()
#         self.k = Parameter(torch.randn(2, input_depth, 4, 4))
#
#     def forward(self):
#         return self.k


class SteerableConv(nn.Module):
    def __init__(self, tag, input_depth):
        super().__init__()
        self.tag = tag
        try:
            kernel_type, k_size = tag.split('_')
        except:
            kernel_type, k_size = tag, 1
        k_size = int(k_size)
        self.conv_layer = {'symmetrical': SymmetricalKernel,
                           'sobel': SobelKernel,
                           'usual': UsualKernel,
                           'square': SquareKernel,
                           'line': LineKernel,
                           'none': Identity}[kernel_type](input_depth, k_size)

        # self.conv_layer = {'symmetrical': SymmetricalKernel,
        #                    'sobel_3': SobelKernel,
        #                    'usual_3': UsualKernel3,
        #                    'usual_4': UsualKernel4,
        #                    'usual_5': UsualKernel5,
        #                    'square_5': SquareKernel5,
        #                    'square_4': SquareKernel4,
        #                    'line_4': LineKernel4,
        #                    'line_3': LineKernel3,
        #                    'line_5': LineKernel5}[tag](input_depth)

    def forward(self, x):
        if self.tag == 'none':
            return x
        kernel = self.conv_layer()
        return F.conv2d(x, kernel, padding=0)
