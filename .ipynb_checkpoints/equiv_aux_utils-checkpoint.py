import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np


def rotate_c(x):
    # rotates 4d tensor 90 degrees clockwisely
    return x.transpose(-2, -1).flip(-1)

def rotate_c_(*args):
    return [rotate_c(arg) for arg in args]

def rotate_cc(x):
    # rotates 4d tensor 90 degrees clockwisely
    return x.transpose(-2, -1).flip(-2)

def rotate_cc_(*args):
    return [rotate_cc(arg) for arg in args]

def rotate4(k):
    k_r = rotate_c(k)
    return [k, k_r, rotate_c(k_r), rotate_cc(k)]

def rotate4_cc(k):
    k_r = rotate_cc(k)
    return [k, k_r, rotate_cc(k_r), rotate_c(k)]

def flip_rotate_8(k):
    return rotate4(k) + flip_(rotate4(k))


def flip(k):
    return k.flip(-1)

def flip_(*args):
    return [arg.flip(-1) for arg in args]

def conv(x, kernel):
    return F.conv2d(x, kernel, padding = 1)

def avgPool2d(x, kSize = 2):
    return F.avg_pool2d(x, kSize)

def avgPool2d_(*args, kSize = 2):
    return [F.avg_pool2d(arg, kSize) for arg in args]

def cat_(x, *args):
    return [torch.cat((x, arg), dim = 1) for arg in args]

def catRelu_(x, *args):
    return [torch.cat((x, F.relu(arg)), dim = 1) for arg in args]

def catReluList(l1, l2):
    return [torch.cat((l1[i], F.relu(l2[i])), dim = 1) for i in range(len(l1))]

def squeeze_(*args):
    return [arg.squeeze() for arg in args]

def averageMultiple(x, num = 8):
    mb, num_depth = x.shape[:2]
    x = x.view(mb, num, int(num_depth / num), *x.shape[2:]).mean(dim = 1)
    return x

def maxMultiple(x, num = 8):
    mb, num_depth = x.shape[:2]
    x = x.view(mb, num, int(num_depth / num), *x.shape[2:]).max(dim = 1)[0]
    return x

def sumOfSquaresMultiple(x, num = 8):
    mb, num_depth = x.shape[:2]
    x = x.view(mb, num, int(num_depth / num), *x.shape[2:])
    x = x * x
    x = x.mean(dim = 1)
    return x

def convList(x, kernels):
    # kernels is a tuple of kernels of the same shape
    k = torch.cat(kernels, dim = 0)
    res = F.conv2d(x, k, padding = 1)
    return torch.split(res, kernels[0].shape[0], dim = 1)

def sumConvList(inputs, kernels):
    inp = torch.cat(inputs, dim = 1)
    k = torch.cat(kernels, dim = 1)
    return F.conv2d(inp, k, padding = 1)


# def catReluSingleMultiple(x, y, num = 8):
#    y_splitted = torch.split(F.relu(y), y.shape[1] / num, dim = 1)
#    y_catted = cat_(x, *ySplitted)
#    return torch.cat(y_catted, dim = 1)

def catReluSingleMultiple(x, y, num = 8):
    mb, num_blockSize, rows, cols = list(y.shape)

    y = F.relu(y).view(mb, num, int(num_blockSize / num), rows, cols)
    x = x.unsqueeze(1).expand(-1, num, -1, -1, -1)

    res = torch.cat((x ,y), dim = 2).view(mb, -1, rows, cols)
    return res

def catSingleMultiple(x, y, num = 8):
    mb, num_blockSize, rows, cols = list(y.shape)

    y = y.view(mb, num, int(num_blockSize / num), rows, cols)
    x = x.unsqueeze(1).expand(-1, num, -1, -1, -1)

    res = torch.cat((x ,y), dim = 2).view(mb, -1, rows, cols)
    return res

def catMultipleMultiple(x, y, num = 8):
    mb, num_blockSize, rows, cols = list(y.shape)

    x = x.view(mb, num, -1, rows, cols)
    y = y.view(mb, num, -1, rows, cols)

    res = torch.cat((x ,y), dim = 2).view(mb, -1, rows, cols).view(mb, -1, rows, cols)
    return res

def squash(x):
    mb, _, rows, cols = x.shape
    x = x.view(mb, 2, -1, rows, cols)
    l = torch.sqrt((x * x).sum(dim = 1, keepdim = True) + 1)
    return (x / l).view(mb, -1, rows, cols)

def catReluMultipleMultiple(x, y, num = 8):
    mb, num_blockSize, rows, cols = list(y.shape)

    x = x.view(mb, num, -1, rows, cols)
    y = F.relu(y).view(mb, num, int(num_blockSize / num), rows, cols)

    res = torch.cat((x ,y), dim = 2).view(mb, -1, rows, cols)
    return res


def random_4_rotation(x):
    to_apply = {0: lambda x: x,
                1: rotate_c,
                2: lambda x: rotate_c(rotate_c(x)),
                3: rotate_cc}
    ind = np.random.randint(4)
    return to_apply[ind](x)


def random_8_flip_rotation(x):
    to_apply = {0: lambda x: x,
                1: rotate_c,
                2: lambda x: rotate_c(rotate_c(x)),
                3: rotate_cc,
                4: flip,
                5: lambda x: rotate_c(flip(x)),
                6: lambda x: rotate_c(rotate_c(flip(x))),
                7: lambda x: rotate_cc(flip(x))}
    ind = np.random.randint(8)
    return to_apply[ind](x)