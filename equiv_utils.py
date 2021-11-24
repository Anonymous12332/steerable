import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from equiv_aux_utils import rotate4, rotate_c_, flip_, flip, rotate_cc_


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class OctaveBatchNorm(nn.Module):
    def __init__(self, depth, num=8):
        super().__init__()
        self.num = num
        self.depth = depth
        self.bn = nn.BatchNorm2d(num_features=depth)

    def forward(self, x):
        mb, num_depth, rows, cols = list(x.shape)
        assert self.num * self.depth == num_depth
        x = x.view(mb, self.num, self.depth, rows, cols)
        x = x.transpose(1, 2).contiguous().view(mb, self.depth, self.num * rows, cols)
        x = self.bn(x)
        x = x.view(mb, self.depth, self.num, rows, cols).transpose(1, 2)
        x = x.contiguous().view(mb, self.num * self.depth, rows, cols)
        return x


class OctaveDrop(nn.Module):
    def __init__(self, pDrop, num=8):
        super().__init__()
        self.num = num
        self.pDrop = pDrop

    def forward(self, x):
        if not self.training:
            return x
        if self.pDrop < 0.001:
            return x
        mb, num_depth = x.shape[0], x.shape[1]
        shape = x.shape
        x = x.view(mb, self.num, int(num_depth / self.num), *shape[2:])
        active = x.new(mb, self.num, 1, *((1,) * len(shape[2:])))
        active[:] = 0
        indices = torch.randint(0, self.num, (mb,))
        for i in range(mb):
            active[i, indices[i]] = 1
        active = active * self.num
        # active.bernoulli_(1.0 - self.pDrop).mul_(1.0 / (1.0 - self.pDrop))
        x = x.mul(active)
        return x.view(*shape)


class OctaveColumnDrop(nn.Module):
    def __init__(self, pDrop, num=8):
        super().__init__()
        self.pDrop = pDrop
        self.num = num

    def forward(self, x):
        if not self.training:
            return x
        if self.pDrop < 0.001:
            return x
        mb, num_depth, rows, cols = x.shape
        x = x.view(mb, self.num, int(num_depth / self.num), rows, cols)
        active = x.new(mb, self.num, 1, rows, cols)
        active.bernoulli_(1.0 - self.pDrop).mul_(1.0 / (1.0 - self.pDrop))
        x = x.mul(active)
        return x.view(mb, num_depth, rows, cols)
