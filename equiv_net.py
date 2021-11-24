import torch
import torch.nn as nn
import torch.nn.functional as F
from equiv_utils import OctaveColumnDrop, OctaveDrop, OctaveBatchNorm, Identity
from flip_utils import FlipTensorToGroupConv, FlipGroupToGroupConv
from rotation_utils import RotationTensorToGroupConv, RotationGroupToGroupConv
from flip_rotation_utils import FlipRotationTensorToGroupConv, FlipRotationGroupToGroupConv, SobelGroupConv, \
    SquareGroupConv, LineGroupConv
from equiv_aux_utils import catReluSingleMultiple, catReluMultipleMultiple, averageMultiple, \
    maxMultiple, sumOfSquaresMultiple
from steerable_utils import SteerableConv


def load_checkpoint(filepath):
    '''
    Function loads pretrained model from *.pth file
    '''
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = torch.load(filepath, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def save_checkpoint(model, filepath):
    checkpoint = {'model': model,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filepath)


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.netName = "Base Net"

    def numParams(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params

    def numTrainableParams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def printNet(self):
        print(self.netName + " will be trained!")
        print("Number of parameters: ", self.numParams())
        print("Number of trainable parameters: ", self.numTrainableParams())

    def getLoss(self, reduction="mean", weight=None):
        return nn.CrossEntropyLoss(reduction=reduction, weight=weight)

    def convertLabels(self, labels):
        return labels


class EquivariantNet(BaseNet):
    def __init__(self, numBlocks, blockSize, equiv_level='flip_rotation', pPoolDrop=[0, 0, 0], pOctaveDrop=0,
                 use_reduction_before=False,
                 reduction="average", k_pool_size=3, numPoolings=3, initDepth=3, numClasses=10, use_bn=True, tag=None,
                 use_reduction_conv=False, loss_type='cosine'):
        super().__init__()
        # assert equiv_level in ['flip', 'rotation', 'flip_rotation']
        self.num_tensors = {'none': 1, 'flip': 2, 'rotation': 4, 'flip_rotation': 8}[equiv_level]
        assert len(pPoolDrop) == numPoolings
        self.netName = f"{equiv_level} equivariant net"
        self.numBlocks, self.blockSize, self.numPoolings = numBlocks, blockSize, numPoolings
        self.tag = tag
        self.use_reduction_conv = use_reduction_conv
        self.use_reduction_before = use_reduction_before

        self.convs3_3, self.convs1d, self.bns_1, self.bns_3, self.poolDrop, self.pools = \
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        for _ in range(numPoolings - 1):
            self.pools.append(nn.AvgPool2d(2))
        self.pools.append(nn.AdaptiveAvgPool2d(1))

        for layer in range(numPoolings):
            self.convs1d.append(nn.ModuleList())
            self.bns_1.append(nn.ModuleList())
            self.convs3_3.append(nn.ModuleList())
            self.bns_3.append(nn.ModuleList())
            self.poolDrop.append(OctaveColumnDrop(pPoolDrop[layer], num=self.num_tensors))
        im_to_set_conv = {'none': nn.Conv2d,
                          'flip': FlipTensorToGroupConv,
                          'rotation': RotationTensorToGroupConv,
                          'flip_rotation': FlipRotationTensorToGroupConv}[equiv_level]
        # set_to_set_1dconv = {'flip': PairToPairEfficientConv1d,
        #                      'rotation': QuadToQuadEfficientConv1d,
        #                      'flip_rotation': OctaveToOctaveEfficientConv1d}[equiv_level]
        set_to_set_conv = {'none': nn.Conv2d,
                           'flip': FlipGroupToGroupConv,
                           'rotation': RotationGroupToGroupConv,
                           'flip_rotation': FlipRotationGroupToGroupConv}[equiv_level]
        size = initDepth
        self.initConv = im_to_set_conv(size, blockSize, 3, padding=1)
        size += blockSize

        bn_layer = OctaveBatchNorm if use_bn else Identity

        self.loss_type = loss_type

        for layer in range(numPoolings):
            for block in range(numBlocks):
                self.convs1d[layer].append(set_to_set_conv(size, blockSize, 1))
                self.bns_1[layer].append(bn_layer(blockSize, num=self.num_tensors))
                self.convs3_3[layer].append(set_to_set_conv(blockSize, blockSize, 3, padding=1))
                self.bns_3[layer].append(bn_layer(blockSize, num=self.num_tensors))
                size += blockSize

        if self.use_reduction_before:
            self.reduction_before = set_to_set_conv(size, blockSize, 1)
            size = blockSize

        self.octavePool = {"average": averageMultiple,
                           "max": maxMultiple,
                           # "square": sumOfSquaresMultiple,
                           "sobel": SobelGroupConv(in_channels=size, kernel_size=k_pool_size, padding=0),
                           "square": SquareGroupConv(in_channels=size, kernel_size=k_pool_size, padding=0),
                           "line": LineGroupConv(in_channels=size, kernel_size=k_pool_size, padding=0)}[reduction]
        # self.octavePool = self.orientationPool[reduction]

        if use_reduction_conv:
            self.reduction_conv = nn.Conv2d(size, blockSize, 1)
            size = blockSize

        if self.tag is not None:
            self.steerable_conv = SteerableConv(self.tag, size)
        self.octDrop = OctaveDrop(pOctaveDrop, num=self.num_tensors)

        self.fc = nn.Linear(size, numClasses)

    def forward(self, x):
        y = self.initConv(x)
        x = catReluSingleMultiple(x, y, num=self.num_tensors)

        for layer in range(self.numPoolings):
            for block in range(self.numBlocks):
                y = F.relu(self.bns_1[layer][block](self.convs1d[layer][block](x)))
                y = self.bns_3[layer][block](self.convs3_3[layer][block](y))
                x = catReluMultipleMultiple(x, y, num=self.num_tensors)
            if layer < self.numPoolings - 1:
                x = self.poolDrop[layer](x)
                x = self.pools[layer](x)
        # print("After last pool: ", x.shape)
        if self.use_reduction_before:
            x = self.reduction_before(x)
        # x = x.view(*x.shape[:2])
        x = self.octDrop(x)
        x = self.octavePool(x, num=self.num_tensors)

        if self.use_reduction_conv:
            x = self.reduction_conv(x)

        if self.tag is not None:
            x = self.steerable_conv(x)

        x = self.poolDrop[layer](x)
        x = self.pools[layer](x)

        x = x.view(*x.shape[:2])

        if self.tag is None:
            x = self.fc(x)
        return x

    def getLoss(self, reduction="mean", weight=None):
        assert self.loss_type in ['cross_entropy', 'mse', 'cosine']
        if self.loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss(reduction=reduction, weight=weight)

        if self.loss_type == 'mse':
            return nn.MSELoss(reduction=reduction)

        if reduction == 'mean':
            return lambda x, y: 0.5 * (1 - nn.CosineSimilarity()(x, y)).mean()
        else:
            return lambda x, y: 0.5 * (1 - nn.CosineSimilarity()(x, y)).sum()
