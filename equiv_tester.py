from equiv_aux_utils import rotate4, flip_
import numpy as np
import torch


def testForSymmetryRotationInvariance(net, shape, numClasses):
    eps = 0.000001
    with torch.no_grad():
        x = torch.randn(1, *shape)
        if torch.cuda.is_available():
            x = x.cuda()
        net.eval()
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        l1 = list(rotate4(x))
        l2 = list(flip_(*l1))
        l = l1 + l2
        outs = [net(inp) for inp in l]
        if numClasses <= 10:
            for i in range(8):
                print(np.array(outs[i].tolist()))

        flipInv = (torch.abs(outs[0] - outs[4]) + torch.abs(outs[1] - outs[5]) + \
                   torch.abs(outs[2] - outs[6]) + torch.abs(outs[3] - outs[7]))

        rotInv = (torch.abs(outs[0] - outs[1]) + torch.abs(outs[1] - outs[2]) + \
                  torch.abs(outs[2] - outs[3]) + torch.abs(outs[4] - outs[5]) + \
                  torch.abs(outs[5] - outs[6]) + torch.abs(outs[6] - outs[7]))

        flipInv = bool((flipInv.mean() < eps).item())
        rotInv = bool((rotInv.mean() < eps).item())
        print("Flip invariant" if flipInv else "Not flip invariant")
        print("Rotationally invariant" if rotInv else "Not rotationally invariant")
        return flipInv, rotInv

def testForAdditionInvariance(net):
    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32)
        randCh = torch.randn(1, 3, 1, 1)
        randHor = torch.randn(1, 3, 1, 32)
        randVert = torch.randn(1, 3, 32, 1)
        randDiag = torch.randn(3)
        if (torch.cuda.is_available()):
            x = x.cuda()
            randCh = randCh.cuda()
            randHor = randHor.cuda()
            randVert = randVert.cuda()
            randDiag = randDiag.cuda()

        net.eval()
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

        xCh = x + randCh
        xHor = x + randHor
        xVert = x + randVert
        xDiag = x.clone()
        xDiag[0, torch.arange(3).view(3,1), torch.arange(32), torch.arange(32)] += randDiag.view(3,1)

        l = [x, xCh, xHor, xVert, xDiag]
        outs = [net(inp) for inp in l]
        modes = ["Plain", "Channel", "Horizontal", "Vertical", "Diagonal"]
        for out, mode in zip(outs, modes):
                print(mode, ": ", np.array(out.tolist()))
