import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, inc, outc, stride) -> None:
        super().__init__()
        midc = inc

        self.conv1 = nn.Conv3d(inc, midc, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(midc)
        self.conv2 = nn.Conv3d(midc, midc, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(midc)
        self.conv3 = nn.Conv3d(midc, outc, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(outc)

        self.relu = nn.ReLU(inplace=True)

        self.skip = nn.Sequential(
            nn.Conv3d(inc, outc, 1, stride, bias=False),
            nn.BatchNorm3d(outc)
        )

    def forward(self, xin):
        ide = self.skip(xin)

        x = self.conv1(xin)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = x + ide
        x = self.relu(x)

        return x
    
class SEGNET(nn.Module):
    def __init__(self, inc, outc, midc=16, stages=4):
        super().__init__()
        self.inc, self.stages = inc, stages
        # input convolution
        self.inconv = BasicBlock(inc, midc, 1)
        # encoder
        self.enc = nn.ModuleList()
        stagec = midc
        for k in range(stages):
            self.enc.append(BasicBlock(stagec, stagec*2, 2))
            stagec *= 2
        # decoder
        self.dec = nn.ModuleList()
        for k in range(stages):
            self.dec.append(nn.ModuleList([
                nn.ConvTranspose3d(stagec, stagec//2, 2, 2),
                BasicBlock(stagec, stagec//2, 1)
            ]))
            stagec //= 2
        # output
        self.convout = nn.Sequential(nn.Conv3d(stagec, outc, 1, 1, 0), nn.BatchNorm3d(outc))
    def forward(self, x):
        x = self.inconv(x)
        # encoder
        feas = [x]
        for layer in self.enc:
            x = layer(x)
            feas.append(x)
        # decoder
        fea = feas[-1]
        for i, (up, merge) in enumerate(self.dec):
            fea = up(fea)
            fea = merge(torch.cat([fea, feas[-2-i]], 1))
        # output
        out = self.convout(fea)
        return out