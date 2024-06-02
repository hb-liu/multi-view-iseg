import torch
import torch.nn as nn

class _Residual_Block(nn.Module):
    def __init__(self, inc, outc):
        super(_Residual_Block, self).__init__()
        midc = outc
        if inc != outc:
            self.conv_expand = nn.Conv3d(inc, outc, 1, bias=False)
        else:
            self.conv_expand = None
        self.conv1 = nn.Conv3d(inc, midc, 3, 1, 1)
        self.bn1 = nn.InstanceNorm3d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(midc, outc, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm3d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity = self.conv_expand(x)
        else:
            identity = x
        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity))
        return output

class Encoder(nn.Module):
    def __init__(self, inc, zdim):
        super().__init__()
        channels = (32, 64, 128, 256)
        self.zdim = zdim
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv3d(inc, cc, 5, 1, 2, bias=False),
            nn.InstanceNorm3d(cc),
            nn.LeakyReLU(0.2)
        )
        for ch in channels[1:]:
            self.main.append(_Residual_Block(cc, ch))
            self.main.append(nn.AvgPool3d(2))
            cc = ch
        self.main.append(_Residual_Block(cc, cc))
        self.fc = nn.Conv3d(cc, zdim, 3, 1, 1)

    def forward(self, x, with_embed=False):
        e0 = self.main[2](self.main[1](self.main[0](x)))
        e1 = self.main[4](self.main[3](e0))
        e2 = self.main[6](self.main[5](e1))
        e3 = self.main[8](self.main[7](e2))
        e4 = self.main[9](e3)
        y = self.fc(e4)
        if with_embed:
            return y, (e0, e1, e2)
        else:
            return y

class Decoder(nn.Module):
    def __init__(self, outc, zdim):
        super(Decoder, self).__init__()
        channels = (32, 64, 128, 256)
        cc = channels[-1]
        self.fc = nn.Sequential(
            nn.Conv3d(zdim, cc, 3, 1, 1),
            nn.ReLU(True),
        )
        self.main = nn.Sequential()
        for i, ch in enumerate(channels[::-1]):
            if i == 0:
                self.main.append(_Residual_Block(cc, ch))
            else:
                self.main.append(_Residual_Block(int(cc*1.5), ch))
            self.main.append(nn.ConvTranspose3d(ch, ch, 4, 2, 1))
            cc = ch
        self.main.append(_Residual_Block(cc, cc))
        self.main.append(nn.Conv3d(cc, outc, 5, 1, 2))
        self.main.append(nn.Sigmoid())

    def forward(self, z, e):
        y = self.fc(z)
        y1 = self.main[1](self.main[0](y))
        y2 = self.main[3](self.main[2](torch.cat((y1, e[2]), dim=1)))
        y3 = self.main[5](self.main[4](torch.cat((y2, e[1]), dim=1)))
        y4 = self.main[6](torch.cat((y3, e[0]), dim=1))
        y5 = self.main[8](y4)
        out = self.main[10](self.main[9](y5))
        return out

class ck(nn.Module):
    def __init__(self, inc, outc, stride, norm):
        super().__init__()
        self.conv = nn.Conv3d(inc, outc, 3, stride, 1)
        if norm:
            self.norm = nn.InstanceNorm3d(outc)
        else:
            self.norm = None
        self.relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.conv1 = ck(inc, 64, 2, False)
        self.conv2 = ck(64, 128, 2, True)
        self.conv3 = ck(128, 256, 2, True)
        self.conv4 = ck(256, 512, 1, True)
        self.conv5 = nn.Conv3d(512, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x