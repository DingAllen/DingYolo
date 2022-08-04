import torch
from torch import nn


class CBS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CBS, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activition = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activition(x)
        return x


class ELAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELAN, self).__init__()

        c = in_channels // 2
        self.cbs1 = CBS(in_channels, c, 1, 1)
        self.cbs2 = CBS(in_channels, c, 1, 1)
        self.cbs3 = CBS(c, c, 3, 1)
        self.cbs4 = CBS(c, c, 3, 1)
        self.cbs5 = CBS(c, c, 3, 1)
        self.cbs6 = CBS(c, c, 3, 1)
        self.cbs_all = CBS(c * 4, out_channels, 3, 1)

    def forward(self, x):
        x1 = self.cbs1(x)
        x2 = self.cbs2(x)
        x3 = self.cbs4(self.cbs3(x2))
        x4 = self.cbs6(self.cbs5(x3))
        x_all = torch.cat([x1, x2, x3, x4], dim=1)
        return self.cbs_all(x_all)


class MP1(nn.Module):
    def __init__(self, in_channels):
        super(MP1, self).__init__()
        c = in_channels // 2
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbs1 = CBS(in_channels, c, 1, 1)
        self.cbs2 = CBS(in_channels, c, 1, 1)
        self.cbs3 = CBS(c, c, 3, 2)

    def forward(self, x):
        x1 = self.cbs1(self.max_pool(x))
        x2 = self.cbs3(self.cbs2(x))
        return torch.cat([x1, x2], dim=1)


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.extractor = nn.Sequential(
            CBS(3, 32, 3, 1),
            CBS(32, 64, 3, 2),
            CBS(64, 64, 3, 1),
            CBS(64, 128, 3, 2),
            ELAN(128, 256)
        )
        self.calcu1 = nn.Sequential(
            MP1(256),
            ELAN(256, 512)
        )
        self.calcu2 = nn.Sequential(
            MP1(512),
            ELAN(512, 1024)
        )
        self.calcu3 = nn.Sequential(
            MP1(1024),
            ELAN(1024, 1024)
        )

    def forward(self, x):
        x = self.extractor(x)
        x1 = self.calcu1(x)
        x2 = self.calcu2(x1)
        x3 = self.calcu3(x2)
        return x1, x2, x3


if __name__ == '__main__':
    x = torch.randn((10, 3, 640, 640))
    backbone = Backbone()
    print(x.size())
    x3, x2, x1 = backbone(x)
    print(x3.size())
    print(x2.size())
    print(x1.size())
