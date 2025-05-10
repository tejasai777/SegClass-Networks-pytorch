
"""
Code Credits:

MobileNetV3 Model  
- Source Code: (https://github.com/maciejbalawejder/Deep-Learning-Collection/blob/main/ConvNets/MobileNetV3/mobilenetv3_pytorch.py)  
- Original Paper: Searching for MobileNetV3 by **Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, and Hartwig Adam**   
- https://arxiv.org/pdf/1905.02244
- **YouTube Video** (MobileNetV3 paper walkthrough): (https://youtu.be/0oqs-inp7sA?si=-D5CuitiBLebP3Vw)

"""
import torch
import torch.nn as nn
from torch import Tensor

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        act=nn.ReLU(),
        groups=1,
        bn=True,
        bias=False
    ):
        super().__init__()
        padding = kernel_size // 2
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.c(x)))

class SeBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        C = in_channels
        r = C // 4
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(C, r, bias=False)
        self.fc2 = nn.Linear(r, C, bias=False)
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        f = self.globpool(x)
        f = torch.flatten(f,1)
        f = self.relu(self.fc1(f))
        f = self.hsigmoid(self.fc2(f))
        f = f[:,:,None,None]
        return x * f

class BNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        exp_size: int,
        se: bool,
        act: torch.nn.modules.activation,
        stride: int
    ):
        super().__init__()
        self.add = (in_channels == out_channels and stride == 1)
        self.block = nn.Sequential(
            ConvBlock(in_channels, exp_size, 1, 1, act),
            ConvBlock(exp_size, exp_size, kernel_size, stride, act, exp_size),
            SeBlock(exp_size) if se else nn.Identity(),
            ConvBlock(exp_size, out_channels, 1, 1, act=nn.Identity())
        )

    def forward(self, x: Tensor) -> Tensor:
        res = self.block(x)
        return res + x if self.add else res

class MobileNetV3(nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = False):
        super().__init__()
        config = self.config("large")
        self.conv = ConvBlock(3, 16, 3, 2, nn.Hardswish())
        self.blocks = nn.ModuleList([
            BNeck(in_c, out_c, k, exp, se, nl, s)
            for k, exp, in_c, out_c, se, nl, s in config
        ])
        last_out, last_exp = config[-1][3], config[-1][1]
        out_channels = 1280
        self.classifier = nn.Sequential(
            ConvBlock(last_out, last_exp, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d((1,1)),
            ConvBlock(last_exp, out_channels, 1, 1, nn.Hardswish(), bn=False, bias=True),
            nn.Dropout(0.8),
            nn.Conv2d(out_channels, num_classes, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def config(self, name):
        HE, RE = nn.Hardswish(), nn.ReLU()
        large = [
            [3, 16, 16, 16, False, RE, 1],
            [3, 64, 16, 24, False, RE, 2],
            [3, 72, 24, 24, False, RE, 1],
            [5, 72, 24, 40, True, RE, 2],
            [5, 120, 40, 40, True, RE, 1],
            [5, 120, 40, 40, True, RE, 1],
            [3, 240, 40, 80, False, HE, 2],
            [3, 200, 80, 80, False, HE, 1],
            [3, 184, 80, 80, False, HE, 1],
            [3, 184, 80, 80, False, HE, 1],
            [3, 480, 80, 112, True, HE, 1],
            [3, 672, 112, 112, True, HE, 1],
            [5, 672, 112, 160, True, HE, 2],
            [5, 960, 160, 160, True, HE, 1],
            [5, 960, 160, 160, True, HE, 1]
        ]
        small = [
            [3, 16, 16, 16, True, RE, 2],
            [3, 72, 16, 24, False, RE, 2],
            [3, 88, 24, 24, False, RE, 1],
            [5, 96, 24, 40, True, HE, 2],
            [5, 240, 40, 40, True, HE, 1],
            [5, 240, 40, 40, True, HE, 1],
            [5, 120, 40, 48, True, HE, 1],
            [5, 144, 48, 48, True, HE, 1],
            [5, 288, 48, 96, True, HE, 2],
            [5, 576, 96, 96, True, HE, 1],
            [5, 576, 96, 96, True, HE, 1]
        ]
        if name == "large": return large
        if name == "small": return small