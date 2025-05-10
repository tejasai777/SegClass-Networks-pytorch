
"""
Code Credits:

EfficientNetV2 Model  
- Source: 
(https://github.com/maciejbalawejder/Deep-Learning-Collection/tree/main/ConvNets/EfficientNetV2)  
- Original Paper: EfficientNetV2: Smaller Models and Faster Training by **Mingxing Tan and Quoc V. Le**  

"""

import torch
import torch.nn as nn
from torch import Tensor

class StochasticDepth(nn.Module):
    """
    Implements stochastic depth as in EfficientNetV2.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        survival_rate = 1.0 - self.p
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(mask_shape, dtype=x.dtype, device=x.device).bernoulli_(survival_rate)
        return x / survival_rate * mask


def get_config(name: str):
    """
    Returns block configuration strings for EfficientNetV2 variants S, M, L.
    """
    configs = {
        "S": [
            "r2_k3_s1_e1_i24_o24_c1",
            "r4_k3_s2_e4_i24_o48_c1",
            "r4_k3_s2_e4_i48_o64_c1",
            "r6_k3_s2_e4_i64_o128_c1",
            "r9_k3_s1_e6_i128_o160_c0",
            "r15_k3_s2_e6_i160_o256_c0"
        ],
        "M": [
            "r3_k3_s1_e1_i24_o24_c1",
            "r5_k3_s2_e4_i24_o48_c1",
            "r5_k3_s2_e4_i48_o64_c1",
            "r7_k3_s2_e4_i64_o128_c1",
            "r14_k3_s1_e6_i128_o160_c0",
            "r18_k3_s2_e6_i160_o256_c0"
        ],
        "L": [
            "r4_k3_s1_e1_i32_o32_c1",
            "r7_k3_s2_e4_i32_o64_c1",
            "r7_k3_s2_e4_i64_o96_c1",
            "r10_k3_s2_e4_i96_o192_c0",
            "r19_k3_s1_e6_i192_o224_c0",
            "r25_k3_s2_e6_i224_o384_c0",
            "r7_k3_s1_e6_i384_o640_c0"
        ]
    }
    if name not in configs:
        raise ValueError(f"Unknown EfficientNetV2 config: {name}")
    return configs[name]

class ConvBlock(nn.Module):
    """ Basic Conv-BN-SiLU/Identity block """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        act: bool = True,
        bias: bool = False
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

class SeBlock(nn.Module):
    """ Squeeze-and-Excitation block """
    def __init__(self, in_channels: int, squeeze_channels: int):
        super().__init__()
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        self.act = nn.SiLU(inplace=True)
        self.sig = nn.Sigmoid()
    def forward(self, x: Tensor) -> Tensor:
        f = self.globpool(x)
        f = self.act(self.fc1(f))
        f = self.sig(self.fc2(f))
        return x * f

class MBConv(nn.Module):
    """ Mobile inverted bottleneck block with optional SE and stochastic depth """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        exp_ratio: int,
        sd_prob: float
    ):
        super().__init__()
        hidden = in_channels * exp_ratio
        self.use_res = (in_channels == out_channels and stride == 1)
        layers = []
        if exp_ratio != 1:
            layers.append(ConvBlock(in_channels, hidden, 1, 1))
        layers.append(ConvBlock(hidden, hidden, kernel_size, stride, groups=hidden))
        layers.append(SeBlock(hidden, max(1, hidden//4)))
        layers.append(ConvBlock(hidden, out_channels, 1, 1, act=False))
        self.block = nn.Sequential(*layers)
        self.sd = StochasticDepth(sd_prob)
    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if self.use_res:
            out = self.sd(out)
            return x + out
        return out

class FusedMBConv(nn.Module):
    """ Fused MBConv block (no SE) """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        exp_ratio: int,
        sd_prob: float
    ):
        super().__init__()
        hidden = in_channels * exp_ratio
        self.use_res = (in_channels == out_channels and stride == 1)
        if exp_ratio != 1:
            self.block = nn.Sequential(
                ConvBlock(in_channels, hidden, kernel_size, stride),
                ConvBlock(hidden, out_channels, 1, 1, act=False)
            )
        else:
            self.block = ConvBlock(in_channels, out_channels, kernel_size, stride)
        self.sd = StochasticDepth(sd_prob)
    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if self.use_res:
            out = self.sd(out)
            return x + out
        return out

class EfficientNetV2(nn.Module):
    """ EfficientNetV2 for segmentation/classification within modular pipeline """
    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = False,
        config_name: str = 'L',
        sd_prob: float = 0.2,
        dropout_prob: float = 0.5
    ):
        super().__init__()
        config = get_config(config_name)
        # Stem
        first = config[0].split('_')
        in_ch = int(first[4][1:])
        self.stem = ConvBlock(3, in_ch, 3, 2)
        # Blocks
        total = sum(int(c.split('_')[0][1:]) for c in config)
        self.blocks = nn.ModuleList()
        blk_id = 0
        prev_ch = in_ch
        for stage in config:
            parts = stage.split('_')
            reps = int(parts[0][1:]); k=int(parts[1][1:]); s=int(parts[2][1:]); e=int(parts[3][1:]); i=int(parts[4][1:]); o=int(parts[5][1:]); fuse=parts[6]=='c1'
            for r in range(reps):
                blk_id += 1
                prob = sd_prob * blk_id / total
                stride = s if r==0 else 1
                if fuse:
                    self.blocks.append(FusedMBConv(prev_ch, o, k, stride, e, prob))
                else:
                    self.blocks.append(MBConv(prev_ch, o, k, stride, e, prob))
                prev_ch = o
        # Head
        self.head_conv = ConvBlock(prev_ch, 1280, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(1280, num_classes)
        )
    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)