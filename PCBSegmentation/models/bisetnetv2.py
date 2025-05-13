"""
Code credits:
  -Paper:      BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation
  -Url:        https://arxiv.org/abs/2004.02147
  -github: Zh320- Real-time Semantic Segmentation in PyTorch- https://github.com/zh320/realtime-semantic-segmentation-pyt
  -Date:       2023/09/03
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import (
    conv3x3, conv1x1,
    DWConvBNAct, PWConvBNAct,
    ConvBNAct, Activation, SegHead
)
from models.model_registry import register_model, aux_models

@register_model(aux_models)
class BiSeNetMulticlass(nn.Module):
    """
    BiSeNet V2 variant for multiclass segmentation.
    Registered under key 'bisenetmulticlass'.
    """
    def __init__(self, num_class=4, n_channel=3, act_type='relu', use_aux=False):
        super().__init__()
        self.use_aux = use_aux
        self.detail_branch = DetailBranch(n_channel, 128, act_type)
        self.semantic_branch = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer = BilateralGuidedAggregationLayer(128, 128, act_type)
        self.seg_head = SegHead(128, num_class, act_type)

    def forward(self, x, is_training=False):
        size = x.size()[2:]
        x_d = self.detail_branch(x)
        if self.use_aux:
            x_s, aux2, aux3, aux4, aux5 = self.semantic_branch(x)
        else:
            x_s = self.semantic_branch(x)
        x = self.bga_layer(x_d, x_s)
        x = self.seg_head(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)

        if self.use_aux and is_training:
            return x, (aux2, aux3, aux4, aux5)
        return x


class DetailBranch(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__(
            ConvBNAct(in_channels, 64, 3, 2, act_type=act_type),
            ConvBNAct(64, 64, 3, 1, act_type=act_type),
            ConvBNAct(64, 64, 3, 2, act_type=act_type),
            ConvBNAct(64, 64, 3, 1, act_type=act_type),
            ConvBNAct(64, 128, 3, 1, act_type=act_type),
            ConvBNAct(128, 128, 3, 2, act_type=act_type),
            ConvBNAct(128, 128, 3, 1, act_type=act_type),
            ConvBNAct(128, out_channels, 3, 1, act_type=act_type)
        )


class SemanticBranch(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, act_type='relu', use_aux=False):
        super().__init__()
        self.use_aux = use_aux
        self.stage1to2 = StemBlock(in_channels, 16, act_type)
        self.stage3 = nn.Sequential(
            GatherExpansionLayer(16, 32, 2, act_type),
            GatherExpansionLayer(32, 32, 1, act_type),
        )
        self.stage4 = nn.Sequential(
            GatherExpansionLayer(32, 64, 2, act_type),
            GatherExpansionLayer(64, 64, 1, act_type),
        )
        self.stage5_1to4 = nn.Sequential(
            GatherExpansionLayer(64, 128, 2, act_type),
            GatherExpansionLayer(128, 128, 1, act_type),
            GatherExpansionLayer(128, 128, 1, act_type),
            GatherExpansionLayer(128, 128, 1, act_type),
        )
        self.stage5_5 = ContextEmbeddingBlock(128, out_channels, act_type)

        if self.use_aux:
            self.seg_head2 = SegHead(16, num_class, act_type)
            self.seg_head3 = SegHead(32, num_class, act_type)
            self.seg_head4 = SegHead(64, num_class, act_type)
            self.seg_head5 = SegHead(128, num_class, act_type)

    def forward(self, x):
        x = self.stage1to2(x)
        if self.use_aux:
            aux2 = self.seg_head2(x)

        x = self.stage3(x)
        if self.use_aux:
            aux3 = self.seg_head3(x)

        x = self.stage4(x)
        if self.use_aux:
            aux4 = self.seg_head4(x)

        x = self.stage5_1to4(x)
        if self.use_aux:
            aux5 = self.seg_head5(x)

        x = self.stage5_5(x)

        if self.use_aux:
            return x, aux2, aux3, aux4, aux5
        return x


class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.conv_init = ConvBNAct(in_channels, out_channels, 3, 2, act_type=act_type)
        self.left_branch = nn.Sequential(
            ConvBNAct(out_channels, out_channels//2, 1, act_type=act_type),
            ConvBNAct(out_channels//2, out_channels, 3, 2, act_type=act_type)
        )
        self.right_branch = nn.MaxPool2d(3, 2, 1)
        self.conv_last = ConvBNAct(out_channels*2, out_channels, 3, 1, act_type=act_type)

    def forward(self, x):
        x = self.conv_init(x)
        x_left = self.left_branch(x)
        x_right = self.right_branch(x)
        x = torch.cat([x_left, x_right], dim=1)
        return self.conv_last(x)


class GatherExpansionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type='relu', expand_ratio=6):
        super().__init__()
        self.stride = stride
        hid_channels = int(round(in_channels * expand_ratio))

        layers = [ConvBNAct(in_channels, in_channels, 3, act_type=act_type)]
        if stride == 2:
            layers.extend([
                DWConvBNAct(in_channels, hid_channels, 3, 2, act_type='none'),
                DWConvBNAct(hid_channels, hid_channels, 3, 1, act_type='none')
            ])
            self.right_branch = nn.Sequential(
                DWConvBNAct(in_channels, in_channels, 3, 2, act_type='none'),
                PWConvBNAct(in_channels, out_channels, act_type='none')
            )
        else:
            layers.append(DWConvBNAct(in_channels, hid_channels, 3, 1, act_type='none'))
        layers.append(PWConvBNAct(hid_channels, out_channels, act_type='none'))
        self.left_branch = nn.Sequential(*layers)
        self.act = Activation(act_type)

    def forward(self, x):
        res = self.left_branch(x)
        if self.stride == 2:
            res = self.right_branch(x) + res
        else:
            res = x + res
        return self.act(res)


class ContextEmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(in_channels)
        )
        self.conv_mid = ConvBNAct(in_channels, in_channels, 1, act_type=act_type)
        self.conv_last = conv3x3(in_channels, out_channels)

    def forward(self, x):
        res = self.pool(x)
        res = self.conv_mid(res)
        return self.conv_last(res + x)


class BilateralGuidedAggregationLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.detail_high = nn.Sequential(
            DWConvBNAct(in_channels, in_channels, 3, act_type=act_type),
            conv1x1(in_channels, in_channels)
        )
        self.detail_low = nn.Sequential(
            DWConvBNAct(in_channels, in_channels, 3, 2, act_type=act_type),
            nn.AvgPool2d(3, 2, 1)
        )
        self.semantic_high = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3, act_type=act_type),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )
        self.semantic_low = nn.Sequential(
            DWConvBNAct(in_channels, in_channels, 3, act_type=act_type),
            conv1x1(in_channels, in_channels),
            nn.Sigmoid()
        )
        self.conv_last = ConvBNAct(in_channels, out_channels, 3, act_type=act_type)

    def forward(self, x_d, x_s):
        dh = self.detail_high(x_d)
        dl = self.detail_low(x_d)
        sh = self.semantic_high(x_s)
        sl = self.semantic_low(x_s)
        x = dh * sh + F.interpolate(dl * sl, dh.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_last(x)
