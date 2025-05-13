"""
Code credits:
 -Paper:      Rethinking BiSeNet For Real-time Semantic Segmentation
 -Url:        https://arxiv.org/abs/2104.13188v1
 -github: Zh320- Real-time Semantic Segmentation in PyTorch- https://github.com/zh320/realtime-semantic-segmentation-pyt
 -Date:       2023/07/29
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, SegHead
from .bisenetv1 import AttentionRefinementModule, FeatureFusionModule
from .model_registry import register_model, aux_models, detail_head_models

@register_model(aux_models, detail_head_models)
class STDC(nn.Module):
    """
    STDC: Rethinking BiSeNet for Real-time Semantic Segmentation (2021).
    Supports auxiliary or detail heads.
    Registered under keys: 'stdc'.
    """
    def __init__(
        self,
        num_class=4,
        n_channel=3,
        encoder_type='stdc1',
        use_detail_head=False,
        use_aux=False,
        act_type='relu'
    ):
        super().__init__()
        repeat_times_hub = {'stdc1': [1,1,1], 'stdc2': [3,4,2]}
        if encoder_type not in repeat_times_hub:
            raise ValueError(f'Unsupported encoder type: {encoder_type}')
        repeat_times = repeat_times_hub[encoder_type]
        assert not (use_detail_head and use_aux), 'Use either detail head or aux head, not both.'
        self.use_detail_head = use_detail_head
        self.use_aux = use_aux

        # Stem
        self.stage1 = ConvBNAct(n_channel, 32, 3, 2)
        self.stage2 = ConvBNAct(32, 64, 3, 2)
        # Feature encoder
        self.stage3 = self._make_stage(64, 256, repeat_times[0], act_type)
        self.stage4 = self._make_stage(256, 512, repeat_times[1], act_type)
        self.stage5 = self._make_stage(512, 1024, repeat_times[2], act_type)

        # Auxiliary segmentation heads
        if self.use_aux:
            self.aux_head3 = SegHead(256, num_class, act_type)
            self.aux_head4 = SegHead(512, num_class, act_type)
            self.aux_head5 = SegHead(1024, num_class, act_type)

        # BiSeNet2 refinement modules
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.arm4 = AttentionRefinementModule(512)
        self.arm5 = AttentionRefinementModule(1024)
        self.conv4 = conv1x1(512, 256)
        self.conv5 = conv1x1(1024, 256)
        self.ffm = FeatureFusionModule(256+256, 128, act_type)

        # Final heads
        self.seg_head = SegHead(128, num_class, act_type)
        if self.use_detail_head:
            self.detail_head = SegHead(256, 1, act_type)

    def _make_stage(self, in_ch, out_ch, reps, act_type):
        layers = [STDCModule(in_ch, out_ch, stride=2, act_type=act_type)]
        for _ in range(reps):
            layers.append(STDCModule(out_ch, out_ch, stride=1, act_type=act_type))
        return nn.Sequential(*layers)

    def forward(self, x, is_training=False):
        h, w = x.size(2), x.size(3)
        x = self.stage1(x)
        x = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        # collect aux outputs
        auxs = []
        if self.use_aux:
            auxs = [
                self.aux_head3(x3),
                self.aux_head4(x4),
                self.aux_head5(x5)
            ]

        # refinement path
        x5_pool = self.pool(x5)
        x5 = x5_pool + self.arm5(x5)
        x5 = self.conv5(x5)
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)

        x4r = self.arm4(x4)
        x4r = self.conv4(x4r)
        x4r = x4r + x5
        x4r = F.interpolate(x4r, scale_factor=2, mode='bilinear', align_corners=True)

        # feature fusion
        fused = self.ffm(x4r, x3)
        out = self.seg_head(fused)
        out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=True)

        if self.use_detail_head and is_training:
            detail = self.detail_head(x3)
            return out, detail
        if self.use_aux and is_training:
            return out, tuple(auxs)
        return out


class STDCModule(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act_type):
        super().__init__()
        assert out_ch % 8 == 0, 'Output channels must be divisible by 8.'
        self.stride = stride
        self.block1 = ConvBNAct(in_ch, out_ch//2, 1)
        self.block2 = ConvBNAct(out_ch//2, out_ch//4, 3, stride)
        self.block3 = ConvBNAct(out_ch//4, out_ch//8, 3)
        self.block4 = ConvBNAct(out_ch//8, out_ch//8, 3)
        if stride == 2:
            self.pool = nn.AvgPool2d(3, 2, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        if self.stride == 2:
            x1 = self.pool(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.act(out)
