
"""
Code Credits:

-Extracted **PCBClassNet** for multi-class classification and adapted it in PyTorch.  
- Source : (https://github.com/CandleLabAI/PCBSegClassNet/tree/main)  
- Published Paper: (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4241188)  
- Authors: **Dhruv Makwana, Sai Chandra Teja R, and Sparsh Mittal**—researchers specializing in deep learning and computer vision.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x) + x

class EDSREncoder(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(num_features, num_features, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return self.tail(x)

class PCBClassifier(nn.Module):
    def __init__(self, num_classes=6, pretrained: bool = False):
        super().__init__()
        self.encoder = EDSREncoder(in_channels=3, num_features=64, num_blocks=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
