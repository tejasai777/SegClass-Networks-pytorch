"""
Code Credits:

Mantravadi et al., 2023  
- Adapted dlnpnet network for multi class classification 
- Dilated Involutional Pyramid Network (DInPNet)**: A Novel Model for Printed Circuit Board (PCB) Components Classification  
- Presented at**: 24th International Symposium on Quality Electronic Design (ISQED)  
- GitHub Repository** (https://github.com/CandleLabAI/DInPNet-PCB-Component-Classification)  
- **Published Paper** (IEEE Xplore): (https://ieeexplore.ieee.org/document/10129388)
"""

import torch
import torch.nn as nn
from involution import Involution2d

class IBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = (kSize - 1) // 2
        self.conv = Involution2d(in_channels=nIn, out_channels=nOut,
                                 kernel_size=(kSize,kSize), stride=stride,
                                 padding=(padding,padding), bias=False)
        self.batchnorm = nn.BatchNorm2d(nOut, eps=1e-3)
        self.activation = nn.PReLU(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.activation(x)

class IBRDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1):
        super().__init__()
        padding = ((kSize - 1) // 2) * dilation
        self.conv = Involution2d(in_channels=nIn, out_channels=nOut,
                                 kernel_size=(kSize,kSize), stride=stride,
                                 padding=(padding,padding), bias=False,
                                 dilation=dilation)
        self.batchnorm = nn.BatchNorm2d(nOut, eps=1e-3)
        self.activation = nn.PReLU(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.activation(x)

class DInPBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = nOut // 5
        n1 = nOut - 4*n
        self.invo1 = IBR(nIn, n, 3, stride=2)
        self.dilated1 = IBRDilated(n, n1, 3)
        self.dilated2 = IBRDilated(n, n, 3, dilation=2)
        self.dilated4 = IBRDilated(n, n, 3, dilation=4)
        self.dilated8 = IBRDilated(n, n, 3, dilation=8)
        self.dilated16= IBRDilated(n, n, 3, dilation=16)
        self.batchnorm = nn.BatchNorm2d(nOut, eps=1e-3)
        self.activation = nn.PReLU(nOut)

    def forward(self, x):
        o1 = self.invo1(x)
        d1 = self.dilated1(o1)
        d2 = self.dilated2(o1)
        d4 = self.dilated4(o1)
        d8 = self.dilated8(o1)
        d16= self.dilated16(o1)
        out = torch.cat([d1, d2, d4, d8, d16], dim=1)
        return self.activation(self.batchnorm(out))

class DInPNet(nn.Module):
    def __init__(self, num_classes: int=6, pretrained: bool = False):
        super().__init__()
        self.invo1 = Involution2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(16, eps=1e-3)
        self.act   = nn.PReLU()
        self.pool  = nn.MaxPool2d(2)
        self.block = DInPBlock(16, 32)
        self.invo2 = Involution2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64, eps=1e-3)
        self.fc1   = nn.Linear(64*8*8, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.act(self.bn1(self.invo1(x)))
        x = self.pool(x)
        x = self.block(x)
        x = self.pool(x)
        x = self.act(self.bn2(self.invo2(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        return self.fc2(x)
