'''
Code Credits:

Adapted **ConvNeXtTiny** for multi-class classification in PyTorch.  

- **GitHub Repository** (Implementation): (https://github.com/facebookresearch/ConvNeXt)  
- **Torchvision Model** (ConvNeXtTiny): (https://docs.pytorch.org/vision/stable/models/convnext.html)

'''

import torch.nn as nn
from torchvision.models import convnext_tiny

class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes: int=6, pretrained: bool=False):
        super().__init__()
        self.model = convnext_tiny(pretrained=pretrained)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
