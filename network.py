from models.dinpnet import DInPNet
from models.efficientnetv2 import EfficientNetV2
from models.mobilenetv3 import MobileNetV3
from models.pcbclassifier import PCBClassifier
from models.convnext_tiny import ConvNeXtTiny

def build_model(name: str, num_classes: int, pretrained: bool=False, config_name='L'):
    mapping = {
        'dinpnet': DInPNet,
        'efficientnetv2': EfficientNetV2,
        'mobilenetv3': MobileNetV3,
        'pcbclassifier': PCBClassifier,
        'convnext_tiny': ConvNeXtTiny,
    }
    if name == 'efficientnetv2':
        return mapping[name](
            num_classes=num_classes,
            pretrained=pretrained,
            config_name=config_name
        )
    else:
        return mapping[name](
            num_classes=num_classes,
            pretrained=pretrained
        )
    if name not in mapping:
        raise ValueError(f"Unknown model: {name}")
    return mapping[name](num_classes=num_classes, pretrained=pretrained)
