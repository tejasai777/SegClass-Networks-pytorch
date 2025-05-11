from models.unet3plus import UNet3Plus
from models.unetpp import NestedUNet
from models.unet_vgg16 import vgg16bn_unet
from models.attunet import AttU_Net
from models.unet_mobilenet import UNet as MobileUNet

def build_seg_model(name, img_ch=3, num_classes=4):
    n = name.lower()
    if n == 'unet3plus':
        return UNet3Plus((img_ch, ), num_classes, deep_supervision=False, cgm=False)
    elif n in ['unet++', 'unetpp', 'nestedunet']:
        return NestedUNet(num_classes=num_classes, input_channels=img_ch, deep_supervision=False)
    elif n in ['unet_vgg16', 'vgg16']:
        return vgg16bn_unet(output_dim=num_classes, pretrained=False)
    elif n in ['attunet', 'attentionunet']:
        return AttU_Net(img_ch=img_ch, output_ch=num_classes)
    elif n in ['unet_mobilenet', 'mobilenet']:
        return MobileUNet(n_channels=img_ch, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
