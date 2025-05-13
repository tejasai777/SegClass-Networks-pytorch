from models.unet3plus import UNet3Plus
from models.unetpp import NestedUNet
from models.unet_vgg16 import vgg16bn_unet
from models.attunet import AttU_Net
from models.unet_mobilenet import UNet as MobileUNet
from models.ResUnet import ResUnet
from models.Unet import UNET
import segmentation_models_pytorch as smp
from models.segformer import BiSeNetMulticlass
from models.stdc import STDC
from models.ddr import DDRNet
from models.litehrnet import LiteHRNet


def build_seg_model(name, img_ch=3, num_classes=4, img_size=640):
    n = name.lower()
    if n == 'unet3plus':
        return UNet3Plus((img_ch, img_size, img_size), num_classes, deep_supervision=False, cgm=False)
    elif n in ['unet++', 'unetpp', 'nestedunet']:
        return NestedUNet(num_classes=num_classes, input_channels=img_ch, deep_supervision=False)
    elif n in ['unet_vgg16', 'vgg16']:
        return vgg16bn_unet(output_dim=num_classes, pretrained=False)
    elif n in ['attunet', 'attentionunet']:
        return AttU_Net(img_ch=img_ch, output_ch=num_classes)
    elif n in ['unet_mobilenet', 'mobilenet']:
        return MobileUNet(n_channels=img_ch, num_classes=num_classes)
    elif n in ['unet', 'UNET','Unet']:
        return UNET(in_channels = 3, out_channels = num_classes)
    elif n in ['resunet', 'ResUnet']:
        return ResUnet(channel=3,out_channel=num_classes)
    elif n == 'stdc':
        return STDC(num_class=num_classes, n_channel=img_ch, use_aux=False, use_detail_head=False)
    elif n == 'ddrnet':
      return DDRNet(num_class=num_classes, n_channel=img_ch, arch_type='DDRNet-23-slim', use_aux=False)
    elif n in ['litehrnet', 'lite-hrnet-30','litehrnet30']:
      return LiteHRNet(num_class=num_classes, n_channel=img_ch, arch_type='litehrnet30')
    elif n == 'deeplabv3plus_resnet34':
        return smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=img_ch,
            classes=num_classes
        )
    elif n == 'deeplabv3plus_mit_b0':
        return smp.DeepLabV3Plus(
            encoder_name="mit_b0",
            encoder_weights="imagenet",
            in_channels=img_ch,
            classes=num_classes
        )
    elif n == 'bisenetmulticlass':
      return BiSeNetMulticlass(num_class=num_classes, n_channel=img_ch, act_type='relu', use_aux=False)
     
    else:
        raise ValueError(f"Unknown model: {name}")

