import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20,
                           border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                      hue=0.1, p=0.3),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])

def get_val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])
