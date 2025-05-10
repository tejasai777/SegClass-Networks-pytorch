
'''
augmentations to transform train images for geenralisation of images which makes model effective to real world applications
'''
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class AlbumentationsTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        arr    = np.array(img)
        result = self.aug(image=arr)
        return result['image']

def get_train_transforms(image_size=(64,64)):
    h, w = image_size  # target image size
    aug = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.Resize(height=h, width=w),   #resizes all images to 64x64 
        A.Normalize(),
        ToTensorV2(),
    ])
    return AlbumentationsTransform(aug)

def get_test_transforms(image_size=(64,64)):
    h, w = image_size
    aug = A.Compose([
        A.Resize(height=h, width=w),    # for trst images resize normalize and converts to tensor 
        A.Normalize(),
        ToTensorV2(),
    ])
    return AlbumentationsTransform(aug)
