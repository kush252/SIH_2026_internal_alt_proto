import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

class MaskGenerator:
    """
    Generates a random mask for SimMIM.
    Works on a grid of patches (e.g. 224//32 = 7x7 grid).
    """
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        # Grid size for the mask
        self.mask_grid_size = input_size // mask_patch_size
        # The number of model patches (4x4) inside one mask patch (32x32) is (32/4)^2 = 64
        
        self.num_mask_patches = self.mask_grid_size * self.mask_grid_size
        self.num_mask = int(self.num_mask_patches * self.mask_ratio)
        
    def __call__(self):
        # Generate 1D array of zeros and ones
        mask = np.hstack([
            np.ones(self.num_mask),
            np.zeros(self.num_mask_patches - self.num_mask)
        ])
        np.random.shuffle(mask)
        # Reshape to mask grid
        mask = mask.reshape((self.mask_grid_size, self.mask_grid_size))
        # Mask generated at 7x7. Later we upsample it to model patch size (56x56) or image size.
        return mask

class SimMIMTransform:
    def __init__(self, config):
        self.transform = T.Compose([
            T.RandomResizedCrop(config.DATA.image_size, scale=tuple(config.AUGMENTATION.scale_range)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # 90 degree rotations are great for aerial
            T.RandomApply([T.Lambda(lambda x: x.rotate(90))], p=0.5),
            T.ColorJitter(
                brightness=config.AUGMENTATION.color_jitter,
                contrast=config.AUGMENTATION.color_jitter,
                saturation=config.AUGMENTATION.color_jitter
            ),
            T.ToTensor(),
            # Standard ImageNet normalization since we'll use pretrained Swin
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.image_size,
            mask_patch_size=config.SIMMIM.mask_patch_size,
            model_patch_size=config.MODEL.patch_size,
            mask_ratio=config.SIMMIM.mask_ratio
        )
        
    def __call__(self, img):
        # img is a PIL Image
        img = self.transform(img)
        mask = self.mask_generator()
        return img, mask

class SimSiamTransform:
    """
    Generates two independently augmented views of the same image for SimSiam.
    """
    def __init__(self, config):
        self.transform = T.Compose([
            T.RandomResizedCrop(config.DATA.image_size, scale=tuple(config.AUGMENTATION.scale_range)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.Lambda(lambda x: x.rotate(90))], p=0.5),
            T.RandomApply([
                T.ColorJitter(
                    brightness=config.AUGMENTATION.color_jitter,
                    contrast=config.AUGMENTATION.color_jitter,
                    saturation=config.AUGMENTATION.color_jitter,
                    hue=0.1
                )
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            # Standard ImageNet normalization for ResNet50
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, img):
        # img is a PIL Image
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2

