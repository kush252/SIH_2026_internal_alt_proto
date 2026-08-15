import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class SvamitvaDataset(Dataset):
    """
    SVAMITVA Phase 2 supervised segmentation dataset.
    Loads paired Image and Mask, applies identical augmentations,
    and returns separated binary masks for each task.
    """
    def __init__(self, data_dir, file_list, target_builder, config, is_train=True):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "Images"
        self.masks_dir = self.data_dir / "Masks"
        
        self.file_list = file_list
        self.target_builder = target_builder
        self.config = config
        self.is_train = is_train
        
        self.image_size = config.DATA.image_size
        self.aug_cfg = config.AUGMENTATION
        
    def __len__(self):
        return len(self.file_list)
        
    def apply_transform(self, img, mask):
        # Resize to fixed training resolution
        img = TF.resize(img, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
        # Use NEAREST for mask to avoid interpolating colors
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        
        if self.is_train:
            # Random Horizontal Flip
            if random.random() < self.aug_cfg.hflip_prob:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
                
            # Random Vertical Flip
            if random.random() < self.aug_cfg.vflip_prob:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
                
            # Random Rotation (90 degrees multiples to avoid padding issues on borders)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)
                
            # Color jitter (ONLY applied to image)
            if random.random() < 0.5:
                img = TF.adjust_brightness(img, 1.0 + (random.random()-0.5)*self.aug_cfg.color_jitter)
                img = TF.adjust_contrast(img, 1.0 + (random.random()-0.5)*self.aug_cfg.color_jitter)

        # To Tensor (normalizes img to 0-1)
        img = TF.to_tensor(img)
        # Masks are handled by target builder, just keep as numpy array for now
        mask_np = np.array(mask)
        
        return img, mask_np

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        img_path = self.images_dir / filename
        mask_path = self.masks_dir / filename
        
        try:
            # Load images (drop alpha if RGBA)
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')
            
            # Apply geometric augmentations identically
            img_tensor, mask_np = self.apply_transform(img, mask)
            
            # Parse color mask into binary target tensors
            targets = self.target_builder.build_targets(mask_np)
            
            return {
                "image": img_tensor,
                "targets": targets,
                "filename": filename
            }
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            # Return next item on failure
            return self.__getitem__((idx + 1) % len(self))
