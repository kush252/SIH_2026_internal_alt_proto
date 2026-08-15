import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
from glob import glob
import json
from PIL import ImageDraw

class GenericBinaryDataset(Dataset):
    """
    A unified dataset loader for independent binary segmentation tasks.
    It automatically detects the folder structure (e.g. DeepGlobe vs SVAMITVA).
    """
    def __init__(self, data_dir, config, is_train=True):
        self.data_dir = Path(data_dir)
        self.config = config
        self.is_train = is_train
        self.image_size = config.DATA.image_size
        self.aug_cfg = config.AUGMENTATION
        
        # Only parse the single tracking class (e.g. 'building', 'road')
        # Skip background
        self.tracking_classes = {k: v for k, v in config.DATA.classes.items() if k != 'background'}
        
        self.image_paths = []
        self.mask_paths = []
        self._discover_files()
        
        # Simple deterministic train/val split based on 90/10 ratio
        # using a fixed seed to ensure no leakage
        self._apply_split()

    def _discover_files(self):
        print(f"[DEBUG] Checking dataset path: {self.data_dir.absolute()}")
        print(f"[DEBUG] Path exists: {self.data_dir.exists()}")
        
        # Pattern 1: SVAMITVA (Images/ and Masks/ subdirectories)
        images_dir = self.data_dir / "Images"
        masks_dir = self.data_dir / "Masks"
        
        print(f"[DEBUG] Images dir exists: {images_dir.exists()}")
        print(f"[DEBUG] Masks dir exists: {masks_dir.exists()}")
        
        if images_dir.exists() and masks_dir.exists():
            print("[DEBUG] Detected SVAMITVA Pattern.")
            img_files = sorted(glob(str(images_dir / "*")))
            print(f"[DEBUG] Found {len(img_files)} files in Images directory.")
            
            for img_p in img_files:
                name = Path(img_p).name
                mask_p = masks_dir / name
                # Handle edge case where teammate code uses "Mask" inside string
                if not mask_p.exists():
                    name_replaced = name.replace("Image", "Mask")
                    mask_p = masks_dir / name_replaced
                
                if mask_p.exists():
                    self.image_paths.append(img_p)
                    self.mask_paths.append(str(mask_p))
                    
        # Pattern 2: DeepGlobe (Images and masks in same dir: 123_sat.jpg, 123_mask.png)
        else:
            sat_files = sorted(glob(str(self.data_dir / "*_sat.jpg")))
            if len(sat_files) > 0:
                print("[DEBUG] Detected DeepGlobe Pattern.")
                for sat_p in sat_files:
                    mask_p = sat_p.replace("_sat.jpg", "_mask.png")
                    if os.path.exists(mask_p):
                        self.image_paths.append(sat_p)
                        self.mask_paths.append(mask_p)
            
            # Pattern 3: BONAI JSON Pattern (Images and JSON labels in same dir or images/labels dir)
            else:
                json_files = sorted(glob(str(self.data_dir / "*.json")) + glob(str(self.data_dir / "labels" / "*.json")))
                if len(json_files) > 0:
                    print("[DEBUG] Detected BONAI JSON Pattern.")
                    for json_p in json_files:
                        name = Path(json_p).stem
                        # check for image in data_dir
                        img_p_png = self.data_dir / f"{name}.png"
                        img_p_jpg = self.data_dir / f"{name}.jpg"
                        
                        # sometimes images are in 'images' folder if labels are in 'labels'
                        if "labels" in json_p:
                            img_p_png = Path(json_p.replace("labels", "images").replace(".json", ".png"))
                            img_p_jpg = Path(json_p.replace("labels", "images").replace(".json", ".jpg"))
                            
                        if img_p_png.exists():
                            self.image_paths.append(str(img_p_png))
                            self.mask_paths.append(json_p)
                        elif img_p_jpg.exists():
                            self.image_paths.append(str(img_p_jpg))
                            self.mask_paths.append(json_p)
                    
        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"Could not find valid image/mask pairs in {self.data_dir}. "
                f"Please verify this path exists on Kaggle using: !ls \"{self.data_dir}\""
            )
            
    def _apply_split(self):
        # Pair them to shuffle together
        pairs = list(zip(self.image_paths, self.mask_paths))
        rng = random.Random(self.config.SYSTEM.seed)
        rng.shuffle(pairs)
        
        split_idx = int(len(pairs) * 0.9)
        if self.is_train:
            pairs = pairs[:split_idx]
        else:
            pairs = pairs[split_idx:]
            
        self.image_paths, self.mask_paths = zip(*pairs)
        self.image_paths = list(self.image_paths)
        self.mask_paths = list(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def apply_transform(self, img, mask):
        img = TF.resize(img, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        
        if self.is_train:
            if random.random() < self.aug_cfg.hflip_prob:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < self.aug_cfg.vflip_prob:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)
            if random.random() < 0.5:
                img = TF.adjust_brightness(img, 1.0 + (random.random()-0.5)*self.aug_cfg.color_jitter)
                img = TF.adjust_contrast(img, 1.0 + (random.random()-0.5)*self.aug_cfg.color_jitter)

        img = TF.to_tensor(img)
        mask_np = np.array(mask)
        return img, mask_np

    def build_binary_targets(self, rgb_mask):
        """Builds a single binary mask for the tracked class"""
        if rgb_mask.shape[-1] == 4:
            rgb_mask = rgb_mask[..., :3]
            
        targets = {}
        for class_name, target_color in self.tracking_classes.items():
            target_color_np = np.array(target_color, dtype=rgb_mask.dtype)
            
            # Grayscale adaptation (if mask is 2D, make it 3D identical across channels)
            if len(rgb_mask.shape) == 2:
                rgb_mask = np.stack([rgb_mask]*3, axis=-1)
                
            # If target_color is [255,255,255], DeepGlobe is often thresholded at 128
            if np.all(target_color_np == 255):
                # Binary thresholding for white masks
                mask = np.mean(rgb_mask, axis=-1) > 128
            else:
                # Exact color matching
                mask = np.all(rgb_mask == target_color_np, axis=-1)
                
            targets[class_name] = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
            
        return targets

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            if mask_path.endswith('.json'):
                # Render BONAI JSON polygon mask on the fly
                with open(mask_path, 'r') as f:
                    data = json.load(f)
                
                mask_img = Image.new('L', (img.width, img.height), 0)
                draw = ImageDraw.Draw(mask_img)
                
                for ann in data.get('annotations', []):
                    if 'footprint' in ann:
                        poly = ann['footprint']
                        if len(poly) >= 6: # Need at least 3 points (x,y)
                            draw.polygon(poly, outline=255, fill=255)
                            
                mask_img = mask_img.convert('RGB')
            else:
                # Open mask, but support L (grayscale) or RGB
                mask_img = Image.open(mask_path)
                if mask_img.mode != 'RGB':
                    mask_img = mask_img.convert('RGB')
                
            img_tensor, mask_np = self.apply_transform(img, mask_img)
            targets = self.build_binary_targets(mask_np)
            
            return {
                "image": img_tensor,
                "targets": targets,
                "filename": Path(img_path).name
            }
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
