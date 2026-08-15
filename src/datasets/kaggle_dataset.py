import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class KaggleAerialDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.transform = transform
        
        # Collect all valid image paths
        self.image_paths = []
        if self.images_dir.exists():
            self.image_paths = [f for f in self.images_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.png', '.jpeg')]
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                
                if self.transform:
                    img, mask = self.transform(img)
                else:
                    mask = None
                    
                return {
                    "image": img,
                    "mask": mask,
                    "dataset_id": "kaggle",
                    "source_path": str(img_path)
                }
        except Exception as e:
            # Fallback to random if corrupted (rare, handled in inspection ideally)
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
