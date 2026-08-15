import os
import json
from pathlib import Path
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
from PIL import Image

class LandCoverDataset(Dataset):
    def __init__(self, data_dir, patch_size=224, stride=112, transform=None):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        
        self.index_file = self.data_dir / f"patch_index_{patch_size}_{stride}.json"
        self.patches = self._build_or_load_index()
        
    def _build_or_load_index(self):
        # Check original location first
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
                
        # Fallback to local working directory if dataset is read-only (e.g. Kaggle /input)
        local_index_file = Path.cwd() / self.index_file.name
        if local_index_file.exists():
            with open(local_index_file, 'r') as f:
                return json.load(f)
                
        print("Building patch index for LandCover.ai... This may take a moment.")
        patches = []
        if self.images_dir.exists():
            image_files = [f for f in self.images_dir.iterdir() if f.is_file() and f.suffix.lower() == '.tif']
            
            for img_path in image_files:
                try:
                    with rasterio.open(img_path) as src:
                        width = src.width
                        height = src.height
                        
                        # Generate patch coordinates
                        for y in range(0, height - self.patch_size + 1, self.stride):
                            for x in range(0, width - self.patch_size + 1, self.stride):
                                patches.append({
                                    "path": str(img_path),
                                    "x": x,
                                    "y": y
                                })
                except Exception as e:
                    print(f"Skipping {img_path} due to error: {e}")
                    
        # Try to save the index. If dataset dir is read-only, save locally
        try:
            with open(self.index_file, 'w') as f:
                json.dump(patches, f)
            print(f"Saved index to {self.index_file}")
        except (PermissionError, OSError):
            local_index_file = Path.cwd() / self.index_file.name
            with open(local_index_file, 'w') as f:
                json.dump(patches, f)
            print(f"Dataset directory read-only. Saved index locally to {local_index_file}")
            
        print(f"Built index with {len(patches)} patches.")
        return patches
        
    def __len__(self):
        return len(self.patches)
        
    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        
        # Dynamically construct path using just the filename to support moving the index between environments
        # Handle cross-OS parsing (Windows path -> Linux runtime) by explicitly replacing backslashes
        img_filename = patch_info["path"].replace('\\', '/').split('/')[-1]
        img_path = str(self.images_dir / img_filename)
        
        x, y = patch_info["x"], patch_info["y"]
        
        try:
            with rasterio.open(img_path) as src:
                window = Window(x, y, self.patch_size, self.patch_size)
                # Read all channels (assumes 1,2,3 are RGB)
                img_array = src.read((1, 2, 3), window=window)
                # rasterio returns (C, H, W). We need to transpose to (H, W, C) for PIL
                img_array = img_array.transpose(1, 2, 0)
                
                img = Image.fromarray(img_array).convert('RGB')
                
                if self.transform:
                    img, mask = self.transform(img)
                else:
                    mask = None
                    
                return {
                    "image": img,
                    "mask": mask,
                    "dataset_id": "landcover",
                    "source_path": f"{img_path}_{x}_{y}"
                }
        except Exception as e:
            print(f"Error loading {img_path} at ({x},{y}): {e}")
            return self.__getitem__((idx + 1) % len(self))
