import os
import json
import glob
from collections import defaultdict
from PIL import Image
import numpy as np
from pathlib import Path

# Increase max image pixels to avoid DecompressionBombWarning for large GeoTIFFs
Image.MAX_IMAGE_PIXELS = None

def inspect_dataset(dataset_dir):
    print(f"Inspecting dataset at {dataset_dir}...")
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"
    
    if not images_dir.exists():
        print(f"  Warning: No 'images' directory found in {dataset_dir}")
        return None
        
    image_files = [f for f in images_dir.iterdir() if f.is_file()]
    mask_files = [f for f in masks_dir.iterdir() if f.is_file()] if masks_dir.exists() else []
    
    extensions = defaultdict(int)
    dimensions = defaultdict(int)
    channels_dict = defaultdict(int)
    dtypes = defaultdict(int)
    
    num_corrupted = 0
    num_usable = 0
    
    # We will sample up to 100 images to estimate stats
    sample_size = min(100, len(image_files))
    sample_files = np.random.choice(image_files, sample_size, replace=False) if len(image_files) > 0 else []
    
    pixel_sums = np.zeros(3)
    pixel_sq_sums = np.zeros(3)
    pixel_count = 0
    
    print(f"  Found {len(image_files)} total files in images dir. Sampling {sample_size} for detailed stats...")
    
    for i, img_path in enumerate(sample_files):
        ext = img_path.suffix.lower()
        extensions[ext] += 1
        
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                mode = img.mode
                
                dimensions[f"{w}x{h}"] += 1
                channels_dict[mode] += 1
                
                # Convert to numpy for stats
                img_array = np.array(img)
                dtypes[str(img_array.dtype)] += 1
                
                # Check for NaNs or Infinity (more relevant for floating point GeoTIFFs)
                if not np.isfinite(img_array).all():
                    print(f"  Warning: Invalid pixels found in {img_path.name}")
                
                # Calculate mean/std (assuming RGB/RGBA/Grayscale)
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
                
                # Basic RGB channel stats
                if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                    pixel_sums += img_array.sum(axis=(0,1))[:3]
                    pixel_sq_sums += (img_array.astype(np.float64)**2).sum(axis=(0,1))[:3]
                    pixel_count += w * h
                    
            num_usable += 1
        except Exception as e:
            num_corrupted += 1
            print(f"  Error reading {img_path.name}: {e}")
            
    # Calculate global mean and std from samples
    mean = (pixel_sums / pixel_count) if pixel_count > 0 else [0, 0, 0]
    variance = (pixel_sq_sums / pixel_count) - (mean ** 2) if pixel_count > 0 else [0, 0, 0]
    std = np.sqrt(np.maximum(variance, 0))
            
    return {
        "dataset_name": dataset_path.name,
        "total_images": len(image_files),
        "total_masks": len(mask_files),
        "extensions": dict(extensions),
        "dimensions_sample": dict(dimensions),
        "modes_sample": dict(channels_dict),
        "dtypes_sample": dict(dtypes),
        "corrupted_sample": num_corrupted,
        "usable_sample": num_usable,
        "estimated_stats": {
            "mean": mean.tolist(),
            "std": std.tolist()
        }
    }

def main():
    data_root = Path(r"d:\Kush\2nd Year\Hackathons\SIH\data")
    datasets_to_inspect = ["aerial_imagery_semantic", "landcover"]
    
    report = []
    
    for ds_name in datasets_to_inspect:
        ds_dir = data_root / ds_name
        if ds_dir.exists():
            ds_report = inspect_dataset(ds_dir)
            if ds_report:
                report.append(ds_report)
        else:
            print(f"Directory {ds_dir} does not exist.")
            
    report_path = Path(r"d:\Kush\2nd Year\Hackathons\SIH\src\dataset_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nInspection complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
