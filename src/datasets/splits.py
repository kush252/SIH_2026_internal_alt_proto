import os
from pathlib import Path
import random

def get_svamitva_splits(data_dir, split_dir="splits", seed=42):
    """
    Deterministically splits the SVAMITVA dataset into train (80%), 
    val (10%), and test (10%).
    Saves the splits to text files to ensure consistency across runs.
    """
    images_dir = Path(data_dir) / "Images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found at {images_dir}")
        
    all_files = sorted([f.name for f in images_dir.iterdir() if f.suffix.lower() == '.png'])
    
    # Check if splits already exist
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)
    
    train_file = split_path / "train.txt"
    val_file = split_path / "val.txt"
    test_file = split_path / "test.txt"
    
    if train_file.exists() and val_file.exists() and test_file.exists():
        with open(train_file, 'r') as f:
            train_files = f.read().splitlines()
        with open(val_file, 'r') as f:
            val_files = f.read().splitlines()
        with open(test_file, 'r') as f:
            test_files = f.read().splitlines()
        print(f"Loaded existing splits: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")
        return train_files, val_files, test_files
        
    # Generate new deterministic splits
    print("Generating new deterministic train/val/test splits...")
    rng = random.Random(seed)
    rng.shuffle(all_files)
    
    total = len(all_files)
    train_idx = int(0.8 * total)
    val_idx = int(0.9 * total)
    
    train_files = all_files[:train_idx]
    val_files = all_files[train_idx:val_idx]
    test_files = all_files[val_idx:]
    
    # Save them
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_files))
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_files))
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_files))
        
    print(f"Created splits: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")
    return train_files, val_files, test_files
