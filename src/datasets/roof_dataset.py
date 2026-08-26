import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class RoofCSVDataset(Dataset):
    def __init__(self, img_dir, df, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Try to find the filename column
        filename_cols = ['filename', 'image_name', 'image', 'file_name', 'id']
        filename = None
        for col in filename_cols:
            if col in row and not pd.isna(row[col]):
                filename = str(row[col])
                if not filename.endswith('.png') and not filename.endswith('.jpg'):
                    filename += '.png'
                break
                
        if not filename:
            # We should have dropped NaNs in get_roof_dataloaders, but just in case:
            print(f"Warning: Row {idx} has missing filename. Returning dummy image.")
            return Image.new('RGB', (224, 224)), int(row['mapped_label'])
            
        img_path = os.path.join(self.img_dir, filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # If file not found, print a warning and return a dummy image to prevent crash
            print(f"Warning: Could not load {img_path}")
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        # We mapped the labels in get_roof_dataloaders
        class_idx = int(row['mapped_label'])
        
        return image, class_idx

def get_roof_dataloaders(config):
    """
    Creates DataLoaders for Roof Classification reading from a flat folder of images and a CSV.
    """
    dataset_path = config.DATA.dataset_path
    csv_path = getattr(config.DATA, 'csv_path', None)
    
    if not csv_path or not os.path.exists(csv_path):
        raise ValueError(f"CSV path {csv_path} not found. Please provide a valid config.DATA.csv_path or --csv_path")
        
    df = pd.read_csv(csv_path)
    
    # 0. DROP MISSING IMAGES (mimicking split.py logic)
    initial_len = len(df)
    filename_cols = ['filename', 'image_name', 'image', 'file_name', 'id']
    actual_fname_col = None
    for col in filename_cols:
        if col in df.columns:
            actual_fname_col = col
            break
            
    if actual_fname_col:
        df = df.dropna(subset=[actual_fname_col])
        print(f"Dropped {initial_len - len(df)} missing images. Remaining: {len(df)}")
    
    print(f"Loaded CSV with {len(df)} valid rows. Columns: {list(df.columns)}")
    
    # Map labels to 0-3 first to stratify properly
    label_cols = ['material_class', 'label', 'material', 'roof_type', 'class', 'roof_material']
    label_col = None
    for col in label_cols:
        if col in df.columns:
            label_col = col
            break
    if not label_col:
        # Fallback to second column
        label_col = df.columns[1]
        
    print(f"Using column '{label_col}' as the label.")
        
    raw_to_class = {
        'AmorphousConcrete': 0, # RCC
        'ClayTiles': 1,         # TILED
        'ConcreteTiles': 1,     # TILED
        'MetalSheetMaterials': 2# TIN
    }
    
    # Everything else gets mapped to 3 (OTHER)
    df['mapped_label'] = df[label_col].map(lambda x: raw_to_class.get(str(x), 3))
    
    # Class-stratified split (85% train, 15% val)
    train_df, val_df = train_test_split(
        df, 
        test_size=0.15, 
        random_state=config.SYSTEM.seed, 
        stratify=df['mapped_label']
    )
    
    print(f"Split dataset: {len(train_df)} train, {len(val_df)} val")
    
    # Robust augmentations for top-down drone imagery.
    # Critical: Use Resize (NOT RandomResizedCrop) to preserve the full building crop context.
    train_transform = transforms.Compose([
        transforms.Resize((config.DATA.image_size, config.DATA.image_size)),
        transforms.RandomHorizontalFlip(p=config.AUGMENTATION.hflip_prob),
        transforms.RandomVerticalFlip(p=config.AUGMENTATION.vflip_prob),
        transforms.RandomRotation(config.AUGMENTATION.random_rotation),
        transforms.ColorJitter(
            brightness=config.AUGMENTATION.color_jitter,
            contrast=config.AUGMENTATION.color_jitter,
            saturation=config.AUGMENTATION.color_jitter
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # For validation, just resize and normalize
    val_transform = transforms.Compose([
        transforms.Resize((config.DATA.image_size, config.DATA.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = RoofCSVDataset(dataset_path, train_df, transform=train_transform)
    val_dataset = RoofCSVDataset(dataset_path, val_df, transform=val_transform)
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.TRAINING.batch_size, 
        shuffle=True,
        num_workers=config.TRAINING.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.TRAINING.batch_size, 
        shuffle=False,
        num_workers=config.TRAINING.num_workers,
        pin_memory=True
    )
        
    num_classes = 4 # [RCC, TILED, TIN, OTHER]
    
    return train_loader, val_loader, num_classes
