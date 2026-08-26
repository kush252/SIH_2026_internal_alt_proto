import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_roof_dataloaders(config):
    """
    Creates DataLoaders for Roof Classification assuming an ImageFolder structure:
    dataset_path/
      Concrete/
      Tin/
      Thatched/
      ...
    """
    dataset_path = config.DATA.dataset_path
    
    # Robust augmentations for top-down drone imagery.
    # Critical: Use Resize (NOT RandomResizedCrop) to preserve the full building crop context.
    # The model is trained on raw rectangular bounding box crops — cropping further destroys context.
    train_transform = transforms.Compose([
        transforms.Resize((config.DATA.image_size, config.DATA.image_size)),
        transforms.RandomHorizontalFlip(p=config.AUGMENTATION.hflip_prob),
        transforms.RandomVerticalFlip(p=config.AUGMENTATION.vflip_prob),
        # 90deg rotation: roof material doesn't depend on orientation in top-down drone imagery
        transforms.RandomRotation(config.AUGMENTATION.random_rotation),
        transforms.ColorJitter(
            brightness=config.AUGMENTATION.color_jitter,
            contrast=config.AUGMENTATION.color_jitter,
            saturation=config.AUGMENTATION.color_jitter
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # For validation/testing, we just resize and normalize
    val_transform = transforms.Compose([
        transforms.Resize((config.DATA.image_size, config.DATA.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if dataset has train/val splits, otherwise just use the root for training
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")
    
    if os.path.exists(train_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform) if os.path.exists(val_dir) else None
    else:
        # Fallback if no explicit train/val folders exist (just treat root as train)
        train_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
        val_dataset = None
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.TRAINING.batch_size, 
        shuffle=True,
        num_workers=config.TRAINING.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.TRAINING.batch_size, 
            shuffle=False,
            num_workers=config.TRAINING.num_workers,
            pin_memory=True
        )
        
    num_classes = len(train_dataset.classes)
    print(f"Discovered {num_classes} roof classes: {train_dataset.classes}")
    
    return train_loader, val_loader, num_classes
