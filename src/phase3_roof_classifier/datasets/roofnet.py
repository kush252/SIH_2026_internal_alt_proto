import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class RoofNetDataset(Dataset):
    def __init__(self, metadata_csv, images_dir, split='train', img_size=224):
        """
        Args:
            metadata_csv (str): Path to roofnet_phase3_split.csv
            images_dir (str): Path to xBD_cropped_roofs
            split (str): 'train', 'val', or 'test'
            img_size (int): Target image size for ConvNeXt
        """
        self.images_dir = images_dir
        self.split = split
        self.img_size = img_size
        
        # Load and filter split
        df = pd.read_csv(metadata_csv)
        self.df = df[df['new_split'] == split].reset_index(drop=True)
        
        # Mapping string class to int
        self.class_to_idx = {'RCC': 0, 'TILED': 1, 'TIN': 2, 'OTHER': 3}
        
        # Base transforms (used for val/test)
        self.base_transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])
        
        # Training augmentations
        self.train_transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.images_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for corrupted images (should be rare)
            image = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            
        if self.split == 'train':
            image = self.train_transforms(image)
        else:
            image = self.base_transforms(image)
            
        label = self.class_to_idx[row['phase3_class']]
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'filename': img_name
        }
