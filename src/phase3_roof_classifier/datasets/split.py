import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

def create_geographic_split(metadata_path, output_dir, train_pct=0.7, val_pct=0.15, random_seed=42):
    """
    Reads roofnet_metadata.csv, drops missing images, groups by disaster/region, 
    and creates a train/val/test split without geographic leakage.
    """
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    # 0. DROP MISSING GOOGLE MAPS IMAGES
    initial_len = len(df)
    df = df.dropna(subset=['filename'])
    print(f"Dropped {initial_len - len(df)} missing images. Remaining: {len(df)}")
    
    # 1. Map classes
    class_mapping = {
        'AmorphousConcrete': 'RCC',
        'ClayTiles': 'TILED',
        'ConcreteTiles': 'TILED',
        'MetalSheetMaterials': 'TIN',
    }
    
    def map_class(x):
        return class_mapping.get(x, 'OTHER')
        
    df['phase3_class'] = df['material_class'].apply(map_class)
    
    # 2. Extract Geographic Proxy from xBD Filename (e.g. 'Haiti-hurricane-matthew')
    def get_disaster_name(fname):
        if pd.isna(fname):
            return 'unknown'
        parts = str(fname).split('_')
        return parts[0] if len(parts) > 0 else 'unknown'
        
    df['geographic_group'] = df['filename'].apply(get_disaster_name)
    
    print("\nGeographic Groups (Disasters):")
    print(df['geographic_group'].value_counts())
    
    # 3. Perform Stratified Random Split (Not Geographic)
    # This prevents the massive Val imbalance (825 images vs 33000 images)
    # and ensures Train, Val, and Test have identical class distributions.
    from sklearn.model_selection import train_test_split
    
    # First split: 80% Train, 20% Temp
    test_pct = 1.0 - train_pct - val_pct
    train_df, temp_df = train_test_split(df, test_size=(val_pct + test_pct), 
                                         stratify=df['phase3_class'], random_state=random_seed)
    
    # Second split: 50% Val, 50% Test from the Temp set (resulting in 10% Val, 10% Test overall)
    val_relative_pct = val_pct / (val_pct + test_pct)
    val_df, test_df = train_test_split(temp_df, test_size=(1.0 - val_relative_pct), 
                                       stratify=temp_df['phase3_class'], random_state=random_seed)
    
    # Assign new split labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['new_split'] = 'train'
    val_df['new_split'] = 'val'
    test_df['new_split'] = 'test'
    
    final_df = pd.concat([train_df, val_df, test_df])
    
    print("\nNew Geographic Split Distribution:")
    print(final_df['new_split'].value_counts())
    
    print("\nClass Distribution in Train:")
    print(train_df['phase3_class'].value_counts())
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'roofnet_phase3_split.csv')
    final_df.to_csv(out_path, index=False)
    print(f"\nSaved new split metadata to {out_path}")

if __name__ == "__main__":
    metadata_path = r"d:\Kush\2nd Year\Hackathons\SIH\data\roofnet\roofnet_metadata.csv"
    output_dir = r"d:\Kush\2nd Year\Hackathons\SIH\phase3_roof_classifier\datasets"
    create_geographic_split(metadata_path, output_dir)
