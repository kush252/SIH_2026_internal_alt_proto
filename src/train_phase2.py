import os
import random
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from datasets.generic_binary_dataset import GenericBinaryDataset
from models.task_heads import Phase2MultiTaskModel
from training.trainer_phase2 import Phase2Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Supervised Segmentation Training")
    parser.add_argument("--config", type=str, default="configs/phase2_building.yaml", help="Path to config file")
    parser.add_argument("--task", type=str, default=None, choices=["building", "road", "water"], help="Override task class (dynamically configures model)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override dataset_path in config")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Override Phase 1 encoder checkpoint path")
    parser.add_argument("--resume", type=str, default=None, help="Path to phase2_latest.pt to resume training")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Apply CLI overrides if provided
    if args.task:
        config.DATA.classes = {'background': [0, 0, 0], args.task: [255, 255, 255]}
        config.LOSS.weights = {args.task: 1.0}
        config.SYSTEM.output_dir = f"outputs_phase2_{args.task}"
        
    if args.dataset_path:
        config.DATA.dataset_path = args.dataset_path
    if args.checkpoint_path:
        config.MODEL.encoder.checkpoint_path = args.checkpoint_path
        
    set_seed(config.SYSTEM.seed)
    
    device = torch.device(config.SYSTEM.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset & Splits
    print(f"Setting up dataset from {config.DATA.dataset_path}...")
    
    train_dataset = GenericBinaryDataset(config.DATA.dataset_path, config, is_train=True)
    val_dataset = GenericBinaryDataset(config.DATA.dataset_path, config, is_train=False)
    
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
    
    # 2. Model setup
    print("\nInitializing Phase 2 Model...")
    model = Phase2MultiTaskModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Dataset      : {config.DATA.dataset_path}")
    print(f"Train Images : {len(train_dataset)}")
    print(f"Val Images   : {len(val_dataset)}")
    print(f"Batch Size   : {config.TRAINING.batch_size}")
    print(f"Total Params : {total_params:,}")
    print(f"Trainable    : {trainable_params:,}")
    print("=" * 50 + "\n")
    
    # 3. Trainer
    trainer = Phase2Trainer(model, config, train_loader, val_loader, device, resume_path=args.resume)
    
    # 4. Fit
    trainer.fit()
    
if __name__ == "__main__":
    main()
