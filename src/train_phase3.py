import argparse
import os
import random
import numpy as np
import torch

from utils.config import load_config
from datasets.roof_dataset import get_roof_dataloaders
from models.roof_classifier import Phase3RoofClassifier
from training.trainer_phase3 import Phase3Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Roof Classification Training")
    parser.add_argument('--config', type=str, default=r"configs\phase3_roof.yaml")
    parser.add_argument('--dataset_path', type=str, help="Path to roof crops dataset (overrides config)")
    parser.add_argument('--csv_path', type=str, help="Path to metadata CSV file")
    parser.add_argument('--checkpoint_path', type=str, help="Path to Phase 1 encoder checkpoint")
    parser.add_argument('--output_dir', type=str, help="Path to save outputs")
    parser.add_argument('--batch_size', type=int, help="Override batch size")
    parser.add_argument('--epochs', type=int, help="Override epochs")
    parser.add_argument('--resume', type=str, help="Path to Phase 3 checkpoint to resume from")
    parser.add_argument('--class_weights', type=float, nargs='+', 
                        help="Override class weights e.g. --class_weights 3.38 1.23 0.74 0.82")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Overrides
    if args.dataset_path:
        config.DATA.dataset_path = args.dataset_path
    if args.csv_path:
        config.DATA.csv_path = args.csv_path
    if args.checkpoint_path:
        config.MODEL.encoder.checkpoint_path = args.checkpoint_path
    if args.output_dir:
        config.SYSTEM.output_dir = args.output_dir
    if args.batch_size:
        config.TRAINING.batch_size = args.batch_size
    if args.epochs:
        config.TRAINING.epochs = args.epochs
        
    set_seed(config.SYSTEM.seed)
    os.makedirs(config.SYSTEM.output_dir, exist_ok=True)
    
    print("Initializing Phase 3 Roof Classification Pipeline...")
    
    print("Setting up dataset...")
    train_loader, val_loader, num_classes = get_roof_dataloaders(config)
    
    # Load class weights from config or CLI override
    # Weights order must match dataset class folder alphabetical order (ImageFolder sorts classes alphabetically)
    class_weights = getattr(config, 'CLASS_WEIGHTS', None)
    if args.class_weights:
        class_weights_list = args.class_weights
        print(f"Using CLI class weights: {class_weights_list}")
    elif class_weights is not None:
        class_weights_list = list(getattr(class_weights, 'weights', []))
        print(f"Using config class weights: {class_weights_list}")
    else:
        class_weights_list = None
        print("No class weights specified. Using unweighted loss.")
    
    print("Initializing Model...")
    model = Phase3RoofClassifier(config, num_classes=num_classes)
    
    trainer = Phase3Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=config.SYSTEM.output_dir,
        class_weights=class_weights_list,
        resume_path=args.resume
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
