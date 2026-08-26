import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from phase3_roof_classifier.datasets.roofnet import RoofNetDataset
from phase3_roof_classifier.models.classifier import RoofClassifier
from phase3_roof_classifier.training.trainer import Trainer
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', type=str, default=r'd:\Kush\2nd Year\Hackathons\SIH\phase3_roof_classifier\datasets\roofnet_phase3_split.csv')
    parser.add_argument('--images_dir', type=str, default=r'd:\Kush\2nd Year\Hackathons\SIH\data\roofnet\xBD_cropped_roofs\xBD_cropped_roofs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--model_name', type=str, default='convnext_tiny')
    parser.add_argument('--save_dir', type=str, default='phase3_roof_classifier/outputs')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Datasets & DataLoaders
    print("Loading datasets...")
    train_dataset = RoofNetDataset(args.metadata_csv, args.images_dir, split='train')
    val_dataset = RoofNetDataset(args.metadata_csv, args.images_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Total Train Images: {len(train_dataset)}")
    print(f"Total Val Images: {len(val_dataset)}")
    
    # 2. Model
    print(f"Initializing model {args.model_name}...")
    model = RoofClassifier(model_name=args.model_name, num_classes=4, pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")
    
    # 3. Loss & Optimizer
    # Class weights softened using square root to prevent massive over-penalization of minority classes
    # Original: [11.47, 1.53, 0.55, 0.68] -> Softened: [3.38, 1.23, 0.74, 0.82]
    weights = torch.tensor([3.38, 1.23, 0.74, 0.82], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Increased weight decay to 1e-2 to heavily penalize over-complexity and fight overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 4. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir
    )
    
    # 5. Train
    print("Starting training...")
    trainer.fit(num_epochs=args.epochs)

if __name__ == '__main__':
    main()
