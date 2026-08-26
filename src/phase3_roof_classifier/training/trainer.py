import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import numpy as np
from tqdm import tqdm
import os
import json

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        self.scaler = torch.amp.GradScaler('cuda')
        os.makedirs(save_dir, exist_ok=True)
        self.best_f1 = 0.0
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        preds, targets = [], []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')
        print(f"Train - Loss: {total_loss/len(self.train_loader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        return {'loss': total_loss/len(self.train_loader), 'acc': acc, 'f1': f1}

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        preds, targets = [], []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        acc = accuracy_score(targets, preds)
        f1_macro = f1_score(targets, preds, average='macro')
        f1_per_class = f1_score(targets, preds, average=None)
        cm = confusion_matrix(targets, preds)
        
        print(f"Val - Loss: {total_loss/len(self.val_loader):.4f} | Acc: {acc:.4f} | Macro F1: {f1_macro:.4f}")
        print("Per-class F1 [RCC, TILED, TIN, OTHER]:", f1_per_class)
        print("Confusion Matrix:")
        print(cm)
        
        # Save best model
        if f1_macro > self.best_f1:
            self.best_f1 = f1_macro
            save_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'f1': f1_macro
            }, save_path)
            print(f"Saved new best model to {save_path}")
            
        return {'loss': total_loss/len(self.val_loader), 'acc': acc, 'f1': f1_macro, 'cm': cm.tolist()}

    def fit(self, num_epochs):
        history = {'train': [], 'val': []}
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            if self.scheduler:
                self.scheduler.step()
                
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)
            
            # Save history
            with open(os.path.join(self.save_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=4)
