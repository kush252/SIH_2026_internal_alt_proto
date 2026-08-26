import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import json
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from .checkpoint import save_checkpoint

class Phase3Trainer:
    def __init__(self, model, train_loader, val_loader, config, output_dir, class_weights=None, resume_path=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device(config.SYSTEM.device)
        self.history = []
        
        self.model.to(self.device)
        if torch.cuda.device_count() > 1 and self.device.type == 'cuda':
            print(f"Using {torch.cuda.device_count()} GPUs via DataParallel!")
            self.model = nn.DataParallel(self.model)
            
        # Separate learning rates for pretrained encoder vs fresh classification head
        param_groups = [
            {'params': self.model.module.encoder.parameters() if hasattr(self.model, 'module') else self.model.encoder.parameters(), 
             'lr': float(config.OPTIMIZER.encoder_lr)},
            {'params': self.model.module.classifier.parameters() if hasattr(self.model, 'module') else self.model.classifier.parameters(), 
             'lr': float(config.OPTIMIZER.head_lr)}
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(config.OPTIMIZER.weight_decay)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.TRAINING.epochs * len(train_loader)
        )
        
        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=config.TRAINING.use_amp)
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=config.TRAINING.use_amp)
            
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
        
        # Weighted loss to handle class imbalance (RCC, TILED, TIN, OTHER)
        # Default sqrt-softened weights from xBD distribution: [3.38, 1.23, 0.74, 0.82]
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            print(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("WARNING: No class weights provided. Class imbalance may hurt minority class recall.")
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_f1 = 0.0  # Track macro F1, not accuracy (imbalanced dataset)
        
    def train(self):
        print(f"Starting Phase 3 training for {self.config.TRAINING.epochs} epochs on {self.device}")
        
        for epoch in range(self.current_epoch, self.config.TRAINING.epochs):
            self.model.train()
            epoch_loss = 0.0
            all_preds = []
            all_labels = []
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.TRAINING.epochs} [Train]")
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                try:
                    amp_context = torch.amp.autocast('cuda', enabled=self.config.TRAINING.use_amp)
                except AttributeError:
                    amp_context = torch.cuda.amp.autocast(enabled=self.config.TRAINING.use_amp)
                    
                with amp_context:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    
                self.scaler.scale(loss).backward()
                
                if self.config.TRAINING.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.TRAINING.clip_grad)
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                
                loss_val = loss.item()
                epoch_loss += loss_val
                self.global_step += 1
                
                _, predicted = torch.max(logits.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                running_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
                pbar.set_postfix({'Loss': f"{loss_val:.4f}", 'Acc': f"{running_acc:.1f}%"})
                
                # Log to tensorboard every 10 steps (but don't print to console)
                if self.global_step % 10 == 0:
                    self.writer.add_scalar("Loss/train", loss_val, self.global_step)
                    self.writer.add_scalar("Accuracy/train", running_acc, self.global_step)
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            train_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
            train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            # Print epoch summary
            print(f"==== Epoch {epoch+1} Train | Loss: {avg_epoch_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.4f} ====")
            
            epoch_record = {'epoch': epoch+1, 'train_loss': avg_epoch_loss, 'train_acc': train_acc, 'train_f1': train_f1}
            
            # Val metric = macro F1 (not accuracy) because class imbalance makes accuracy misleading
            val_f1 = train_f1  # Default if no val set
            if self.val_loader:
                val_loss, val_acc, val_f1 = self.evaluate(epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.writer.add_scalar("F1_macro/val", val_f1, epoch)
                print(f"==== Epoch {epoch+1} Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f} ====\n")
                epoch_record.update({'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1})
            else:
                print() # Just a newline for spacing
                
            self.history.append(epoch_record)
            with open(os.path.join(self.output_dir, 'history.json'), 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # Save best checkpoint based on macro F1, NOT accuracy
            is_best = val_f1 > self.best_f1
            if is_best:
                self.best_f1 = val_f1
                print(f"New best macro F1: {self.best_f1:.4f} — saving best.pt")
                
            save_checkpoint(
                self.model, self.optimizer, self.scheduler, 
                epoch, self.global_step, self.config, 
                self.output_dir, is_best
            )
                
        self.writer.close()
        print(f"Training complete. Best Val Macro F1: {self.best_f1:.4f}")
        
    @torch.no_grad()
    def evaluate(self, epoch):
        """Returns (val_loss, val_accuracy, val_macro_f1)."""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.TRAINING.epochs} [Val]  ")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            try:
                amp_context = torch.amp.autocast('cuda', enabled=self.config.TRAINING.use_amp)
            except AttributeError:
                amp_context = torch.cuda.amp.autocast(enabled=self.config.TRAINING.use_amp)
                
            with amp_context:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
            val_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            running_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{running_acc:.1f}%"})
            
        val_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return val_loss / len(self.val_loader), val_acc, val_f1
