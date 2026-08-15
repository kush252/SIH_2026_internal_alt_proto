import os
import torch
import torch.nn as nn
from tqdm import tqdm
from .metrics import MultitaskMetrics
from losses.multitask_loss import SemanticBoundaryLoss
import time

class Phase2Trainer:
    def __init__(self, model, config, train_loader, val_loader, device, resume_path=None):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = SemanticBoundaryLoss(config).to(device)
        self.metrics = MultitaskMetrics(list(config.LOSS.weights.keys()))
        
        # Differential Learning Rates
        encoder_params = []
        decoder_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'semantic_head' in name or 'boundary_head' in name:
                head_params.append(param)
            elif 'decoder' in name:
                decoder_params.append(param)
                
        self.optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': float(config.OPTIMIZER.encoder_lr)},
            {'params': decoder_params, 'lr': float(config.OPTIMIZER.decoder_lr)},
            {'params': head_params, 'lr': float(config.OPTIMIZER.head_lr)}
        ], weight_decay=config.OPTIMIZER.weight_decay)
        
        # Determine scaler (PyTorch >= 2.0 uses torch.amp, older uses torch.cuda.amp)
        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=config.TRAINING.use_amp)
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=config.TRAINING.use_amp)
            
        self.best_metric = 0.0
        self.start_epoch = 0
        
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming training from {resume_path}...")
            checkpoint = torch.load(resume_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_metric = checkpoint.get('best_metric', 0.0)
            print(f"Resumed at Epoch {self.start_epoch} with Best Metric: {self.best_metric:.4f}")
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        
        for step, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=self.config.TRAINING.use_amp):
                preds = self.model(images)
                loss, loss_dict = self.criterion(preds, targets)
                # Grad accum
                loss = loss / self.config.TRAINING.accumulation_steps
                
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.config.TRAINING.accumulation_steps == 0:
                if self.config.TRAINING.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.TRAINING.clip_grad)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
            total_loss += loss.item() * self.config.TRAINING.accumulation_steps
            pbar.set_postfix({'loss': f"{loss.item() * self.config.TRAINING.accumulation_steps:.4f}"})
            
        return total_loss / len(self.train_loader)
        
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        self.metrics.reset()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Val")
        class_names = list(self.config.LOSS.weights.keys())
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=self.config.TRAINING.use_amp):
                preds = self.model(images)
                loss, loss_dict = self.criterion(preds, targets)
                
            # For metrics, we use the raw semantic predictions directly
            semantic_logits = preds['semantic']
            semantic_preds = {}
            for i, name in enumerate(class_names):
                semantic_preds[name] = semantic_logits[:, i:i+1]
                
            total_loss += loss.item()
            self.metrics.update(semantic_preds, targets)
            
        val_metrics = self.metrics.compute()
        val_metrics['loss'] = total_loss / len(self.val_loader)
        
        print(f"\n--- Validation Epoch {epoch} ---")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")
            
        return val_metrics
        
    def fit(self):
        print(f"Starting Phase 2 training on {self.device}")
        
        out_dir = self.config.SYSTEM.output_dir
        os.makedirs(out_dir, exist_ok=True)
        
        for epoch in range(self.start_epoch, self.config.TRAINING.epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            # Simple metric tracking: average of all task IoUs
            mean_iou = sum([v for k, v in val_metrics.items() if 'iou' in k]) / len(self.config.LOSS.weights)
            
            # Save checkpoint
            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_metric': max(self.best_metric, mean_iou)
            }
            
            torch.save(state, os.path.join(out_dir, 'phase2_latest.pt'))
            
            if mean_iou > self.best_metric:
                self.best_metric = mean_iou
                torch.save(state, os.path.join(out_dir, 'phase2_best.pt'))
                print(f"New best model saved with Mean IoU: {mean_iou:.4f}")
