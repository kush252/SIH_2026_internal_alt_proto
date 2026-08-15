import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import time

from .checkpoint import save_checkpoint
from visualization.reconstruction import save_reconstruction_grid

class Trainer:
    def __init__(self, model, dataloader, config, output_dir, resume_path=None):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device(config.SYSTEM.device)
        
        self.model.to(self.device)
        if torch.cuda.device_count() > 1 and self.device.type == 'cuda':
            print(f"Using {torch.cuda.device_count()} GPUs via DataParallel!")
            self.model = nn.DataParallel(self.model)
        
        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.TRAINING.learning_rate,
            weight_decay=config.TRAINING.weight_decay
        )
        
        # Simple Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.TRAINING.epochs * len(dataloader)
        )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.TRAINING.use_amp)
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            from .checkpoint import load_checkpoint
            self.current_epoch, self.global_step = load_checkpoint(
                resume_path, self.model, self.optimizer, self.scheduler
            )
            print(f"Resumed at epoch {self.current_epoch}, step {self.global_step}")
        
    def train(self):
        print(f"Starting training for {self.config.TRAINING.epochs} epochs on {self.device}")
        
        for epoch in range(self.current_epoch, self.config.TRAINING.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(self.dataloader):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config.TRAINING.use_amp):
                    loss, rec, mask_up = self.model(images, masks)
                    
                    if loss.dim() > 0:
                        loss = loss.mean()
                        
                    # Normalize loss for gradient accumulation
                    loss = loss / self.config.TRAINING.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.TRAINING.accumulation_steps == 0:
                    if self.config.TRAINING.clip_grad > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.TRAINING.clip_grad)
                        
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                
                loss_val = loss.item() * self.config.TRAINING.accumulation_steps
                epoch_loss += loss_val
                self.global_step += 1
                
                if self.global_step % 10 == 0:
                    self.writer.add_scalar("Loss/train", loss_val, self.global_step)
                    self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], self.global_step)
                    print(f"Epoch [{epoch}/{self.config.TRAINING.epochs}] Step [{batch_idx}/{len(self.dataloader)}] Loss: {loss_val:.4f}")
            
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            print(f"==== Epoch {epoch} Average Loss: {avg_epoch_loss:.4f} ====")
            
            # Save Checkpoint & Visualize every epoch
            is_best = avg_epoch_loss < self.best_loss
            if is_best:
                self.best_loss = avg_epoch_loss
                
            save_checkpoint(
                self.model, self.optimizer, self.scheduler, 
                epoch, self.global_step, self.config, 
                self.output_dir, is_best
            )
            
            # Visualize the last batch
            self.model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.config.TRAINING.use_amp):
                    loss, rec, mask_up = self.model(images, masks)
                vis_dir = os.path.join(self.output_dir, "visualizations")
                save_reconstruction_grid(images, images, rec, mask_up, epoch, self.global_step, vis_dir)
                
        self.writer.close()
        print("Training complete.")
