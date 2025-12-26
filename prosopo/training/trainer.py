"""Training loop with mixed precision and gradient accumulation."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import MultiStepLR, LinearLR, SequentialLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from ..models import Prosopo
from ..data import CASIAWebFaceDataset, get_train_transforms
from .config import TrainingConfig
from .utils import ModelEMA


class Trainer:
    """
    Trainer for Prosopo face embedding model.
    
    Features:
        - Mixed precision (FP16) training
        - Gradient accumulation
        - Checkpoint save/resume
        - LFW validation
        - WandB Logging
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging
        if self.config.use_wandb and wandb is not None:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.__dict__
            )
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize training components
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler() if config.use_amp else None
        
# ... (In __init__)
        self.train_loader = self._build_dataloader()
        
        # Initialize EMA
        self.model_ema = None
        if self.config.use_ema:
            print(f"Initializing Model EMA (decay={self.config.ema_decay})")
            self.model_ema = ModelEMA(self.model, decay=self.config.ema_decay)
        
        # Training state
        self.start_epoch = 0
# ...
    
    def _train_epoch(self, epoch: int):
        # ...
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                # Unscale for gradient clipping
                if self.config.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            else:
                loss.backward()
                if self.config.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.model_ema is not None:
                    self.model_ema.update(self.model)
                
                self.global_step += 1
# ...
        self.global_step = 0
        
        # Resume if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
    
    def _build_model(self) -> Prosopo:
        """Build and initialize model."""
        # Try to load class count, fallback to autodetect or manual config
        class_indices = {}
        if self.config.class_indices_path and os.path.exists(self.config.class_indices_path):
            with open(self.config.class_indices_path, 'r') as f:
                class_indices = json.load(f)
            num_classes = len(class_indices)
        else:
            # Fallback for when we don't have the indices file yet (it's built during dataset init)
            # This requires us to trust the user provided value or dataset scan
            print("Warning: Class indices path not found. Will be determined by dataset.")
            # We defer until dataset is loaded, but Prosopo model needs num_classes immediately.
            # HACK: For now assume standard CASIA-WebFace size if not found, or let Dataset check.
            # Real fix: Scan directories here if needed.
            folders = [d for d in Path(self.config.data_root).iterdir() if d.is_dir()]
            num_classes = len(folders)
        
        print(f"Building Prosopo model with {num_classes} classes")
        
        model = Prosopo(
            num_classes=num_classes,
            backbone=self.config.backbone,
            embedding_dim=self.config.embedding_dim,
            pretrained=self.config.pretrained,
            arcface_scale=self.config.arcface_scale,
            arcface_margin=self.config.arcface_margin,
            # easy_margin=self.config.easy_margin # Passed via property later if not supported in init
        )
        # Manually set easy_margin if the model class hasn't been updated yet (handled in Prosopo update)
        if hasattr(model.arcface, 'easy_margin'):
            model.arcface.easy_margin = self.config.easy_margin
            
        return model.to(self.device)
    
    def _build_optimizer(self) -> AdamW:
        """Build AdamW optimizer (Better convergence than SGD)."""
        return AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
    
    def _build_scheduler(self) -> SequentialLR:
        """Build learning rate scheduler with Warmup."""
        # 1. Warmup scheduler
        warmup = LinearLR(
            self.optimizer, 
            start_factor=0.01, 
            end_factor=1.0, 
            total_iters=self.config.warmup_epochs
        )
        
        # 2. Main scheduler (MultiStep)
        main_scheduler = MultiStepLR(
            self.optimizer,
            milestones=self.config.lr_milestones,
            gamma=self.config.lr_gamma,
        )
        
        # Combine
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[self.config.warmup_epochs]
        )
    
    def _build_dataloader(self) -> DataLoader:
        """Build training dataloader."""
        dataset = CASIAWebFaceDataset(
            data_root=self.config.data_root,
            transform=get_train_transforms(),
            class_indices_path=self.config.class_indices_path,
        )
        
        print(f"Training dataset: {len(dataset)} images, {dataset.num_classes} classes")
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def train(self):
        """Run full training loop."""
        print(f"\nStarting training from epoch {self.start_epoch}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  AMP: {self.config.use_amp}")
        print(f"  WandB: {self.config.use_wandb}")
        print()
        
        for epoch in range(self.start_epoch, self.config.epochs):
            self._train_epoch(epoch)
            
            # Step scheduler (SequentialLR steps per epoch)
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
            
            # Validation
            if (epoch + 1) in self.config.val_epochs:
                self._validate(epoch + 1)
        
        print("\nTraining complete!")
        self.save_checkpoint(self.config.epochs, final=True)
        
        if self.config.use_wandb and wandb is not None:
            wandb.finish()
    
    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        self.optimizer.zero_grad()
        
        # Get current LR for logging
        current_lr = self.scheduler.get_last_lr()[0]
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
from torch.optim import SGD, AdamW

# ... (inside Trainer)

    def _build_optimizer(self) -> AdamW:
        """Build AdamW optimizer (Better than SGD for AdaFace/Modern setups)."""
        return AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

# ... (inside train_epoch)

            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.config.use_amp):
                logits = self.model(images, labels)
                # Label Smoothing = 0.1 prevents Overfitting
                loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
                loss = loss / self.config.accumulation_steps
            
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient Clipping
                if self.config.clip_grad_norm > 0:
                     self.scaler.unscale_(self.optimizer)
                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            else:
                loss.backward()
                if self.config.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.model_ema is not None:
                    self.model_ema.update(self.model)

                self.global_step += 1
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            batch_size = labels.size(0)
            correct = (predicted == labels).sum().item()
            
            total_samples += batch_size
            total_correct += correct
            total_loss += loss.item() * self.config.accumulation_steps
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = total_correct / total_samples
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2%}',
                'lr': f'{current_lr:.6f}',
            })
            
            # WandB logging (every 50 steps)
            if self.config.use_wandb and wandb is not None and batch_idx % 50 == 0:
                wandb.log({
                    "train/loss": loss.item() * self.config.accumulation_steps,
                    "train/accuracy": correct / batch_size,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch,
                    "train/step": self.global_step
                })
        
        # End of epoch logging
        print(f"  Epoch {epoch+1} complete: loss={avg_loss:.4f}, acc={accuracy:.2%}")
        if self.config.use_wandb and wandb is not None:
            wandb.log({
                "epoch/loss": avg_loss,
                "epoch/accuracy": accuracy
            })
    
    def _validate(self, epoch: int):
        """Run LFW validation."""
        print(f"\n  Running LFW validation at epoch {epoch}...")
        
        try:
            from ..evaluation import evaluate_lfw
            
            accuracy, threshold = evaluate_lfw(
                self.model,
                self.config.lfw_root,
                self.config.lfw_pairs_path,
                device=self.device,
            )
            
            print(f"  LFW Accuracy: {accuracy:.2%} @ threshold {threshold:.3f}")
            
            if self.config.use_wandb and wandb is not None:
                wandb.log({
                    "val/lfw_accuracy": accuracy,
                    "val/lfw_threshold": threshold,
                    "val/epoch": epoch
                })
            
        except Exception as e:
            print(f"  LFW validation failed: {e}")
            if self.config.use_wandb and wandb is not None:
                wandb.alert(
                    title="Validation Failed",
                    text=f"LFW validation failed at epoch {epoch}: {str(e)}"
                )
    
    def save_checkpoint(self, epoch: int, final: bool = False):
        """Save training checkpoint to Drive."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        if final:
            path = os.path.join(self.config.checkpoint_dir, "prosopo_final.pth")
        else:
            path = os.path.join(self.config.checkpoint_dir, f"epoch_{epoch}.pth")
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        if self.model_ema is not None:
             checkpoint['model_ema_state_dict'] = self.model_ema.module.state_dict()
        
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint and resume training."""
        print(f"Loading checkpoint: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        if self.model_ema is not None and 'model_ema_state_dict' in checkpoint:
            self.model_ema.module.load_state_dict(checkpoint['model_ema_state_dict'])
        
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"  Resumed from epoch {self.start_epoch}")
