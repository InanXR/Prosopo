"""Training loop with mixed precision and gradient accumulation."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from ..models import Prosopo
from ..data import CASIAWebFaceDataset, get_train_transforms
from .config import TrainingConfig


class Trainer:
    """
    Trainer for Prosopo face embedding model.
    
    Features:
        - Mixed precision (FP16) training
        - Gradient accumulation
        - Checkpoint save/resume
        - LFW validation
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize training components
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler() if config.use_amp else None
        
        # Initialize dataloader
        self.train_loader = self._build_dataloader()
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        
        # Resume if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
    
    def _build_model(self) -> Prosopo:
        """Build and initialize model."""
        # Load class count
        with open(self.config.class_indices_path, 'r') as f:
            class_indices = json.load(f)
        num_classes = len(class_indices)
        
        print(f"Building Prosopo model with {num_classes} classes")
        
        model = Prosopo(
            num_classes=num_classes,
            backbone=self.config.backbone,
            embedding_dim=self.config.embedding_dim,
            pretrained=self.config.pretrained,
            arcface_scale=self.config.arcface_scale,
            arcface_margin=self.config.arcface_margin,
        )
        
        return model.to(self.device)
    
    def _build_optimizer(self) -> SGD:
        """Build SGD optimizer with momentum."""
        return SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
    
    def _build_scheduler(self) -> MultiStepLR:
        """Build learning rate scheduler."""
        return MultiStepLR(
            self.optimizer,
            milestones=self.config.lr_milestones,
            gamma=self.config.lr_gamma,
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
        print(f"  Batch size: {self.config.batch_size} x {self.config.accumulation_steps} = {self.config.batch_size * self.config.accumulation_steps} effective")
        print(f"  AMP: {self.config.use_amp}")
        print(f"  Checkpoints: {self.config.checkpoint_dir}")
        print()
        
        for epoch in range(self.start_epoch, self.config.epochs):
            self._train_epoch(epoch)
            
            # Step scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
            
            # Validation
            if (epoch + 1) in self.config.val_epochs:
                self._validate(epoch + 1)
        
        print("\nTraining complete!")
        self.save_checkpoint(self.config.epochs, final=True)
    
    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                logits = self.model(images, labels)
                loss = F.cross_entropy(logits, labels)
                loss = loss / self.config.accumulation_steps
            
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item() * self.config.accumulation_steps
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = total_correct / total_samples
            current_lr = self.scheduler.get_last_lr()[0]
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2%}',
                'lr': f'{current_lr:.6f}',
            })
        
        print(f"  Epoch {epoch+1} complete: loss={avg_loss:.4f}, acc={accuracy:.2%}")
    
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
            
        except Exception as e:
            print(f"  LFW validation failed: {e}")
    
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
        
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"  Resumed from epoch {self.start_epoch}")
