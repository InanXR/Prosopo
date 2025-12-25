"""Training configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration for Prosopo training."""
    
    # Data
    data_root: str = "/content/data/CASIA-WebFace-aligned"
    class_indices_path: Optional[str] = None
    num_workers: int = 2
    
    # Model
    backbone: str = "resnet50"
    embedding_dim: int = 512
    pretrained: bool = True
    
    # ArcFace
    arcface_scale: float = 64.0
    arcface_margin: float = 0.5
    
    # Training
    batch_size: int = 128
    accumulation_steps: int = 2
    epochs: int = 25
    
    # Optimizer
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # LR Schedule
    lr_milestones: List[int] = field(default_factory=lambda: [10, 18, 22])
    lr_gamma: float = 0.1
    
    # Checkpointing
    checkpoint_dir: str = "/content/drive/MyDrive/prosopo/checkpoints"
    save_every: int = 1  # Save every N epochs
    resume_from: Optional[str] = None
    
    # Validation
    val_epochs: List[int] = field(default_factory=lambda: [10, 15, 20, 25])
    lfw_root: str = "/content/data/lfw"
    lfw_pairs_path: str = "/content/data/pairs.txt"
    
    # Mixed precision
    use_amp: bool = True
    
    def __post_init__(self):
        if self.class_indices_path is None:
            self.class_indices_path = f"{self.data_root}/class_indices.json"
