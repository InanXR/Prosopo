"""Training configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration for Prosopo training."""
    
    # Data
    # Recommended: Link your dataset to this path in Kaggle
    data_root: str = "/kaggle/input/ms1m-arcface/ms1m_align_112" # Example path
    class_indices_path: Optional[str] = None
    num_workers: int = 4
    image_size: int = 224  # High Resolution
    
    # Model
    backbone: str = "resnet100"  # Deeper Backbone
    embedding_dim: int = 512
    pretrained: bool = True
    
    # ArcFace / SubCenter
    arcface_scale: float = 64.0
    arcface_margin: float = 0.5
    sub_centers: int = 3  # K=3 sub-centers for robust training
    easy_margin: bool = False
    
    # Training
    batch_size: int = 128  # Safe for P100 16GB
    accumulation_steps: int = 1
    epochs: int = 25
    
    # Optimizer
    lr: float = 1e-3  # Lower LR for fine-tuning
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # LR Schedule (Milestones for 25 epochs)
    lr_milestones: List[int] = field(default_factory=lambda: [10, 18, 22])
    lr_gamma: float = 0.1
    warmup_epochs: int = 3
    
    # Checkpointing
    checkpoint_dir: str = "/kaggle/working/checkpoints"
    save_every: int = 1
    resume_from: Optional[str] = None
    
    # Validation
    val_epochs: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    lfw_root: str = "/kaggle/input/lfw-dataset/lfw"
    lfw_pairs_path: str = "/kaggle/input/lfw-dataset/pairs.txt"
    
    # Mixed precision & Logging
    use_amp: bool = True
    use_wandb: bool = True
    wandb_project: str = "prosopo-face-recognition"
    wandb_entity: Optional[str] = None

    # Advanced Regularization
    use_ema: bool = True
    ema_decay: float = 0.999
    clip_grad_norm: float = 5.0
    drop_path_rate: float = 0.1
    
    def __post_init__(self):
        if self.class_indices_path is None:
            # Default to local if not specified, but usually Kaggle input is read-only
            # so we might need to copy/generate this to working dir if it doesn't exist
            pass
