"""Training utilities."""

from .trainer import Trainer
from .config import TrainingConfig

# Alias for simpler imports
Config = TrainingConfig

__all__ = ['Trainer', 'TrainingConfig', 'Config']
