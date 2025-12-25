"""Model architectures."""

from .backbone import get_backbone
from .arcface import ArcFaceHead
from .prosopo import Prosopo

__all__ = ['get_backbone', 'ArcFaceHead', 'Prosopo']
