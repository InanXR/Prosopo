"""Prosopo - Face embedding model trained from scratch using ArcFace loss."""

__version__ = "0.1.0"

# Explicitly expose submodules for proper package resolution
from . import models
from . import data
from . import training
from . import evaluation
