"""Ensemble model for combining multiple face recognition models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .prosopo import Prosopo

class EnsembleProsopo(nn.Module):
    """
    Ensemble of Prosopo models.
    Averages the normalized embeddings from multiple models.
    """
    def __init__(self, models: List[Prosopo]):
        super().__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x, tta: bool = True):
        embeddings = []
        for model in self.models:
            emb = model.get_embedding(x, tta=tta)
            embeddings.append(emb)
            
        # Stack: (num_models, batch, dim)
        embeddings = torch.stack(embeddings, dim=0)
        
        # Mean Pooling
        # Note: We must re-normalize after averaging!
        ensemble_emb = torch.mean(embeddings, dim=0)
        
        return F.normalize(ensemble_emb, p=2, dim=1)

    @classmethod
    def from_checkpoints(cls, checkpoints: List[str], config, device='cuda'):
        """Load multiple models from checkpoints."""
        models = []
        for ckpt in checkpoints:
            print(f"Loading ensemble member from {ckpt}...")
            
            # Recreate model structure
            # Note: We assume all models share the same config for now, 
            # or you can parse config from the checkpoint itself
            model = Prosopo(
                num_classes=10575, # Should match checkpoint
                backbone=config.backbone,
                embedding_dim=config.embedding_dim,
                sub_centers=config.sub_centers
            )
            
            # Load weights
            state = torch.load(ckpt, map_location=device)['model_state_dict']
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            models.append(model)
            
        return cls(models)
