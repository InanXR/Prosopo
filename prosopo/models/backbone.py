"""Backbone networks for feature extraction."""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def get_backbone(name: str = "resnet50", pretrained: bool = True) -> tuple[nn.Module, int]:
    """
    Get a backbone network for feature extraction.
    
    Args:
        name: Backbone architecture name ('resnet50')
        pretrained: Whether to load pretrained weights
        
    Returns:
        Tuple of (backbone_module, output_features_dim)
    """
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        
        # Remove the final FC layer
        out_features = model.fc.in_features  # 2048
        model.fc = nn.Identity()
        
        return model, out_features
    else:
        raise ValueError(f"Unknown backbone: {name}. Supported: resnet50")


class BackboneWithEmbedding(nn.Module):
    """Backbone + embedding projection layer."""
    
    def __init__(
        self,
        backbone_name: str = "resnet50",
        embedding_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.backbone, backbone_out = get_backbone(backbone_name, pretrained)
        
        # Embedding projection
        self.embedding = nn.Sequential(
            nn.Linear(backbone_out, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        self.embedding_dim = embedding_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from input image."""
        features = self.backbone(x)
        embedding = self.embedding(features)
        return embedding
