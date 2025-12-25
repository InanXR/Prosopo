"""Full Prosopo model combining backbone and ArcFace head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import BackboneWithEmbedding
from .arcface import ArcFaceHead


class Prosopo(nn.Module):
    """
    Prosopo: Face embedding model with ArcFace training head.
    
    During training: returns logits for cross-entropy loss
    During inference: returns normalized embeddings for similarity search
    """
    
    def __init__(
        self,
        num_classes: int = 10575,
        backbone: str = "resnet50",
        embedding_dim: int = 512,
        pretrained: bool = True,
        arcface_scale: float = 64.0,
        arcface_margin: float = 0.5,
    ):
        """
        Args:
            num_classes: Number of identity classes in training set
            backbone: Backbone architecture name
            embedding_dim: Dimension of face embeddings
            pretrained: Use pretrained backbone weights
            arcface_scale: ArcFace scale parameter
            arcface_margin: ArcFace angular margin (radians)
        """
        super().__init__()
        
        # Feature extraction backbone
        self.backbone = BackboneWithEmbedding(
            backbone_name=backbone,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
        )
        
        # ArcFace classification head (used only during training)
        self.arcface = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
        )
        
        self.embedding_dim = embedding_dim
        
    def forward(
        self, 
        x: Tensor, 
        labels: Tensor = None
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, 3, 112, 112) input face images
            labels: (batch,) integer labels (required for training)
            
        Returns:
            Training (labels provided): (batch, num_classes) logits
            Inference (no labels): (batch, embedding_dim) normalized embeddings
        """
        # Extract embeddings
        embeddings = self.backbone(x)
        
        if labels is not None:
            # Training mode: return logits for cross-entropy
            logits = self.arcface(embeddings, labels)
            return logits
        else:
            # Inference mode: return normalized embeddings
            return F.normalize(embeddings, p=2, dim=1)
    
    def get_embedding(self, x: Tensor) -> Tensor:
        """Extract normalized embedding for inference."""
        embeddings = self.backbone(x)
        return F.normalize(embeddings, p=2, dim=1)
    
    @torch.no_grad()
    def compute_similarity(self, emb1: Tensor, emb2: Tensor) -> Tensor:
        """Compute cosine similarity between two embeddings."""
        return F.cosine_similarity(emb1, emb2)
