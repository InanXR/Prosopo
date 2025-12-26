"""Full Prosopo model combining backbone and ArcFace head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import BackboneWithEmbedding
from .arcface import AdaFaceHead


class Prosopo(nn.Module):
    """
    Prosopo: Face embedding model with AdaFace training head.
    
    Features:
        - IR-SE-50 Backbone
        - AdaFace Loss
        - Test-Time Augmentation (TTA)
    """
    
    def __init__(
        self,
        num_classes: int = 10575,
        backbone: str = "resnet50",
        embedding_dim: int = 512,
        pretrained: bool = True,
        arcface_scale: float = 64.0,
        arcface_margin: float = 0.5,
        easy_margin: bool = False,
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
        
        # Feature extraction backbone (Configured as IR-SE-50 internally)
        self.backbone = BackboneWithEmbedding(
            backbone_name=backbone,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
        )
        
        # AdaFace classification head (Renamed from arcface for clarity, but keeping attr name)
        self.arcface = AdaFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
            easy_margin=easy_margin,
        )
        
        self.embedding_dim = embedding_dim
        
    def forward(
        self, 
        x: Tensor, 
        labels: Tensor = None
    ) -> Tensor:
        """
        Forward pass.
        """
        # Extract embeddings (Un-normalized for AdaFace use)
        embeddings = self.backbone(x)
        
        if labels is not None:
            # Training mode: return logits
            # AdaFace handles normalization internally to compute quality norms
            logits = self.arcface(embeddings, labels)
            return logits
        else:
            # Inference mode: return normalized embeddings
            return F.normalize(embeddings, p=2, dim=1)
    
    def get_embedding(self, x: Tensor, tta: bool = True) -> Tensor:
        """
        Extract normalized embedding for inference.
        
        Args:
            x: Input image tensor
            tta: Use Test-Time Augmentation (Horizontal Flip)
        """
        if tta:
            # Original
            emb1 = self.backbone(x)
            
            # Flipped
            x_flip = torch.flip(x, dims=[3])
            emb2 = self.backbone(x_flip)
            
            # Average (sum since we normalize after)
            embeddings = emb1 + emb2
        else:
            embeddings = self.backbone(x)
            
        return F.normalize(embeddings, p=2, dim=1)
    
    @torch.no_grad()
    def compute_similarity(self, emb1: Tensor, emb2: Tensor) -> Tensor:
        """Compute cosine similarity between two embeddings."""
        return F.cosine_similarity(emb1, emb2)
