"""ArcFace head for angular margin-based face recognition."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    
    Paper: https://arxiv.org/abs/1801.07698
    
    Loss = -log(exp(s * cos(θ_yi + m)) / 
                (exp(s * cos(θ_yi + m)) + Σ exp(s * cos(θ_j))))
    
    Where:
        s = scale factor (typically 64)
        m = angular margin (typically 0.5 radians = 28.6°)
        θ = angle between feature and weight vectors
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 10575,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of identity classes
            scale: Scaling factor for logits
            margin: Angular margin in radians (0.5 rad ≈ 28.6°)
            easy_margin: Use easy margin variant
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Weight matrix: each row is a class center
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute margin terms for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Threshold for numerical stability
        self.mm = math.sin(math.pi - margin) * margin  # Fallback margin
        
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """
        Compute ArcFace logits.
        
        Args:
            embeddings: (batch, embedding_dim) normalized embeddings
            labels: (batch,) integer class labels
            
        Returns:
            (batch, num_classes) scaled logits with margin applied to target class
        """
        # L2 normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity: cos(θ)
        cos_theta = F.linear(embeddings, weight)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # Numerical stability
        
        # Compute sin(θ) from cos(θ)
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))
        
        # Compute cos(θ + m) using angle addition formula:
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        # Handle edge case where θ + m > π (cos becomes non-monotonic)
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            # Use fallback when cos(θ) < cos(π - m)
            cos_theta_m = torch.where(
                cos_theta > self.th,
                cos_theta_m,
                cos_theta - self.mm
            )
        
        # Create one-hot mask for target classes
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Apply margin only to target class
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        
        # Scale logits
        logits = logits * self.scale
        
        return logits
    
    def extra_repr(self) -> str:
        return f"embedding_dim={self.embedding_dim}, num_classes={self.num_classes}, scale={self.scale}, margin={self.margin}"
