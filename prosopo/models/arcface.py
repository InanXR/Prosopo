"""AdaFace and Sub-center ArcFace heads."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaFaceHead(nn.Module):
    """
    AdaFace: Quality Adaptive Margin for Face Recognition.
    """
    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 10575,
        scale: float = 64.0,
        margin: float = 0.5,
        h: float = 0.333,
        t_alpha: float = 0.01,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.h = h
        self.t_alpha = t_alpha
        
        # Class centers
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.normal_(self.weight, std=0.01)
        
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        # Norms
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        emb_norm = torch.norm(embeddings, p=2, dim=1)
        embeddings_norm = embeddings / (emb_norm.unsqueeze(1) + 1e-10)
        
        # Cosine
        logits = F.linear(embeddings_norm, weight_norm)
        
        # Adaptive Margin
        batch_norm = emb_norm.clamp(max=100)
        norm_ratio = batch_norm / (torch.mean(batch_norm).detach() + 1e-5)
        m_adaptive = self.margin * norm_ratio.clamp(min=0.2, max=1.2)
        
        # Apply margin
        one_hot = F.one_hot(labels, num_classes=self.num_classes)
        target_cosine = logits[one_hot.bool()]
        
        theta = torch.acos(target_cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        target_logits_margins = torch.cos(theta + m_adaptive)
        
        diff = target_logits_margins - target_cosine
        logits = logits + one_hot * diff.unsqueeze(1)
        
        return logits * self.scale


class SubCenterArcFaceHead(nn.Module):
    """
    Sub-center ArcFace: ArcFace with K sub-centers per class.
    
    Robust to noisy labels and large intra-class variation.
    Uses max pooling over K sub-centers to pick the best matching subclass center.
    """
    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 10575,
        scale: float = 64.0,
        margin: float = 0.5,
        sub_centers: int = 3,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.sub_centers = sub_centers
        self.easy_margin = easy_margin
        
        # Weights: (num_classes * sub_centers, embedding_dim)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * sub_centers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine: (batch, num_classes * sub_centers)
        cosine = F.linear(embeddings, weight)
        
        # Reshape to (batch, num_classes, sub_centers)
        cosine = cosine.view(-1, self.num_classes, self.sub_centers)
        
        # Max pool over sub-centers -> (batch, num_classes)
        cosine, _ = torch.max(cosine, dim=2)
        
        # Standard ArcFace on the max-pooled logits
        return self._apply_margin(cosine, labels)
        
    def _apply_margin(self, cosine: Tensor, labels: Tensor) -> Tensor:
        # 1. Get theta
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = cosine.acos()
        
        # 2. Add margin to target
        one_hot = F.one_hot(labels, num_classes=self.num_classes)
        
        # cos(theta + m)
        target_logits = torch.cos(theta + self.margin)
        
        # 3. Combine
        diff = target_logits - cosine
        cosine = cosine + one_hot * diff
        
        # 4. Scale
        return cosine * self.scale
