"""Backbone networks for feature extraction.
Implements ResNet-IR-SE-50 (InsightFace style).
"""

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module


class SEBlock(Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = Sequential(
            Linear(channel, channel // reduction, bias=False),
            PReLU(),
            Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class IRBlock(Module):
    """ResNet-IR block with optional SE and Stochastic Depth."""
    def __init__(self, in_channel, out_channel, stride, use_se=True, drop_path=0.0):
        super(IRBlock, self).__init__()
        if in_channel == out_channel:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, out_channel, (1, 1), stride, bias=False),
                BatchNorm2d(out_channel)
            )
            
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, out_channel, (3, 3), (1, 1), 1, bias=False),
            PReLU(out_channel),
            Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
            BatchNorm2d(out_channel)
        )
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channel)
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        if self.use_se:
            res = self.se(res)
        return self.drop_path(res) + shortcut


class ResNetIRSE(Module):
    """
    ResNet-IR-SE-50 backbone.
    Improved ResNet with Pre-Activation, PReLU, and SE blocks.
    Ideal for Face Recognition.
    """
    def __init__(self, num_layers=50, dropout=0.0, output_feature_dim=512):
        super(ResNetIRSE, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        
        if num_layers == 50:
            layers = [3, 4, 14, 3]
        elif num_layers == 100:
            layers = [3, 13, 30, 3]
        elif num_layers == 152:
            layers = [3, 8, 36, 3]

        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(64),
            PReLU(64)
        )
        
        modules = []
        # Stacking layers
        filter_list = [64, 64, 128, 256, 512]
        for i in range(4):
            modules.append(self._make_layer(
                filter_list[i], 
                filter_list[i + 1], 
                layers[i], 
                stride=2
            ))
            
        self.body = Sequential(*modules)
        
        self.output_layer = Sequential(
            BatchNorm2d(512),
            nn.Dropout(dropout),
            nn.Flatten(),
            Linear(512 * 7 * 7, output_feature_dim),
            BatchNorm1d(output_feature_dim)
        )
        
        self._init_weights()

    def _make_layer(self, in_channel, out_channel, blocks, stride):
        layers = []
        layers.append(IRBlock(in_channel, out_channel, stride, use_se=True))
        for _ in range(1, blocks):
            layers.append(IRBlock(out_channel, out_channel, 1, use_se=True))
        return Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


class BackboneWithEmbedding(nn.Module):
    """
    Backbone compatibility wrapper.
    Matches the checks in Prosopo model init.
    """
    def __init__(
        self,
        backbone_name: str = "resnet50",
        embedding_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        
        print(f"Initializing Backbone: {backbone_name}")
        
        if "100" in backbone_name:
            depth = 100
        elif "152" in backbone_name:
            depth = 152
        else:
            depth = 50
            
        print(f"  Architecture: ResNet-IR-SE-{depth}")
        self.backbone = ResNetIRSE(num_layers=depth, output_feature_dim=embedding_dim)
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        return self.backbone(x)
