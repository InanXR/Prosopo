<p align="center">
  <img src="assets/Prosopo_Banner_21.9.png" alt="Prosopo Dataset Banner" width="100%">
</p>

# Prosopo

> State-of-the-art face embedding model using ResNet-IR-SE-100 and AdaFace. Optimized for 224x224 input and Kaggle environments.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 🎯 Features

*   **Backbone**: ResNet-IR-SE-100 (Deep 100-layer network with Squeeze-and-Excitation).
*   **Input**: High-Resolution 224x224 face crops.
*   **Loss**: AdaFace (Adaptive Margin).
*   **Regularization**: Model EMA, Stochastic Depth (DropPath), Gradient Clipping, Label Smoothing.
*   **Augmentation**: RandAugment (Shear, Rotate, ColorJitter, Solarize, etc.).
*   **Inference**: Test-Time Augmentation (TTA) & Model Ensembling.

## 🏗️ Architecture

```
Input (224×224×3) → ResNet-100 (IR-SE) → 512-d Embedding → AdaFace Head
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/prosopo/prosopo.git
cd prosopo
pip install -r requirements.txt
```

### Inference

```python
import torch
from PIL import Image
import numpy as np
from prosopo.models import Prosopo
from prosopo.data import get_val_transforms

# Load model
model = Prosopo(backbone='resnet100', num_classes=10575, loss_type='adaface') # Use loss_type='adaface'
model.load_state_dict(torch.load('prosopo_final.pth'))
model.eval()

# Load and transform image
transform = get_val_transforms(image_size=224)
image = Image.open('face.jpg').convert('RGB')
image = transform(image=np.array(image))['image'].unsqueeze(0)

# Extract embedding (with TTA)
with torch.no_grad():
    embedding = model.get_embedding(image, tta=True)  # (1, 512)
```

## 📁 Project Structure

```
prosopo/
├── prosopo/
│   ├── models/
│   │   ├── backbone.py       # ResNet-IR-SE (50, 100, 152) + DropPath
│   │   ├── arcface.py        # AdaFace & SubCenter ArcFace
│   │   ├── ensemble.py       # Model Ensembling
│   │   └── prosopo.py        # Main Model Wrapper
│   ├── data/
│   │   ├── dataset.py        # Dataset Loader
│   │   └── transforms.py     # RandAugment (Albumentations)
│   ├── training/
│   │   ├── config.py         # Hyperparameters
│   │   ├── trainer.py        # Training Loop (AMP, EMA, Clipping)
│   │   └── utils.py          # Model EMA
├── scripts/
│   ├── preprocess.py         # Distributed Preprocessing
│   └── generate_commands.py  # Helper to launch shards
```

## 🔧 Preprocessing (Distributed)

To process massive datasets (e.g., MS1M) on Kaggle, use sharding:

```bash
# Generate commands for 5 separate workers
python scripts/generate_commands.py --shards 5
```

Worker 1 runs:
```bash
python scripts/preprocess.py --input /path/to/raw --output /path/to/aligned --num-shards 5 --shard-id 0
```

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | **ResNet-IR-SE-100** |
| Input Size | **224x224** |
| Embedding | 512-d |
| Batch Size | 128 (P100) or 64 (if OOM) |
| Optimizer | **AdamW** (lr=1e-3, wd=5e-4) |
| Loss | **AdaFace** |
| EMA Decay | 0.999 |
| Augmentation | **RandAugment** |
| Precision | Mixed (FP16) |

## 📚 References

- [ArcFace: Additive Angular Margin Loss](https://arxiv.org/abs/1801.07698)
- [AdaFace: Quality Adaptive Margin](https://arxiv.org/abs/2204.00964)
- [Sub-center ArcFace](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)
- [InsightFace](https://github.com/deepinsight/insightface)

## 📄 License

Apache License 2.0
