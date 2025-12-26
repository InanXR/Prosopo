<p align="center">
  <img src="assets/Prosopo_Banner_21.9.png" alt="Prosopo Dataset Banner" width="100%">
</p>

# Prosopo

> A face embedding model trained from scratch using ArcFace loss.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Results

| Benchmark | Accuracy | Threshold |
|-----------|----------|-----------|
| LFW | **99.X%** | 0.XXX |

*Results after training on CASIA-WebFace (~500K images, 10K identities)*

## 🏗️ Architecture

```
Input (112×112×3) → ResNet-50 → 512-d Embedding → ArcFace Head
```

- **Backbone:** ResNet-50 (pretrained on ImageNet)
- **Embedding:** 512-dimensional L2-normalized vectors
- **Loss:** ArcFace (s=64, m=0.5)
- **Training:** Mixed precision (FP16), gradient accumulation

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/prosopo.git
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
model = Prosopo(num_classes=10575)
model.load_state_dict(torch.load('prosopo_final.pth'))
model.eval()

# Load and transform image
transform = get_val_transforms()
image = Image.open('face.jpg').convert('RGB')
image = transform(image=np.array(image))['image'].unsqueeze(0)

# Extract embedding
with torch.no_grad():
    embedding = model.get_embedding(image)  # (1, 512)

# Compare two faces
similarity = torch.cosine_similarity(embedding1, embedding2)
is_same = similarity > 0.6  # Typical threshold
```

### Training on Colab

1. Open `notebooks/train.ipynb` in Google Colab
2. Enable GPU runtime (T4 recommended)
3. Run all cells (~8-12 hours training)

## 📁 Project Structure

```
prosopo/
├── notebooks/
│   └── train.ipynb           # Colab training notebook
├── prosopo/
│   ├── models/
│   │   ├── backbone.py       # ResNet-50 backbone
│   │   ├── arcface.py        # ArcFace head
│   │   └── prosopo.py        # Full model
│   ├── data/
│   │   ├── dataset.py        # CASIA-WebFace loader
│   │   ├── transforms.py     # Augmentations
│   │   └── lfw.py            # LFW evaluation dataset
│   ├── training/
│   │   ├── config.py         # Training configuration
│   │   └── trainer.py        # Training loop
│   └── evaluation/
│       └── lfw_eval.py       # LFW benchmark
├── scripts/
│   └── preprocess.py         # Face alignment with MTCNN
└── requirements.txt
```

## 🔧 Preprocessing

Before training, align faces using MTCNN:

```bash
python scripts/preprocess.py \
    --input /path/to/CASIA-WebFace \
    --output /path/to/aligned
```

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-50 |
| Embedding Dim | 512 |
| Batch Size | 128 (×2 accumulation = 256 effective) |
| Optimizer | SGD (lr=0.1, momentum=0.9) |
| LR Schedule | Step decay at [10, 18, 22] |
| ArcFace Scale | 64.0 |
| ArcFace Margin | 0.5 rad (28.6°) |
| Mixed Precision | FP16 |
| Epochs | 25 |

## 📚 References

- [ArcFace: Additive Angular Margin Loss](https://arxiv.org/abs/1801.07698)
- [CASIA-WebFace Dataset](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)
- [LFW Benchmark](http://vis-www.cs.umass.edu/lfw/)

## 📄 License

MIT License
