"""CASIA-WebFace dataset loader."""

import os
import json
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CASIAWebFaceDataset(Dataset):
    """
    CASIA-WebFace dataset for face recognition training.
    
    Expected directory structure (after alignment):
        data_root/
            identity_0/
                image_0.jpg
                image_1.jpg
                ...
            identity_1/
                ...
    
    The class_indices.json maps folder names to continuous integer labels.
    """
    
    def __init__(
        self,
        data_root: str,
        transform: Optional[Callable] = None,
        class_indices_path: Optional[str] = None,
    ):
        """
        Args:
            data_root: Path to aligned CASIA-WebFace directory
            transform: Albumentations transform pipeline
            class_indices_path: Path to class_indices.json (auto-generated if None)
        """
        self.data_root = Path(data_root)
        self.transform = transform
        
        # Build or load class mapping
        if class_indices_path and os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                self.class_to_idx = json.load(f)
        else:
            self.class_to_idx = self._build_class_mapping()
            
            # Save mapping for reproducibility
            if class_indices_path:
                with open(class_indices_path, 'w') as f:
                    json.dump(self.class_to_idx, f, indent=2)
        
        self.num_classes = len(self.class_to_idx)
        
        # Build image list
        self.samples = self._build_samples()
        
    def _build_class_mapping(self) -> dict:
        """Create folder_name -> continuous_index mapping."""
        folders = sorted([
            d.name for d in self.data_root.iterdir() 
            if d.is_dir()
        ])
        return {folder: idx for idx, folder in enumerate(folders)}
    
    def _build_samples(self) -> list:
        """Build list of (image_path, label) tuples."""
        samples = []
        
        for folder_name, label in self.class_to_idx.items():
            folder_path = self.data_root / folder_name
            
            if not folder_path.exists():
                continue
                
            for img_file in folder_path.iterdir():
                if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    samples.append((str(img_file), label))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            image: (3, 112, 112) tensor
            label: integer class label
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms (supports both torchvision and albumentations)
        if self.transform is not None:
            # Check if it's albumentations (expects dict with 'image' key)
            if hasattr(self.transform, 'transforms'):
                # Albumentations compose
                image = np.array(image)
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Torchvision transforms (works on PIL images directly)
                image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self) -> dict:
        """Get number of images per class."""
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts
