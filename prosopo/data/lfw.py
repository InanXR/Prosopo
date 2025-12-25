"""LFW (Labeled Faces in the Wild) dataset for evaluation."""

import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LFWDataset(Dataset):
    """
    LFW dataset for extracting embeddings.
    
    Expected structure:
        lfw_root/
            Person_Name/
                Person_Name_0001.jpg
                Person_Name_0002.jpg
                ...
    """
    
    def __init__(
        self,
        lfw_root: str,
        transform: Optional[Callable] = None,
    ):
        self.lfw_root = Path(lfw_root)
        self.transform = transform
        
        # Build image list
        self.samples = []
        self.name_to_images = {}
        
        for person_dir in sorted(self.lfw_root.iterdir()):
            if not person_dir.is_dir():
                continue
                
            person_name = person_dir.name
            self.name_to_images[person_name] = []
            
            for img_file in sorted(person_dir.iterdir()):
                if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    self.samples.append((str(img_file), person_name))
                    self.name_to_images[person_name].append(str(img_file))
        
        # Create path to index mapping
        self.path_to_idx = {path: idx for idx, (path, _) in enumerate(self.samples)}
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple:
        img_path, person_name = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, person_name, img_path
    
    def get_image_path(self, person_name: str, image_num: int) -> str:
        """Get path for Person_Name_XXXX.jpg format."""
        # LFW format: Person_Name_0001.jpg
        filename = f"{person_name}_{image_num:04d}.jpg"
        return str(self.lfw_root / person_name / filename)


class LFWPairs:
    """
    LFW pairs for verification evaluation.
    
    Parses pairs.txt which contains:
        - 3000 positive pairs (same person)
        - 3000 negative pairs (different people)
    """
    
    def __init__(self, pairs_path: str, lfw_root: str):
        """
        Args:
            pairs_path: Path to pairs.txt
            lfw_root: Path to LFW images directory
        """
        self.pairs_path = pairs_path
        self.lfw_root = Path(lfw_root)
        
        self.pairs = self._parse_pairs()
        
    def _parse_pairs(self) -> List[Tuple[str, str, int]]:
        """
        Parse pairs.txt file.
        
        Returns:
            List of (path1, path2, is_same) tuples
        """
        pairs = []
        
        with open(self.pairs_path, 'r') as f:
            lines = f.readlines()
        
        # First line is header: num_folds \t pairs_per_fold
        header = lines[0].strip().split()
        
        for line in lines[1:]:
            parts = line.strip().split('\t')
            
            if len(parts) == 3:
                # Positive pair: Name \t img1_num \t img2_num
                name, num1, num2 = parts
                path1 = self._get_image_path(name, int(num1))
                path2 = self._get_image_path(name, int(num2))
                pairs.append((path1, path2, 1))  # Same person
                
            elif len(parts) == 4:
                # Negative pair: Name1 \t img1_num \t Name2 \t img2_num
                name1, num1, name2, num2 = parts
                path1 = self._get_image_path(name1, int(num1))
                path2 = self._get_image_path(name2, int(num2))
                pairs.append((path1, path2, 0))  # Different people
        
        return pairs
    
    def _get_image_path(self, name: str, num: int) -> str:
        """Get full path for a face image."""
        filename = f"{name}_{num:04d}.jpg"
        return str(self.lfw_root / name / filename)
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        """Returns (path1, path2, is_same)."""
        return self.pairs[idx]
    
    def get_fold(self, fold_idx: int, num_folds: int = 10) -> Tuple[List, List]:
        """
        Get train/test split for k-fold cross-validation.
        
        Args:
            fold_idx: Fold index (0 to num_folds-1)
            num_folds: Total number of folds
            
        Returns:
            (train_pairs, test_pairs)
        """
        fold_size = len(self.pairs) // num_folds
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size
        
        test_pairs = self.pairs[test_start:test_end]
        train_pairs = self.pairs[:test_start] + self.pairs[test_end:]
        
        return train_pairs, test_pairs
