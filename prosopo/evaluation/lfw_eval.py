"""LFW evaluation protocol."""

import os
from typing import Tuple, List
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from ..data import LFWPairs, get_val_transforms


def extract_embeddings(
    model: torch.nn.Module,
    image_paths: List[str],
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """
    Extract embeddings for all images.
    
    Returns:
        Dict mapping image_path -> embedding tensor
    """
    model.eval()
    transform = get_val_transforms()
    
    embeddings = {}
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                image = np.array(image)
                
                # Apply transform
                transformed = transform(image=image)
                batch_images.append(transformed['image'])
                valid_paths.append(path)
            except Exception:
                continue
        
        if not batch_images:
            continue
        
        # Stack and move to device
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Extract embeddings
        with torch.no_grad():
            batch_embeddings = model.get_embedding(batch_tensor)
        
        # Store embeddings
        for path, emb in zip(valid_paths, batch_embeddings):
            embeddings[path] = emb.cpu()
    
    return embeddings


def find_best_threshold(
    similarities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Find threshold that maximizes accuracy.
    
    Returns:
        (best_threshold, best_accuracy)
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.0, 0.01)
    
    best_acc = 0.0
    best_thresh = 0.0
    
    for thresh in thresholds:
        predictions = similarities >= thresh
        accuracy = np.mean(predictions == labels)
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_thresh = thresh
    
    return best_thresh, best_acc


def evaluate_lfw(
    model: torch.nn.Module,
    lfw_root: str,
    pairs_path: str,
    device: torch.device = None,
    num_folds: int = 10,
) -> Tuple[float, float]:
    """
    Evaluate model on LFW benchmark using standard protocol.
    
    Args:
        model: Prosopo model
        lfw_root: Path to LFW images directory
        pairs_path: Path to pairs.txt
        device: Torch device
        num_folds: Number of folds for cross-validation
        
    Returns:
        (accuracy, threshold)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Load pairs
    pairs = LFWPairs(pairs_path, lfw_root)
    
    # Collect all unique image paths
    all_paths = set()
    for path1, path2, _ in pairs:
        all_paths.add(path1)
        all_paths.add(path2)
    
    all_paths = list(all_paths)
    
    # Extract embeddings for all images
    embeddings = extract_embeddings(model, all_paths, device)
    
    # Compute similarities for all pairs
    similarities = []
    labels = []
    
    for path1, path2, is_same in pairs:
        if path1 not in embeddings or path2 not in embeddings:
            continue
        
        emb1 = embeddings[path1]
        emb2 = embeddings[path2]
        
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        similarities.append(similarity)
        labels.append(is_same)
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # 10-fold cross-validation
    fold_size = len(similarities) // num_folds
    accuracies = []
    thresholds = []
    
    for fold in range(num_folds):
        # Split into train and test
        test_start = fold * fold_size
        test_end = test_start + fold_size
        
        test_sim = similarities[test_start:test_end]
        test_labels = labels[test_start:test_end]
        
        train_sim = np.concatenate([similarities[:test_start], similarities[test_end:]])
        train_labels = np.concatenate([labels[:test_start], labels[test_end:]])
        
        # Find best threshold on training data
        best_thresh, _ = find_best_threshold(train_sim, train_labels)
        
        # Evaluate on test data
        predictions = test_sim >= best_thresh
        accuracy = np.mean(predictions == test_labels)
        
        accuracies.append(accuracy)
        thresholds.append(best_thresh)
    
    mean_accuracy = np.mean(accuracies)
    mean_threshold = np.mean(thresholds)
    
    return mean_accuracy, mean_threshold


def compute_roc(
    model: torch.nn.Module,
    lfw_root: str,
    pairs_path: str,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve for LFW evaluation.
    
    Returns:
        (fpr, tpr, auc_score)
    """
    if device is None:
        device = next(model.parameters()).device
    
    pairs = LFWPairs(pairs_path, lfw_root)
    
    all_paths = set()
    for path1, path2, _ in pairs:
        all_paths.add(path1)
        all_paths.add(path2)
    
    embeddings = extract_embeddings(model, list(all_paths), device)
    
    similarities = []
    labels = []
    
    for path1, path2, is_same in pairs:
        if path1 not in embeddings or path2 not in embeddings:
            continue
        
        emb1 = embeddings[path1]
        emb2 = embeddings[path2]
        
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        similarities.append(similarity)
        labels.append(is_same)
    
    fpr, tpr, _ = roc_curve(labels, similarities)
    auc_score = auc(fpr, tpr)
    
    return fpr, tpr, auc_score
