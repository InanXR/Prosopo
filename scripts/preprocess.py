#!/usr/bin/env python3
"""
Preprocess CASIA-WebFace dataset: detect faces, align, and save.

This script uses MTCNN to detect faces and aligns them to a canonical
112x112 format suitable for training.

Usage:
    python preprocess.py --input /path/to/CASIA-WebFace --output /path/to/aligned

Run this ONCE before training. Takes ~2-4 hours for the full dataset.
"""

import argparse
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Lazy import to allow CPU-only environments
MTCNN = None
DEVICE = None


def lazy_import_mtcnn():
    """Import MTCNN only when needed."""
    global MTCNN, DEVICE
    if MTCNN is None:
        import torch
        from facenet_pytorch import MTCNN as _MTCNN
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        MTCNN = _MTCNN(keep_all=False, device=DEVICE, post_process=False)
    return MTCNN


# Target landmarks for 112x112 aligned face
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # Left eye
    [73.5318, 51.5014],   # Right eye
    [56.0252, 71.7366],   # Nose tip
    [41.5493, 92.3655],   # Left mouth corner
    [70.7299, 92.2041],   # Right mouth corner
], dtype=np.float32)


def estimate_transform(src_landmarks: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transform from source to reference landmarks.
    
    Returns:
        2x3 affine transformation matrix
    """
    from skimage.transform import SimilarityTransform
    
    tform = SimilarityTransform()
    tform.estimate(src_landmarks, REFERENCE_LANDMARKS)
    
    return tform.params[:2]  # 2x3 matrix


def align_face(image: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Align face to canonical 112x112 pose.
    
    Args:
        image: BGR image
        landmarks: 5x2 array of facial landmarks
        
    Returns:
        112x112 aligned face or None if alignment fails
    """
    if landmarks is None or len(landmarks) != 5:
        return None
    
    try:
        transform_matrix = estimate_transform(landmarks)
        aligned = cv2.warpAffine(
            image,
            transform_matrix,
            (112, 112),
            borderMode=cv2.BORDER_REPLICATE
        )
        return aligned
    except Exception:
        return None


def process_image(
    input_path: str,
    output_path: str,
    mtcnn,
) -> bool:
    """
    Detect face, align, and save.
    
    Returns:
        True if face was successfully aligned and saved
    """
    try:
        # Load image
        image = Image.open(input_path).convert('RGB')
        image_np = np.array(image)
        
        # Detect face and landmarks
        boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
        
        if landmarks is None or len(landmarks) == 0:
            return False
        
        # Use first detected face
        face_landmarks = landmarks[0]  # 5x2 array
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Align face
        aligned = align_face(image_bgr, face_landmarks)
        
        if aligned is None:
            return False
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save aligned face (convert back to RGB)
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        Image.fromarray(aligned_rgb).save(output_path, quality=95)
        
        return True
        
    except Exception as e:
        return False


def process_identity_folder(
    identity_path: Path,
    output_root: Path,
    mtcnn,
) -> tuple[int, int]:
    """
    Process all images for one identity.
    
    Returns:
        (success_count, total_count)
    """
    identity_name = identity_path.name
    output_folder = output_root / identity_name
    
    success_count = 0
    total_count = 0
    
    for img_file in identity_path.iterdir():
        if img_file.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
            
        total_count += 1
        output_path = output_folder / img_file.name
        
        if process_image(str(img_file), str(output_path), mtcnn):
            success_count += 1
    
    return success_count, total_count


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess CASIA-WebFace dataset with face alignment'
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Path to raw CASIA-WebFace directory'
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help='Path for aligned output directory'
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip identities that already have output folders'
    )
    
    args = parser.parse_args()
    
    input_root = Path(args.input)
    output_root = Path(args.output)
    
    if not input_root.exists():
        print(f"Error: Input directory not found: {input_root}")
        return
    
    # Initialize MTCNN
    print("Initializing MTCNN face detector...")
    mtcnn = lazy_import_mtcnn()
    print(f"Using device: {DEVICE}")
    
    # Get all identity folders
    identity_folders = sorted([
        d for d in input_root.iterdir()
        if d.is_dir()
    ])
    
    print(f"Found {len(identity_folders)} identities")
    
    # Process each identity
    total_success = 0
    total_images = 0
    
    for identity_path in tqdm(identity_folders, desc="Processing identities"):
        if args.skip_existing:
            output_folder = output_root / identity_path.name
            if output_folder.exists() and any(output_folder.iterdir()):
                continue
        
        success, total = process_identity_folder(identity_path, output_root, mtcnn)
        total_success += success
        total_images += total
    
    # Build class mapping
    print("\nBuilding class indices...")
    class_indices = {}
    for idx, folder in enumerate(sorted(output_root.iterdir())):
        if folder.is_dir():
            class_indices[folder.name] = idx
    
    # Save class mapping
    indices_path = output_root / "class_indices.json"
    with open(indices_path, 'w') as f:
        json.dump(class_indices, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"  Aligned: {total_success}/{total_images} images ({total_success/total_images*100:.1f}%)")
    print(f"  Classes: {len(class_indices)}")
    print(f"  Class indices saved to: {indices_path}")


if __name__ == '__main__':
    main()
