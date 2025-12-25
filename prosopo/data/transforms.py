"""Image transforms for training and validation."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# Face-specific normalization (0.5, 0.5, 0.5)
FACE_MEAN = (0.5, 0.5, 0.5)
FACE_STD = (0.5, 0.5, 0.5)


def get_train_transforms(image_size: int = 112) -> A.Compose:
    """
    Get training augmentations.
    
    Includes:
        - Horizontal flip
        - Color jitter
        - Random erasing
        - Gaussian blur
        - Normalization
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.3
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.CoarseDropout(
            max_holes=1,
            max_height=int(image_size * 0.2),
            max_width=int(image_size * 0.2),
            min_holes=1,
            min_height=int(image_size * 0.05),
            min_width=int(image_size * 0.05),
            fill_value=128,
            p=0.2
        ),
        A.Normalize(mean=FACE_MEAN, std=FACE_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 112) -> A.Compose:
    """Get validation/inference transforms (no augmentation)."""
    return A.Compose([
        A.Normalize(mean=FACE_MEAN, std=FACE_STD),
        ToTensorV2(),
    ])
