"""Setup script for Prosopo package."""

from setuptools import setup, find_packages

setup(
    name="prosopo",
    version="0.1.0",
    description="Face embedding model trained from scratch using ArcFace loss",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/YOUR_USERNAME/prosopo",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "albumentations>=1.3.0",
        "scikit-learn>=1.2.0",
        "scikit-image>=0.21.0",
        "tqdm>=4.65.0",
        "facenet-pytorch>=2.5.0",
        "opencv-python>=4.8.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
