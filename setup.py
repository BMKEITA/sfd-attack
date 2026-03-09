#!/usr/bin/env python3
"""
Setup script for Spatial-Frequency Attack project.
Run: python setup.py install
"""

from setuptools import setup, find_packages

setup(
    name="sfd_attack",
    version="1.0.0",
    description="Spatial-Frequency Domain Adversarial Attack",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.2.0",
        "scikit-image>=0.18.0",
        "scikit-learn",
        "scipy>=1.6.0",
        "tqdm>=4.62.0",
        "pillow>=8.3.0",
        "opencv-python>=4.5.0",
        "lpips>=0.1.4",
        "kornia>=0.6.0",
        "timm>=0.5.4",
    ],
    python_requires=">=3.8",
)