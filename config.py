"""
Configuration file for Spatial-Frequency Domain Adversarial Attack.
Author: Your Name
Date: 2025
"""

import os
from pathlib import Path
from datetime import datetime
import torch

class Config:
    """Central configuration for all experiments"""
    
    # ==================== PATHS ====================
    BASE_DIR = Path('./experiment_results')
    DATA_DIR = Path('./data')
    RESULTS_DIR = BASE_DIR / 'results'
    LOGS_DIR = BASE_DIR / 'logs'
    FIGURES_DIR = BASE_DIR / 'figures'
    MODELS_DIR = BASE_DIR / 'models'
    CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'
    
    # Create directories
    for dir_path in [RESULTS_DIR, LOGS_DIR, FIGURES_DIR, MODELS_DIR, CHECKPOINTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ==================== EXPERIMENT ID ====================
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    EXPERIMENT_ID = f"SFD_Attack_{TIMESTAMP}"
    
    # ==================== DEVICE ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_GPUS = torch.cuda.device_count()
    
    # ==================== REPRODUCIBILITY ====================
    SEED = 42
    
    # ==================== DATASETS ====================
    DATASETS = {
        'cifar10': {
            'name': 'CIFAR-10',
            'num_classes': 10,
            'image_size': (32, 32),
            'channels': 3,
            'train_samples': 50000,
            'test_samples': 10000,
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010]
        },
        'gtsrb': {
            'name': 'GTSRB',
            'num_classes': 43,
            'image_size': (32, 32),
            'channels': 3,
            'train_samples': 39209,
            'test_samples': 12630,
            'mean': [0.3403, 0.3121, 0.3214],
            'std': [0.2724, 0.2608, 0.2669]
        },
        'imagenet': {
            'name': 'ImageNet',
            'num_classes': 1000,
            'image_size': (224, 224),
            'channels': 3,
            'val_samples': 50000,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'imagenet_subset': {
            'name': 'ImageNet-100',
            'num_classes': 100,
            'image_size': (224, 224),
            'channels': 3,
            'val_samples': 5000,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    
    DEFAULT_DATASET = 'cifar10'
    
    # ==================== MODEL ARCHITECTURES ====================
    MODEL_ARCHITECTURES = {
        # CNNs
        'resnet18': {'family': 'cnn', 'pretrained': True},
        'resnet50': {'family': 'cnn', 'pretrained': True},
        'resnet101': {'family': 'cnn', 'pretrained': True},
        'densenet121': {'family': 'cnn', 'pretrained': True},
        'vgg16': {'family': 'cnn', 'pretrained': True},
        
        # Vision Transformers
        'vit_base': {'family': 'vit', 'pretrained': True},
        'vit_large': {'family': 'vit', 'pretrained': True},
        'deit_base': {'family': 'vit', 'pretrained': True},
        'swin_tiny': {'family': 'vit', 'pretrained': True},
        'swin_base': {'family': 'vit', 'pretrained': True},
        
        # MLP-based
        'mlp_mixer': {'family': 'mlp', 'pretrained': True},
        'resmlp': {'family': 'mlp', 'pretrained': True},
        
        # Hybrid
        'convnext_tiny': {'family': 'hybrid', 'pretrained': True},
        'coatnet': {'family': 'hybrid', 'pretrained': True},
    }
    
    DEFAULT_MODELS = ['resnet18', 'vit_base', 'swin_tiny', 'mlp_mixer']
    
    # ==================== ATTACK CONFIGURATIONS ====================
    ATTACK_VARIANTS = ['sequential', 'joint', 'adaptive']
    EPSILONS = [0.01, 0.03, 0.05, 0.1]
    ITERATIONS = 50
    FREQ_WEIGHTS = [0.3, 0.5, 0.7]
    MOMENTUM = 0.9
    
    # ==================== HVS CONFIGURATION ====================
    HVS_PEAK_FREQ = 4.0  # Cycles per degree
    HVS_THRESHOLD = 0.5   # Mask threshold
    
    # ==================== DEFENSES ====================
    DEFENSES = [
        # Input transformations
        {'name': 'No Defense', 'key': 'none', 'params': {}},
        {'name': 'Gaussian Blur (k=3)', 'key': 'gaussian_blur', 'params': {'kernel_size': 3, 'sigma': 1.0}},
        {'name': 'Gaussian Blur (k=5)', 'key': 'gaussian_blur', 'params': {'kernel_size': 5, 'sigma': 1.5}},
        {'name': 'Median Filter (k=3)', 'key': 'median_filter', 'params': {'kernel_size': 3}},
        {'name': 'Median Filter (k=5)', 'key': 'median_filter', 'params': {'kernel_size': 5}},
        
        # Compression
        {'name': 'JPEG (Q=50)', 'key': 'jpeg_compression', 'params': {'quality': 50}},
        {'name': 'JPEG (Q=75)', 'key': 'jpeg_compression', 'params': {'quality': 75}},
        {'name': 'JPEG (Q=90)', 'key': 'jpeg_compression', 'params': {'quality': 90}},
        
        # Quantization
        {'name': 'Bit Depth (3-bit)', 'key': 'bit_depth', 'params': {'bits': 3}},
        {'name': 'Bit Depth (4-bit)', 'key': 'bit_depth', 'params': {'bits': 4}},
        {'name': 'Bit Depth (5-bit)', 'key': 'bit_depth', 'params': {'bits': 5}},
        
        # Noise
        {'name': 'Random Noise (s=0.01)', 'key': 'random_noise', 'params': {'noise_level': 0.01}},
        {'name': 'Random Noise (s=0.02)', 'key': 'random_noise', 'params': {'noise_level': 0.02}},
        {'name': 'Random Noise (s=0.05)', 'key': 'random_noise', 'params': {'noise_level': 0.05}},
        
        # Frequency domain
        {'name': 'Low-pass Filter', 'key': 'frequency_filter', 'params': {'filter_type': 'low_pass', 'cutoff': 0.3}},
        {'name': 'High-pass Filter', 'key': 'frequency_filter', 'params': {'filter_type': 'high_pass', 'cutoff': 0.1}},
        {'name': 'Band-stop Filter', 'key': 'frequency_filter', 'params': {'filter_type': 'band_stop', 'cutoff': 0.2, 'bandwidth': 0.2}},
        
        # Adversarial training models
        {'name': 'Adversarial Training (PGD)', 'key': 'adv_trained', 'params': {'model_path': 'models/adv_pgd.pth'}},
        {'name': 'Adversarial Training (TRADES)', 'key': 'adv_trained', 'params': {'model_path': 'models/adv_trades.pth'}},
    ]
    
    # ==================== SOTA BASELINES ====================
    SOTA_BASELINES = [
        {'name': 'FGSM (2015)', 'key': 'fgsm', 'reference': 'Goodfellow et al. ICLR 2015'},
        {'name': 'PGD (2018)', 'key': 'pgd', 'reference': 'Madry et al. ICLR 2018'},
        {'name': 'SFA (2025)', 'key': 'sfa', 'reference': 'Tang et al. KBS 2025'},
        {'name': 'FACL (2024)', 'key': 'facl', 'reference': 'Yang et al. AAAI 2024'},
        {'name': 'LEA2 (2023)', 'key': 'lea2', 'reference': 'Qian et al. ICCV 2023'},
        {'name': 'Frequency-constrained (2024)', 'key': 'freq_constrained', 'reference': 'Chen et al. ICL 2024'},
    ]
    
    # ==================== PERCEPTUAL METRICS ====================
    PERCEPTUAL_METRICS = ['ssim', 'lpips', 'psnr', 'fid', 'mse', 'l2_norm']
    
    # ==================== BATCH SIZES ====================
    BATCH_SIZES = {
        'cifar10': 128,
        'imagenet': 32,
        'gtsrb': 128,
        'medical': 16,
    }
    
    # ==================== TRAINING ====================
    TRAIN_EPOCHS = 50
    TRAIN_LR = 0.001
    TRAIN_MOMENTUM = 0.9
    TRAIN_WEIGHT_DECAY = 5e-4
    
    @classmethod
    def get_experiment_path(cls):
        """Get experiment-specific path"""
        exp_path = cls.RESULTS_DIR / cls.EXPERIMENT_ID
        exp_path.mkdir(exist_ok=True)
        return exp_path
    
    @classmethod
    def print_summary(cls):
        """Print configuration summary"""
        print("="*80)
        print("SPATIAL-FREQUENCY DOMAIN ATTACK - CONFIGURATION")
        print("="*80)
        print(f"Experiment ID: {cls.EXPERIMENT_ID}")
        print(f"Device: {cls.DEVICE} ({cls.NUM_GPUS} GPUs)")
        print(f"Seed: {cls.SEED}")
        print(f"\nDatasets: {list(cls.DATASETS.keys())}")
        print(f"Default dataset: {cls.DEFAULT_DATASET}")
        print(f"\nModel architectures: {len(cls.MODEL_ARCHITECTURES)}")
        print(f"Default models: {cls.DEFAULT_MODELS}")
        print(f"\nAttack variants: {cls.ATTACK_VARIANTS}")
        print(f"Epsilons: {cls.EPSILONS}")
        print(f"\nDefenses: {len(cls.DEFENSES)}")
        print(f"\nSOTA baselines: {len(cls.SOTA_BASELINES)}")
        print("="*80)