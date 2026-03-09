"""
Utility functions for the project.
Author: Your Name
"""

import os
import json
import pickle
import numpy as np
import torch
import random
from pathlib import Path
from datetime import datetime

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data, path):
    """Save data to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    """Load data from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)

def save_pickle(data, path):
    """Save data to pickle file"""
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    """Load data from pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressBar:
    """Simple progress bar"""
    
    def __init__(self, total, prefix='', length=50):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
    
    def update(self, n=1):
        self.current += n
        percent = self.current / self.total
        filled = int(self.length * percent)
        bar = '¦' * filled + '¦' * (self.length - filled)
        print(f'\r{self.prefix} |{bar}| {percent:.1%}', end='', flush=True)
        
        if self.current >= self.total:
            print()


def count_parameters(model):
    """Count number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model, input_size=(1, 3, 32, 32)):
    """Compute FLOPs for model (simplified)"""
    try:
        from thop import profile
        input = torch.randn(input_size)
        flops, params = profile(model, inputs=(input,), verbose=False)
        return flops / 1e9  # GFLOPs
    except ImportError:
        print("thop not installed. Install with: pip install thop")
        return 0


def normalize_image(img):
    """Normalize image to [0,1] range"""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 0:
        return (img - img_min) / (img_max - img_min)
    return img


def denormalize_image(img, mean, std):
    """Denormalize image with given mean and std"""
    mean = torch.tensor(mean).view(-1, 1, 1).to(img.device)
    std = torch.tensor(std).view(-1, 1, 1).to(img.device)
    return img * std + mean


def clip_image(img, min_val=0, max_val=1):
    """Clip image to valid range"""
    return torch.clamp(img, min_val, max_val)


def compute_gradient_norm(model):
    """Compute gradient norm for model"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')