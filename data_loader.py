"""
Data loader for multiple datasets.
Author: Your Name
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from config import Config

class DataLoaderFactory:
    """Factory class for loading multiple datasets"""
    
    def __init__(self, seed=Config.SEED):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def get_cifar10(self, train=False, batch_size=None, num_samples=None):
        """Load CIFAR-10 dataset"""
        if batch_size is None:
            batch_size = Config.BATCH_SIZES.get('cifar10', 128)
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2023, 0.1994, 0.2010])
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2023, 0.1994, 0.2010])
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root=Config.DATA_DIR,
            train=train,
            download=True,
            transform=transform_train if train else transform_test
        )
        
        if num_samples and num_samples < len(dataset):
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = Subset(dataset, indices)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=4,
            pin_memory=True
        )
        
        return loader
    
    def get_gtsrb(self, train=False, batch_size=None, num_samples=None):
        """Load GTSRB traffic sign dataset"""
        if batch_size is None:
            batch_size = Config.BATCH_SIZES.get('gtsrb', 128)
        
        # GTSRB has different transforms for train/test
        if train:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomRotation(15),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                                   std=[0.2724, 0.2608, 0.2669])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                                   std=[0.2724, 0.2608, 0.2669])
            ])
        
        try:
            # Try using torchvision's GTSRB dataset
            dataset = torchvision.datasets.GTSRB(
                root=Config.DATA_DIR,
                split='train' if train else 'test',
                download=True,
                transform=transform
            )
        except:
            # Fallback for older torchvision
            print("Torchvision GTSRB not available, using ImageFolder...")
            split = 'train' if train else 'test'
            data_path = Config.DATA_DIR / 'GTSRB' / split
            if not data_path.exists():
                print(f"Please download GTSRB to {data_path}")
                # Return CIFAR-10 as fallback
                return self.get_cifar10(train, batch_size, num_samples)
            dataset = torchvision.datasets.ImageFolder(
                root=str(data_path),
                transform=transform
            )
        
        if num_samples and num_samples < len(dataset):
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = Subset(dataset, indices)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=4,
            pin_memory=True
        )
        
        return loader
    
    def get_imagenet_subset(self, train=False, batch_size=None, num_samples=None):
        """Load ImageNet-100 subset (first 100 classes for efficiency)"""
        if batch_size is None:
            batch_size = Config.BATCH_SIZES.get('imagenet', 32)
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Path to ImageNet validation set
        split = 'train' if train else 'val'
        data_path = Config.DATA_DIR / 'imagenet' / split
        
        if not data_path.exists():
            print(f"Warning: ImageNet not found at {data_path}")
            print("Please download ImageNet and organize as:")
            print(f"  {Config.DATA_DIR}/imagenet/train/")
            print(f"  {Config.DATA_DIR}/imagenet/val/")
            print("Using CIFAR-10 as fallback...")
            return self.get_cifar10(train, batch_size, num_samples)
        
        # Load full dataset
        full_dataset = torchvision.datasets.ImageFolder(
            root=str(data_path),
            transform=transform
        )
        
        # Get first 100 classes
        classes = full_dataset.classes[:100]
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Filter to only include images from first 100 classes
        indices = []
        for idx, (_, class_idx) in enumerate(full_dataset):
            class_name = full_dataset.classes[class_idx]
            if class_name in classes:
                indices.append(idx)
        
        if num_samples:
            indices = indices[:num_samples]
        
        subset = Subset(full_dataset, indices)
        
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=4,
            pin_memory=True
        )
        
        return loader
    
    def get_imagenet_full(self, train=False, batch_size=None, num_samples=None):
        """Load full ImageNet validation set"""
        if batch_size is None:
            batch_size = Config.BATCH_SIZES.get('imagenet', 32)
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        split = 'train' if train else 'val'
        data_path = Config.DATA_DIR / 'imagenet' / split
        
        if not data_path.exists():
            print(f"Warning: ImageNet not found at {data_path}")
            return self.get_cifar10(train, batch_size, num_samples)
        
        dataset = torchvision.datasets.ImageFolder(
            root=str(data_path),
            transform=transform
        )
        
        if num_samples and num_samples < len(dataset):
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = Subset(dataset, indices)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=4,
            pin_memory=True
        )
        
        return loader
    
    def get_medical(self, train=False, batch_size=None, num_samples=None):
        """Load ISIC 2019 medical dataset (placeholder)"""
        print("Medical dataset not implemented, using CIFAR-10 fallback")
        return self.get_cifar10(train, batch_size, num_samples)
    
    def get_dataset(self, name='cifar10', train=False, batch_size=None, num_samples=None):
        """Get dataset by name"""
        if name == 'cifar10':
            return self.get_cifar10(train, batch_size, num_samples)
        elif name == 'gtsrb':
            return self.get_gtsrb(train, batch_size, num_samples)
        elif name == 'imagenet_subset':
            return self.get_imagenet_subset(train, batch_size, num_samples)
        elif name == 'imagenet':
            return self.get_imagenet_full(train, batch_size, num_samples)
        elif name == 'medical':
            return self.get_medical(train, batch_size, num_samples)
        else:
            raise ValueError(f"Unknown dataset: {name}")
    
    def get_sample_batch(self, dataset='cifar10', num_samples=20):
        """Get a fixed sample batch for consistent evaluation"""
        loader = self.get_dataset(dataset, train=False, batch_size=num_samples)
        images, labels = next(iter(loader))
        return images, labels


class DatasetStats:
    """Compute dataset statistics"""
    
    @staticmethod
    def compute_mean_std(loader):
        """Compute mean and std of dataset"""
        mean = 0.
        std = 0.
        total = 0
        
        for images, _ in loader:
            batch = images.size(0)
            images = images.view(batch, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total += batch
        
        mean /= total
        std /= total
        
        return mean, std