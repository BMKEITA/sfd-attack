"""
Model architectures for evaluation.
Author: Your Name
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # This was missing!
import torchvision.models as models
from config import Config

class ModelFactory:
    """Factory for creating different model architectures"""
    
    @staticmethod
    def get_model(architecture='resnet18', num_classes=10, pretrained=True):
        """Get model by architecture name"""
        
        if architecture == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            model.fc = nn.Linear(512, num_classes)
            
        elif architecture == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            model.fc = nn.Linear(2048, num_classes)
            
        elif architecture == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
            model.fc = nn.Linear(2048, num_classes)
            
        elif architecture == 'densenet121':
            model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
            model.classifier = nn.Linear(1024, num_classes)
            
        elif architecture == 'vgg16':
            model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
            model.classifier[6] = nn.Linear(4096, num_classes)
            
        elif architecture == 'vit_base':
            try:
                import timm
                model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
                if num_classes != 1000:
                    in_features = model.head.in_features
                    model.head = nn.Linear(in_features, num_classes)
            except ImportError:
                print("timm not installed, falling back to resnet")
                model = ModelFactory.get_model('resnet50', num_classes, pretrained)
                
        elif architecture == 'vit_large':
            try:
                import timm
                model = timm.create_model('vit_large_patch16_224', pretrained=pretrained)
                if num_classes != 1000:
                    in_features = model.head.in_features
                    model.head = nn.Linear(in_features, num_classes)
            except ImportError:
                model = ModelFactory.get_model('resnet101', num_classes, pretrained)
                
        elif architecture == 'swin_tiny':
            try:
                import timm
                model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
                if num_classes != 1000:
                    in_features = model.head.in_features
                    model.head = nn.Linear(in_features, num_classes)
            except ImportError:
                model = ModelFactory.get_model('resnet50', num_classes, pretrained)
                
        elif architecture == 'swin_base':
            try:
                import timm
                model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
                if num_classes != 1000:
                    in_features = model.head.in_features
                    model.head = nn.Linear(in_features, num_classes)
            except ImportError:
                model = ModelFactory.get_model('resnet101', num_classes, pretrained)
                
        elif architecture == 'mlp_mixer':
            try:
                import timm
                model = timm.create_model('mixer_b16_224', pretrained=pretrained)
                if num_classes != 1000:
                    in_features = model.head.in_features
                    model.head = nn.Linear(in_features, num_classes)
            except ImportError:
                model = ModelFactory.get_model('resnet50', num_classes, pretrained)
                
        elif architecture == 'convnext_tiny':
            try:
                import timm
                model = timm.create_model('convnext_tiny', pretrained=pretrained)
                if num_classes != 1000:
                    if hasattr(model, 'head'):
                        in_features = model.head.fc.in_features
                        model.head.fc = nn.Linear(in_features, num_classes)
                    else:
                        in_features = model.classifier[-1].in_features
                        model.classifier[-1] = nn.Linear(in_features, num_classes)
            except ImportError:
                model = ModelFactory.get_model('resnet50', num_classes, pretrained)
                
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return model
    
    @staticmethod
    def get_family(architecture):
        """Get model family"""
        if architecture in ['resnet18', 'resnet50', 'resnet101', 'densenet121', 'vgg16']:
            return 'cnn'
        elif architecture in ['vit_base', 'vit_large', 'deit_base', 'swin_tiny', 'swin_base']:
            return 'vit'
        elif architecture in ['mlp_mixer', 'resmlp']:
            return 'mlp'
        elif architecture in ['convnext_tiny', 'coatnet']:
            return 'hybrid'
        else:
            return 'unknown'


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 (for quick experiments)"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        # Calculate the size after convolutions and pooling
        # Input: 32x32
        # After conv1+pool: 16x16
        # After conv2+pool: 8x8
        # After conv3: 8x8
        # After conv4+pool: 4x4
        # After conv5+pool: 2x2
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 -> 16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16 -> 8
        x = F.relu(self.bn3(self.conv3(x)))              # 8 -> 8
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 8 -> 4
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # 4 -> 2
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PretrainedResNet(nn.Module):
    """Wrapper for pretrained ResNet with adjustable output classes"""
    
    def __init__(self, num_classes=10, arch='resnet18'):
        super(PretrainedResNet, self).__init__()
        
        if arch == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            in_features = 512
        elif arch == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1')
            in_features = 512
        elif arch == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            in_features = 2048
        else:
            raise ValueError(f"Unknown ResNet architecture: {arch}")
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Adjust first conv layer for CIFAR-10 (32x32 images)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # Remove maxpool for 32x32
    
    def forward(self, x):
        return self.backbone(x)


def create_model(arch='simple', num_classes=10, pretrained=False):
    """Create a model by name"""
    if arch == 'simple':
        return SimpleCNN(num_classes)
    elif arch.startswith('resnet'):
        if pretrained:
            return PretrainedResNet(num_classes, arch)
        else:
            return ModelFactory.get_model(arch, num_classes, pretrained)
    else:
        return ModelFactory.get_model(arch, num_classes, pretrained)