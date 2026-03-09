"""
Base classes for all attacks.
Author: Your Name
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np

class BaseAttack(ABC):
    """Base class for all attacks"""
    
    def __init__(self, model, epsilon=0.03, alpha=None, iterations=40, 
                 momentum=0.9, targeted=False, device=None):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha if alpha is not None else epsilon / 4
        self.iterations = iterations
        self.momentum = momentum
        self.targeted = targeted
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.model.eval()
        self.criterion = nn.CrossEntropyLoss()
        self.velocity = None
    
    def _init_velocity(self, shape):
        """Initialize velocity for momentum"""
        self.velocity = torch.zeros(shape, device=self.device)
    
    def _spatial_constraint(self, delta):
        """Apply spatial domain L-infinity constraint"""
        return torch.clamp(delta, -self.epsilon, self.epsilon)
    
    @abstractmethod
    def generate(self, images, labels):
        """Generate adversarial examples"""
        pass
    
    def attack(self, images, labels):
        """Main attack interface"""
        return self.generate(images, labels)
    
    def get_stats(self):
        """Get attack statistics"""
        return {}


class SpatialOnlyAttack(BaseAttack):
    """Base class for spatial-only attacks"""
    
    def __init__(self, model, epsilon=0.03, alpha=None, iterations=40,
                 momentum=0.9, targeted=False, device=None):
        super().__init__(model, epsilon, alpha, iterations, momentum, targeted, device)
    
    def generate(self, images, labels):
        raise NotImplementedError


class FrequencyOnlyAttack(BaseAttack):
    """Base class for frequency-only attacks"""
    
    def __init__(self, model, freq_processor, epsilon=0.03, alpha=None, 
                 iterations=40, momentum=0.9, targeted=False, device=None):
        super().__init__(model, epsilon, alpha, iterations, momentum, targeted, device)
        self.freq_processor = freq_processor
    
    def _frequency_constraint(self, delta, mask_type='low_pass'):
        """Apply frequency domain constraint"""
        mask = self.freq_processor.get_frequency_mask(mask_type=mask_type)
        delta_filtered = self.freq_processor.apply_frequency_filter(delta, mask)
        return delta_filtered
    
    def generate(self, images, labels):
        raise NotImplementedError


class HybridAttack(BaseAttack):
    """Base class for hybrid spatial-frequency attacks"""
    
    def __init__(self, model, freq_processor, epsilon=0.03, alpha=None,
                 iterations=40, freq_weight=0.5, momentum=0.9,
                 targeted=False, device=None):
        super().__init__(model, epsilon, alpha, iterations, momentum, targeted, device)
        self.freq_processor = freq_processor
        self.freq_weight = freq_weight
    
    def _frequency_constraint(self, delta, mask_type='hvs_sensitive'):
        """Apply frequency domain constraint using HVS-aware mask"""
        mask = self.freq_processor.get_frequency_mask(mask_type=mask_type)
        delta_filtered = self.freq_processor.apply_frequency_filter(delta, mask)
        return delta_filtered
    
    def _composite_constraint(self, delta):
        """Apply both spatial and frequency constraints"""
        delta_spatial = self._spatial_constraint(delta)
        delta_freq = self._frequency_constraint(delta_spatial)
        delta_combined = (1 - self.freq_weight) * delta_spatial + self.freq_weight * delta_freq
        delta_combined = self._spatial_constraint(delta_combined)
        return delta_combined
    
    def generate(self, images, labels):
        raise NotImplementedError
