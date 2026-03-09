"""
State-of-the-art baseline attacks from literature.
Author: Your Name
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this if missing
from attacks.base import SpatialOnlyAttack, FrequencyOnlyAttack

class FGSM(SpatialOnlyAttack):
    """Fast Gradient Sign Method (Goodfellow et al. ICLR 2015)"""
    
    def __init__(self, model, epsilon=0.03, targeted=False, device=None):
        super().__init__(model, epsilon, alpha=None, iterations=1, 
                        momentum=0, targeted=targeted, device=device)
        self.variant_name = "FGSM"
    
    def generate(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        images.requires_grad = True
        
        outputs = self.model(images)
        if self.targeted:
            loss = -self.criterion(outputs, labels)
        else:
            loss = self.criterion(outputs, labels)
        
        # Zero gradients before backward
        self.model.zero_grad()
        if images.grad is not None:
            images.grad.zero_()
        
        loss.backward()
        
        with torch.no_grad():
            if self.targeted:
                delta = -self.epsilon * torch.sign(images.grad.data)
            else:
                delta = self.epsilon * torch.sign(images.grad.data)
            
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, 0, 1)
            
            # Get predictions for stats
            adv_outputs = self.model(adv_images)
            adv_preds = torch.argmax(adv_outputs, dim=1)
            success = (adv_preds != labels).float().mean().item()
        
        stats = {
            'success_rate': [success],
            'loss_history': [loss.item()],
            'perturbation_norms': [torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1).mean().item()]
        }
        
        return adv_images, stats

class PGD(SpatialOnlyAttack):
    """Projected Gradient Descent (Madry et al. ICLR 2018)"""
    
    def __init__(self, model, epsilon=0.03, alpha=None, iterations=40,
                 random_start=True, targeted=False, device=None):
        super().__init__(model, epsilon, alpha, iterations, momentum=0,
                        targeted=targeted, device=device)
        self.variant_name = "PGD"
        self.random_start = random_start
    
    def generate(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        if self.random_start:
            delta = torch.rand_like(images) * 2 * self.epsilon - self.epsilon
        else:
            delta = torch.zeros_like(images)
        
        delta = delta.to(self.device)
        delta.requires_grad = True
        
        stats = {
            'success_rate': [],
            'loss_history': [],
            'perturbation_norms': []
        }
        
        for i in range(self.iterations):
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, 0, 1)
            
            outputs = self.model(adv_images)
            if self.targeted:
                loss = -self.criterion(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
            
            self.model.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            
            loss.backward()
            
            with torch.no_grad():
                delta.data = delta.data + self.alpha * torch.sign(delta.grad.data)
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                delta.data = torch.clamp(images + delta.data, 0, 1) - images
            
            # Record stats
            with torch.no_grad():
                adv_outputs = self.model(adv_images)
                adv_preds = torch.argmax(adv_outputs, dim=1)
                success = (adv_preds != labels).float().mean().item()
                
                stats['success_rate'].append(success)
                stats['loss_history'].append(loss.item())
                
                pert_norm = torch.norm(delta.data.view(delta.data.shape[0], -1), p=2, dim=1).mean().item()
                stats['perturbation_norms'].append(pert_norm)
        
        adv_images = images + delta.data
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images, stats

class SFA(FrequencyOnlyAttack):
    """
    SFA: Spatial-Frequency Adversarial Attack (Tang et al. KBS 2025)
    """
    
    def __init__(self, model, freq_processor, epsilon=0.03, alpha=None,
                 iterations=40, targeted=False, device=None):
        super().__init__(model, freq_processor, epsilon, alpha, iterations,
                        momentum=0.9, targeted=targeted, device=device)
        self.variant_name = "SFA (2025)"
    
    def generate(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        delta = torch.zeros_like(images, requires_grad=True, device=self.device)
        self._init_velocity(delta.shape)
        
        stats = {
            'success_rate': [],
            'loss_history': [],
            'perturbation_norms': []
        }
        
        for i in range(self.iterations):
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, 0, 1)
            
            outputs = self.model(adv_images)
            if self.targeted:
                loss = -self.criterion(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
            
            self.model.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            
            with torch.no_grad():
                self.velocity = self.momentum * self.velocity + delta.grad.data
                
                # Apply frequency filter
                delta_freq = self._frequency_constraint(self.velocity, 'band_pass')
                
                delta.data = delta.data + self.alpha * torch.sign(delta_freq)
                delta.data = self._spatial_constraint(delta.data)
                
                adv_images = images + delta.data
                adv_images = torch.clamp(adv_images, 0, 1)
                delta.data = adv_images - images
            
            # Record stats
            with torch.no_grad():
                adv_outputs = self.model(adv_images)
                adv_preds = torch.argmax(adv_outputs, dim=1)
                success = (adv_preds != labels).float().mean().item()
                
                stats['success_rate'].append(success)
                stats['loss_history'].append(loss.item())
                
                pert_norm = torch.norm(delta.data.view(delta.data.shape[0], -1), p=2, dim=1).mean().item()
                stats['perturbation_norms'].append(pert_norm)
        
        adv_images = images + delta.data
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images, stats

class FACL(FrequencyOnlyAttack):
    """
    FACL-Attack: Frequency-Aware Contrastive Learning (Yang et al. AAAI 2024)
    """
    
    def __init__(self, model, freq_processor, epsilon=0.03, alpha=None,
                 iterations=40, targeted=False, device=None):
        super().__init__(model, freq_processor, epsilon, alpha, iterations,
                        momentum=0.9, targeted=targeted, device=device)
        self.variant_name = "FACL (2024)"
    
    def generate(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        delta = torch.zeros_like(images, requires_grad=True, device=self.device)
        
        stats = {
            'success_rate': [],
            'loss_history': [],
            'perturbation_norms': []
        }
        
        for i in range(self.iterations):
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, 0, 1)
            
            outputs = self.model(adv_images)
            if self.targeted:
                loss = -self.criterion(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
            
            self.model.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            
            with torch.no_grad():
                # FACL uses frequency-aware gradient normalization
                delta_fft = self.freq_processor.spatial_to_fft(delta.grad.data)
                magnitude = torch.abs(delta_fft)
                magnitude_norm = magnitude / (magnitude.mean() + 1e-8)
                
                delta_freq = self.freq_processor.fft_to_spatial(delta_fft * magnitude_norm)
                
                delta.data = delta.data + self.alpha * torch.sign(delta_freq)
                delta.data = self._spatial_constraint(delta.data)
                
                adv_images = images + delta.data
                adv_images = torch.clamp(adv_images, 0, 1)
                delta.data = adv_images - images
            
            # Record stats
            with torch.no_grad():
                adv_outputs = self.model(adv_images)
                adv_preds = torch.argmax(adv_outputs, dim=1)
                success = (adv_preds != labels).float().mean().item()
                
                stats['success_rate'].append(success)
                stats['loss_history'].append(loss.item())
                
                pert_norm = torch.norm(delta.data.view(delta.data.shape[0], -1), p=2, dim=1).mean().item()
                stats['perturbation_norms'].append(pert_norm)
        
        adv_images = images + delta.data
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images, stats
