
"""
Your three spatial-frequency hybrid attack variants.
Author: Your Name
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from attacks.base import HybridAttack

class SequentialHybridAttack(HybridAttack):
    """
    Variant A: Sequential Filtering & Injection
    Steps:
    1. Generate spatial perturbation (PGD step)
    2. Apply frequency filter
    3. Combine and project
    """
    
    def __init__(self, model, freq_processor, epsilon=0.03, alpha=None,
                 iterations=40, freq_weight=0.5, momentum=0.9,
                 targeted=False, device=None):
        super().__init__(model, freq_processor, epsilon, alpha, iterations,
                        freq_weight, momentum, targeted, device)
        self.variant_name = "Sequential"
    
    def generate(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        delta = torch.zeros_like(images, requires_grad=True, device=self.device)
        self._init_velocity(delta.shape)
        
        stats = {
            'success_rate': [],
            'loss_history': [],
            'perturbation_norms': [],
            'cls_loss': [],
            'iteration_times': []
        }
        
        for i in range(self.iterations):
            iter_start = time.time()
            
            # Forward pass
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, 0, 1)
            
            outputs = self.model(adv_images)
            if self.targeted:
                loss = -self.criterion(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
            
            # Compute gradient
            self.model.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            
            with torch.no_grad():
                # Update with momentum
                self.velocity = self.momentum * self.velocity + delta.grad.data
                
                # Spatial update
                delta_spatial = delta.data + self.alpha * torch.sign(self.velocity)
                delta_spatial = self._spatial_constraint(delta_spatial)
                
                # Apply frequency filter
                delta_freq = self._frequency_constraint(delta_spatial, 'hvs_sensitive')
                
                # Combine
                delta.data = (1 - self.freq_weight) * delta_spatial + self.freq_weight * delta_freq
                delta.data = self._spatial_constraint(delta.data)
                
                # Project
                adv_images = images + delta.data
                adv_images = torch.clamp(adv_images, 0, 1)
                delta.data = adv_images - images
            
            # Record stats
            with torch.no_grad():
                pred = torch.argmax(self.model(adv_images), dim=1)
                if self.targeted:
                    success = (pred == labels).float().mean()
                else:
                    success = (pred != labels).float().mean()
                
                stats['success_rate'].append(success.item())
                stats['loss_history'].append(loss.item())
                stats['cls_loss'].append(loss.item())
                
                pert_norm = torch.norm(delta.data.view(delta.data.shape[0], -1), p=2, dim=1).mean()
                stats['perturbation_norms'].append(pert_norm.item())
                stats['iteration_times'].append(time.time() - iter_start)
        
        adv_images = images + delta.data
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images, stats


class JointHybridAttack(HybridAttack):
    """
    Variant B: Joint Hybrid Loss Optimization
    Optimizes a composite loss with spatial, frequency, and HVS components simultaneously
    """
    
    def __init__(self, model, freq_processor, epsilon=0.03, alpha=None,
                 iterations=40, freq_weight=0.5, momentum=0.9,
                 targeted=False, device=None):
        super().__init__(model, freq_processor, epsilon, alpha, iterations,
                        freq_weight, momentum, targeted, device)
        self.variant_name = "Joint Hybrid"
    
    def _compute_loss(self, adv_images, targets, original_images=None, delta=None):
        """Compute composite loss with spatial, frequency, and HVS components"""
        
        outputs = self.model(adv_images)
        if self.targeted:
            cls_loss = -self.criterion(outputs, targets)
        else:
            cls_loss = self.criterion(outputs, targets)
        
        # Spatial perturbation loss
        if original_images is not None and delta is not None:
            spatial_loss = torch.norm(delta.view(delta.size(0), -1), p=float('inf'), dim=1).mean()
        else:
            spatial_loss = torch.tensor(0.0, device=self.device)
        
        # HVS-aware frequency loss
        if delta is not None:
            delta_fft = self.freq_processor.spatial_to_fft(delta)
            magnitude = torch.abs(delta_fft)
            
            hvs_weight = self.freq_processor.hvs_insensitivity.unsqueeze(0).unsqueeze(0)
            freq_loss = torch.mean(magnitude * hvs_weight)
            
            sensitive_penalty = torch.mean(
                magnitude * self.freq_processor.hvs_sensitivity.unsqueeze(0).unsqueeze(0)
            )
        else:
            freq_loss = torch.tensor(0.0, device=self.device)
            sensitive_penalty = torch.tensor(0.0, device=self.device)
        
        total_loss = cls_loss + 0.1 * spatial_loss + self.freq_weight * freq_loss - 0.05 * sensitive_penalty
        
        loss_components = {
            'cls_loss': cls_loss.item(),
            'spatial_loss': spatial_loss.item(),
            'freq_loss': freq_loss.item(),
            'sensitive_penalty': sensitive_penalty.item() if isinstance(sensitive_penalty, torch.Tensor) else 0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def generate(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        delta = torch.zeros_like(images, requires_grad=True, device=self.device)
        self._init_velocity(delta.shape)
        
        stats = {
            'success_rate': [],
            'loss_history': [],
            'perturbation_norms': [],
            'cls_loss': [],
            'freq_loss': [],
            'spatial_loss': [],
            'sensitive_penalty': [],
            'iteration_times': []
        }
        
        for i in range(self.iterations):
            iter_start = time.time()
            
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, 0, 1)
            
            total_loss, loss_components = self._compute_loss(adv_images, labels, images, delta)
            
            self.model.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            total_loss.backward()
            
            with torch.no_grad():
                self.velocity = self.momentum * self.velocity + delta.grad.data
                delta.data = delta.data + self.alpha * torch.sign(self.velocity)
                delta.data = self._composite_constraint(delta.data)
                
                adv_images = images + delta.data
                adv_images = torch.clamp(adv_images, 0, 1)
                delta.data = adv_images - images
            
            # Record stats
            with torch.no_grad():
                pred = torch.argmax(self.model(adv_images), dim=1)
                if self.targeted:
                    success = (pred == labels).float().mean()
                else:
                    success = (pred != labels).float().mean()
                
                stats['success_rate'].append(success.item())
                stats['loss_history'].append(total_loss.item())
                stats['cls_loss'].append(loss_components['cls_loss'])
                stats['freq_loss'].append(loss_components['freq_loss'])
                stats['spatial_loss'].append(loss_components['spatial_loss'])
                stats['sensitive_penalty'].append(loss_components['sensitive_penalty'])
                
                pert_norm = torch.norm(delta.data.view(delta.data.shape[0], -1), p=2, dim=1).mean()
                stats['perturbation_norms'].append(pert_norm.item())
                stats['iteration_times'].append(time.time() - iter_start)
        
        adv_images = images + delta.data
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images, stats


class AdaptiveBandAttack(HybridAttack):
    """
    Variant C: Adaptive Frequency Band Selection
    Steps:
    1. Analyze frequency sensitivity of the model
    2. Select most vulnerable frequency bands
    3. Focus attack on selected bands
    """
    
    def __init__(self, model, freq_processor, epsilon=0.03, alpha=None,
                 iterations=40, freq_weight=0.5, momentum=0.9,
                 num_bands=5, top_k=2, targeted=False, device=None):
        super().__init__(model, freq_processor, epsilon, alpha, iterations,
                        freq_weight, momentum, targeted, device)
        self.variant_name = "Adaptive Band"
        self.num_bands = num_bands
        self.top_k = top_k
    
    def _analyze_frequency_sensitivity(self, images, labels):
        """Analyze model sensitivity to different frequency bands"""
        sensitivities = []
        
        with torch.no_grad():
            clean_outputs = self.model(images)
            clean_preds = torch.argmax(clean_outputs, dim=1)
            clean_acc = (clean_preds == labels).float().mean()
            
            for band_idx in range(self.num_bands):
                low_freq = band_idx / self.num_bands
                high_freq = (band_idx + 1) / self.num_bands
                bandwidth = high_freq - low_freq
                center = (low_freq + high_freq) / 2
                
                noise = torch.randn_like(images) * 0.05
                
                mask = self.freq_processor.get_frequency_mask(
                    mask_type='band_pass',
                    cutoff=center,
                    bandwidth=bandwidth
                )
                
                noise_filtered = self.freq_processor.apply_frequency_filter(noise, mask)
                noisy_images = images + noise_filtered
                noisy_images = torch.clamp(noisy_images, 0, 1)
                
                noisy_outputs = self.model(noisy_images)
                noisy_preds = torch.argmax(noisy_outputs, dim=1)
                noisy_acc = (noisy_preds == labels).float().mean()
                
                sensitivity = clean_acc.item() - noisy_acc.item()
                sensitivities.append(sensitivity)
        
        return sensitivities
    
    def _select_vulnerable_bands(self, sensitivities):
        """Select most vulnerable frequency bands"""
        sorted_indices = np.argsort(sensitivities)[::-1]
        return sorted_indices[:self.top_k].tolist()
    
    def generate(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Frequency sensitivity analysis
        band_sensitivities = self._analyze_frequency_sensitivity(images, labels)
        vulnerable_bands = self._select_vulnerable_bands(band_sensitivities)
        
        # Create combined mask for vulnerable bands
        combined_mask = torch.zeros((1, 1, images.shape[-2], images.shape[-1]), device=self.device)
        
        for band_idx in vulnerable_bands:
            low_freq = band_idx / self.num_bands
            high_freq = (band_idx + 1) / self.num_bands
            bandwidth = high_freq - low_freq
            center = (low_freq + high_freq) / 2
            
            mask = self.freq_processor.get_frequency_mask(
                mask_type='band_pass',
                cutoff=center,
                bandwidth=bandwidth
            )
            combined_mask += mask
        
        combined_mask = torch.clamp(combined_mask, 0, 1)
        
        # Perform attack
        delta = torch.zeros_like(images, requires_grad=True, device=self.device)
        self._init_velocity(delta.shape)
        
        stats = {
            'success_rate': [],
            'loss_history': [],
            'perturbation_norms': [],
            'band_sensitivities': band_sensitivities,
            'vulnerable_bands': vulnerable_bands,
            'iteration_times': []
        }
        
        for i in range(self.iterations):
            iter_start = time.time()
            
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
                # Apply band-limiting to gradient
                grad_fft = self.freq_processor.spatial_to_fft(delta.grad.data)
                grad_filtered = grad_fft * combined_mask
                grad_band = self.freq_processor.fft_to_spatial(grad_filtered)
                
                # Update with momentum
                self.velocity = self.momentum * self.velocity + grad_band
                delta.data = delta.data + self.alpha * torch.sign(self.velocity)
                
                # Apply spatial constraint
                delta.data = self._spatial_constraint(delta.data)
                
                # Project
                adv_images = images + delta.data
                adv_images = torch.clamp(adv_images, 0, 1)
                delta.data = adv_images - images
            
            # Record stats
            with torch.no_grad():
                pred = torch.argmax(self.model(adv_images), dim=1)
                if self.targeted:
                    success = (pred == labels).float().mean()
                else:
                    success = (pred != labels).float().mean()
                
                stats['success_rate'].append(success.item())
                stats['loss_history'].append(loss.item())
                
                pert_norm = torch.norm(delta.data.view(delta.data.shape[0], -1), p=2, dim=1).mean()
                stats['perturbation_norms'].append(pert_norm.item())
                stats['iteration_times'].append(time.time() - iter_start)
        
        adv_images = images + delta.data
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images, stats
