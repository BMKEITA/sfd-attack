"""
HVS-aware frequency processor with Contrast Sensitivity Function.
Author: Your Name
"""

import torch
import torch.nn as nn
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from config import Config

class HVSFrequencyProcessor(nn.Module):
    """
    Frequency domain processor with Human Visual System (HVS) modeling.
    Uses Contrast Sensitivity Function (CSF) to identify frequency bands
    where human vision is less sensitive.
    """
    
    def __init__(self, image_size=(32, 32), device=None):
        super().__init__()
        self.image_size = image_size
        self.device = device or Config.DEVICE
        self.peak_freq = Config.HVS_PEAK_FREQ
        self.threshold = Config.HVS_THRESHOLD
        
        self._init_hvs_sensitivity()
    
    def _init_hvs_sensitivity(self):
        """Initialize Human Visual System sensitivity model using CSF"""
        h, w = self.image_size
        u = torch.fft.fftfreq(h, d=1/h).abs().reshape(-1, 1)
        v = torch.fft.fftfreq(w, d=1/w).abs().reshape(1, -1)
        frequency = torch.sqrt(u**2 + v**2)
        
        # Contrast Sensitivity Function (CSF) - peaks at peak_freq cycles/degree
        self.csf = frequency * torch.exp(-frequency / self.peak_freq)
        self.csf = self.csf / self.csf.max()
        self.csf = self.csf.to(self.device)
        
        # HVS sensitivity map (higher = more visible to humans)
        self.hvs_sensitivity = self.csf.clone()
        
        # HVS insensitivity map (higher = less visible to humans)
        self.hvs_insensitivity = 1.0 - self.csf
        self.hvs_insensitivity = self.hvs_insensitivity / self.hvs_insensitivity.max()
        
        # Precompute frequency masks
        self._precompute_masks()
    
    def _precompute_masks(self):
        """Precompute frequency masks for efficiency"""
        h, w = self.image_size
        center_h, center_w = h // 2, w // 2
        
        y = torch.arange(h, device=self.device).float() - center_h
        x = torch.arange(w, device=self.device).float() - center_w
        Y, X = torch.meshgrid(y, x, indexing='ij')
        self.R = torch.sqrt(X**2 + Y**2) / torch.sqrt(torch.tensor(center_h**2 + center_w**2))
        
        # HVS-sensitive mask
        self.hvs_mask = (self.hvs_insensitivity > self.threshold).float()
        
        # Standard masks
        self.low_mask = (self.R <= 0.25).float()
        self.mid_mask = ((self.R > 0.25) & (self.R <= 0.5)).float()
        self.high_mask = (self.R > 0.5).float()
    
    def spatial_to_fft(self, x, shift=True):
        """Convert spatial image to FFT domain"""
        if x.requires_grad:
            x = x.detach()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x_fft = torch.fft.fft2(x, norm='ortho')
        if shift:
            x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))
        return x_fft

    def fft_to_spatial(self, x_fft, shift=True):
        """Convert FFT coefficients back to spatial domain"""
        if x_fft.requires_grad:
            x_fft = x_fft.detach()
        if shift:
            x_fft = torch.fft.ifftshift(x_fft, dim=(-2, -1))
        x = torch.fft.ifft2(x_fft, norm='ortho').real
        return x

    def get_frequency_mask(self, mask_type='hvs_sensitive', cutoff=0.2, bandwidth=0.1):
        """
        Create frequency domain mask based on HVS sensitivity.
        
        Args:
            mask_type: 'high_pass', 'low_pass', 'band_pass', 'hvs_sensitive', 'hvs_protected'
            cutoff: Cutoff frequency (normalized)
            bandwidth: Bandwidth for band-pass filters
            
        Returns:
            Frequency mask tensor [1, 1, H, W]
        """
        if mask_type == 'high_pass':
            mask = (self.R > cutoff).float()
        elif mask_type == 'low_pass':
            mask = (self.R <= cutoff).float()
        elif mask_type == 'band_pass':
            mask = ((self.R > cutoff - bandwidth/2) & 
                   (self.R < cutoff + bandwidth/2)).float()
        elif mask_type == 'band_stop':
            mask = 1.0 - ((self.R > cutoff - bandwidth/2) & 
                         (self.R < cutoff + bandwidth/2)).float()
        elif mask_type == 'hvs_sensitive':
            mask = self.hvs_mask
        elif mask_type == 'hvs_protected':
            mask = 1.0 - self.hvs_mask
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        return mask.unsqueeze(0).unsqueeze(0)
    
    def apply_frequency_filter(self, x, mask):
        """Apply frequency domain filter to image"""
        x_fft = self.spatial_to_fft(x)
        x_fft_filtered = x_fft * mask
        x_filtered = self.fft_to_spatial(x_fft_filtered)
        return x_filtered
    
    def get_frequency_bands(self, x):
        """Split image into low, mid, and high frequency bands"""
        x_fft = self.spatial_to_fft(x)
        
        x_low = self.fft_to_spatial(x_fft * self.low_mask)
        x_mid = self.fft_to_spatial(x_fft * self.mid_mask)
        x_high = self.fft_to_spatial(x_fft * self.high_mask)
        
        return x_low, x_mid, x_high
    
    def compute_frequency_energy(self, x):
        """Compute energy in different frequency bands"""
        x_fft = self.spatial_to_fft(x)
        magnitude = torch.abs(x_fft)
        
        low_energy = (magnitude * self.low_mask).sum().item()
        mid_energy = (magnitude * self.mid_mask).sum().item()
        high_energy = (magnitude * self.high_mask).sum().item()
        total_energy = low_energy + mid_energy + high_energy
        
        return {
            'low': low_energy / total_energy if total_energy > 0 else 0,
            'mid': mid_energy / total_energy if total_energy > 0 else 0,
            'high': high_energy / total_energy if total_energy > 0 else 0
        }
    
    def visualize_hvs_sensitivity(self, save_path=None):
        """Visualize HVS sensitivity map"""
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        
        # HVS sensitivity
        im1 = axes[0].imshow(self.hvs_sensitivity.cpu().numpy(), cmap='hot')
        axes[0].set_title('HVS Sensitivity\n(High = Visible)', fontsize=12)
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # HVS insensitivity (attack focus)
        im2 = axes[1].imshow(self.hvs_insensitivity.cpu().numpy(), cmap='viridis')
        axes[1].set_title('HVS Insensitivity\n(Attack Focus)', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # HVS-sensitive mask
        im3 = axes[2].imshow(self.hvs_mask.cpu().numpy(), cmap='binary')
        axes[2].set_title('HVS Attack Mask\n(Stealthy Region)', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        # Frequency bands
        bands = np.zeros_like(self.R.cpu().numpy())
        bands[self.low_mask.cpu().numpy() > 0] = 1
        bands[self.mid_mask.cpu().numpy() > 0] = 2
        bands[self.high_mask.cpu().numpy() > 0] = 3
        
        im4 = axes[3].imshow(bands, cmap='Set1')
        axes[3].set_title('Frequency Bands\n(1=Low, 2=Mid, 3=High)', fontsize=12)
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3])
        
        plt.suptitle('Human Visual System Frequency Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()