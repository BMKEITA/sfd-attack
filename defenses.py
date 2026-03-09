"""
Comprehensive defense mechanisms against adversarial attacks.
Author: Your Name
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import cv2
from scipy.ndimage import median_filter
import kornia

class DefenseEvaluator:
    """Evaluates attacks against various defenses"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def gaussian_blur(self, images, kernel_size=5, sigma=1.0):
        """Gaussian blur defense"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        images_contiguous = images.contiguous()
        blurred = kornia.filters.gaussian_blur2d(
            images_contiguous, (kernel_size, kernel_size), (sigma, sigma)
        )
        return blurred
    
    def median_filter(self, images, kernel_size=3):
        """Median filter defense"""
        filtered_images = []
        for img in images:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            filtered_np = np.zeros_like(img_np)
            for c in range(3):
                filtered_np[:, :, c] = median_filter(img_np[:, :, c], size=kernel_size)
            filtered_tensor = torch.tensor(filtered_np, device=self.device).permute(2, 0, 1)
            filtered_images.append(filtered_tensor)
        return torch.stack(filtered_images)
    
    def jpeg_compression(self, images, quality=75):
        """JPEG compression defense"""
        compressed_images = []
        for img in images:
            img_np = img.permute(1, 2, 0).cpu().numpy() * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            pil_img = Image.fromarray(img_np)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            compressed_img = Image.open(buffer)
            compressed_img = np.array(compressed_img).astype(np.float32) / 255.0
            compressed_img = compressed_img[:img.shape[1], :img.shape[2], :]
            
            compressed_tensor = torch.tensor(compressed_img, device=self.device).permute(2, 0, 1)
            compressed_images.append(compressed_tensor)
        
        return torch.stack(compressed_images)
    
    def bit_depth_reduction(self, images, bits=4):
        """Bit depth reduction defense"""
        levels = 2 ** bits
        images_quantized = torch.round(images * (levels - 1)) / (levels - 1)
        return images_quantized
    
    def random_noise(self, images, noise_level=0.02):
        """Random noise injection defense"""
        noise = torch.randn_like(images) * noise_level
        noisy_images = images + noise
        noisy_images = torch.clamp(noisy_images, 0, 1)
        return noisy_images
    
    def frequency_filter(self, images, filter_type='low_pass', cutoff=0.3, bandwidth=0.1):
        """Frequency domain filtering defense"""
        from frequency_processor import HVSFrequencyProcessor
        
        h, w = images.shape[-2:]
        freq_processor = HVSFrequencyProcessor(image_size=(h, w), device=self.device)
        
        if filter_type == 'low_pass':
            mask = freq_processor.get_frequency_mask('low_pass', cutoff=cutoff)
        elif filter_type == 'high_pass':
            mask = freq_processor.get_frequency_mask('high_pass', cutoff=cutoff)
        elif filter_type == 'band_stop':
            mask = freq_processor.get_frequency_mask('band_stop', cutoff=cutoff, bandwidth=bandwidth)
        else:
            mask = torch.ones((1, 1, h, w), device=self.device)
        
        filtered_images = []
        for img in images:
            filtered = freq_processor.apply_frequency_filter(img.unsqueeze(0), mask)
            filtered_images.append(filtered.squeeze(0))
        
        return torch.stack(filtered_images)
    
    def feature_squeezing(self, images, bit_depth=5):
        """Feature squeezing defense"""
        return self.bit_depth_reduction(images, bits=bit_depth)
    
    def total_variance_minimization(self, images, lambda_tv=0.1, iterations=10):
        """Total variance minimization defense"""
        def tv_loss(x):
            diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
            diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            return diff_h.mean() + diff_w.mean()
        
        defended = images.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([defended], lr=0.01)
        
        for _ in range(iterations):
            loss = F.mse_loss(defended, images) + lambda_tv * tv_loss(defended)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            defended.data = torch.clamp(defended.data, 0, 1)
        
        return defended.detach()
    
    def test_defense(self, model, clean_images, adv_images, labels, 
                    defense_name, defense_func, **kwargs):
        """Test a specific defense"""
        with torch.no_grad():
            # Apply defense
            if defense_name == 'No Defense':
                defended_adv = adv_images
            else:
                defended_adv = defense_func(adv_images, **kwargs)
            
            # Get predictions
            clean_preds = torch.argmax(model(clean_images), dim=1)
            adv_preds = torch.argmax(model(adv_images), dim=1)
            defended_preds = torch.argmax(model(defended_adv), dim=1)
            
            # Calculate metrics
            clean_acc = (clean_preds == labels).float().mean()
            adv_acc = (adv_preds == labels).float().mean()
            defended_acc = (defended_preds == labels).float().mean()
            
            attack_success = 1.0 - adv_acc
            defended_success = 1.0 - defended_acc
            defense_efficacy = defended_acc - adv_acc
            
            results = {
                'defense': defense_name,
                'clean_accuracy': clean_acc.item(),
                'adv_accuracy': adv_acc.item(),
                'defended_accuracy': defended_acc.item(),
                'attack_success': attack_success.item(),
                'defended_attack_success': defended_success.item(),
                'defense_efficacy': defense_efficacy.item()
            }
            
            return results


class BPDAWrapper(nn.Module):
    """
    Backward Pass Differentiable Approximation wrapper for non-differentiable defenses.
    Allows gradient-based attacks to work with non-differentiable defenses.
    """
    
    def __init__(self, defense_func):
        super().__init__()
        self.defense_func = defense_func
    
    def forward(self, x):
        with torch.no_grad():
            y = self.defense_func(x)
        return x + (y - x).detach()