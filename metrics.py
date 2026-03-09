"""
Perceptual and attack metrics for evaluation.
Author: Your Name
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

class PerceptualMetrics:
    """Compute perceptual quality metrics"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
    
    def ssim(self, img1, img2):
        """Structural Similarity Index"""
        # Detach and ensure no gradients
        if img1.requires_grad:
            img1 = img1.detach()
        if img2.requires_grad:
            img2 = img2.detach()
        
        # Ensure images are on CPU and properly shaped
        if img1.dim() == 4:
            img1 = img1[0]
        if img2.dim() == 4:
            img2 = img2[0]
        
        img1_np = img1.permute(1, 2, 0).cpu().numpy()
        img2_np = img2.permute(1, 2, 0).cpu().numpy()
        
        # Clip to valid range
        img1_np = np.clip(img1_np, 0, 1)
        img2_np = np.clip(img2_np, 0, 1)
        
        # Convert to grayscale for SSIM
        img1_gray = 0.2989 * img1_np[:, :, 0] + 0.5870 * img1_np[:, :, 1] + 0.1140 * img1_np[:, :, 2]
        img2_gray = 0.2989 * img2_np[:, :, 0] + 0.5870 * img2_np[:, :, 1] + 0.1140 * img2_np[:, :, 2]
        
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1_gray, img2_gray, data_range=1.0)
    
    def psnr(self, img1, img2):
        """Peak Signal-to-Noise Ratio"""
        # Detach and ensure no gradients
        if img1.requires_grad:
            img1 = img1.detach()
        if img2.requires_grad:
            img2 = img2.detach()
        
        if img1.dim() == 4:
            img1 = img1[0]
        if img2.dim() == 4:
            img2 = img2[0]
        
        img1_np = img1.permute(1, 2, 0).cpu().numpy()
        img2_np = img2.permute(1, 2, 0).cpu().numpy()
        
        img1_np = np.clip(img1_np, 0, 1)
        img2_np = np.clip(img2_np, 0, 1)
        
        # Convert to grayscale
        img1_gray = 0.2989 * img1_np[:, :, 0] + 0.5870 * img1_np[:, :, 1] + 0.1140 * img1_np[:, :, 2]
        img2_gray = 0.2989 * img2_np[:, :, 0] + 0.5870 * img2_np[:, :, 1] + 0.1140 * img2_np[:, :, 2]
        
        from skimage.metrics import peak_signal_noise_ratio as psnr
        return psnr(img1_gray, img2_gray, data_range=1.0)
    
    def lpips(self, img1, img2):
        """Learned Perceptual Image Patch Similarity"""
        # Detach and ensure no gradients
        if img1.requires_grad:
            img1 = img1.detach()
        if img2.requires_grad:
            img2 = img2.detach()
        
        if img1.dim() == 4:
            img1 = img1[0]
        if img2.dim() == 4:
            img2 = img2[0]
        
        img1 = img1.unsqueeze(0).to(self.device)
        img2 = img2.unsqueeze(0).to(self.device)
        
        # Normalize to [-1, 1] for LPIPS
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1
        
        with torch.no_grad():
            return self.lpips_fn(img1, img2).item()
    
    def mse(self, img1, img2):
        """Mean Squared Error"""
        if img1.requires_grad:
            img1 = img1.detach()
        if img2.requires_grad:
            img2 = img2.detach()
        return F.mse_loss(img1, img2).item()
    
    def l2_norm(self, perturbation):
        """L2 norm of perturbation"""
        if perturbation.requires_grad:
            perturbation = perturbation.detach()
        return torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1).mean().item()
    
    def compute_all(self, clean, adv, perturbation):
        """Compute all perceptual metrics"""
        # Detach all tensors first
        if clean.requires_grad:
            clean = clean.detach()
        if adv.requires_grad:
            adv = adv.detach()
        if perturbation.requires_grad:
            perturbation = perturbation.detach()
        
        ssim_val = self.ssim(clean, adv)
        psnr_val = self.psnr(clean, adv)
        lpips_val = self.lpips(clean, adv)
        mse_val = self.mse(clean, adv)
        l2_val = self.l2_norm(perturbation)
        
        return {
            'ssim': ssim_val,
            'psnr': psnr_val,
            'lpips': lpips_val,
            'mse': mse_val,
            'l2_norm': l2_val
        }


class AttackMetrics:
    """Compute attack effectiveness metrics"""
    
    @staticmethod
    def success_rate(predictions, labels):
        """Attack success rate"""
        return (predictions != labels).float().mean().item()
    
    @staticmethod
    def confidence_drop(clean_probs, adv_probs, labels):
        """Drop in confidence for true class"""
        clean_conf = torch.gather(clean_probs, 1, labels.unsqueeze(1)).squeeze()
        adv_conf = torch.gather(adv_probs, 1, labels.unsqueeze(1)).squeeze()
        return (clean_conf - adv_conf).mean().item()
    
    @staticmethod
    def transferability(source_preds, target_preds, labels):
        """Attack transferability rate"""
        return (target_preds != labels).float().mean().item()
    
    @staticmethod
    def convergence_speed(success_history, threshold=0.95):
        """Iteration at which attack reaches threshold"""
        for i, success in enumerate(success_history):
            if success >= threshold:
                return i
        return len(success_history)