"""
Publication-quality visualization utilities.
Author: Your Name
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import torch

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class PublicationVisualizer:
    """Create publication-quality plots"""
    
    def __init__(self, save_dir='./figures'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality defaults
        #plt.rcParams['font.family'] = 'serif'
        #plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
    
    def plot_attack_progress(self, stats_dict, title="Attack Progress", filename=None):
        """Plot attack progression for multiple attacks"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(stats_dict)))
        
        for (name, stats), color in zip(stats_dict.items(), colors):
            # Success rate
            axes[0, 0].plot(stats['success_rate'], linewidth=2, color=color, label=name)
            
            # Loss
            axes[0, 1].plot(stats['loss_history'], linewidth=2, color=color, label=name)
            
            # Perturbation norm
            axes[1, 0].plot(stats['perturbation_norms'], linewidth=2, color=color, label=name)
            
            # Component losses if available
            if 'cls_loss' in stats and 'freq_loss' in stats:
                axes[1, 1].plot(stats['cls_loss'], '--', linewidth=1.5, color=color, alpha=0.7)
                axes[1, 1].plot(stats['freq_loss'], ':', linewidth=1.5, color=color, alpha=0.7)
        
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_title('Attack Success Rate')
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Progression')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('L2 Norm')
        axes[1, 0].set_title('Perturbation Norm')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Component Losses (dashed=cls, dotted=freq)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_variant_comparison(self, results, filename=None):
        """Compare attack variants with bar charts"""
        variants = list(results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics = [
            ('success_rate', 'Success Rate', 0, 0, [0, 1.2]),
            ('l2_norm', 'Perturbation Norm (L2)', 0, 1, None),  # Changed from 'perturbation_norm'
            ('time', 'Execution Time (s)', 0, 2, None),
            ('ssim', 'SSIM', 1, 0, [0, 1]),
            ('psnr', 'PSNR (dB)', 1, 1, None),
            ('lpips', 'LPIPS', 1, 2, [0, 1])
        ]
            
        for metric, label, i, j, ylim in metrics:
            values = [results[v].get(metric, 0) for v in variants]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(variants)))
            
            bars = axes[i, j].bar(variants, values, color=colors, alpha=0.8)
            axes[i, j].set_xlabel('Attack Variant')
            axes[i, j].set_ylabel(label)
            axes[i, j].set_title(f'{label} by Variant')
            axes[i, j].tick_params(axis='x', rotation=45)
            axes[i, j].grid(True, alpha=0.3, axis='y')
            
            if ylim:
                axes[i, j].set_ylim(ylim)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[i, j].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Attack Variant Comparison', fontsize=16)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_defense_comparison(self, defense_results, filename=None):
        """Compare defense effectiveness"""
        defenses = list(defense_results.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Extract metrics
        attack_success = [defense_results[d]['attack_success'] for d in defenses]
        defended_success = [defense_results[d]['defended_attack_success'] for d in defenses]
        efficacy = [defense_results[d]['defense_efficacy'] for d in defenses]
        
        # Bar positions
        x = np.arange(len(defenses))
        width = 0.35
        
        # Attack success before/after defense
        axes[0].bar(x - width/2, attack_success, width, label='Before Defense', 
                   color='red', alpha=0.7)
        axes[0].bar(x + width/2, defended_success, width, label='After Defense', 
                   color='green', alpha=0.7)
        axes[0].set_xlabel('Defense')
        axes[0].set_ylabel('Attack Success Rate')
        axes[0].set_title('Defense Effectiveness')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(defenses, rotation=45, ha='right')
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Defense efficacy
        colors = ['green' if e > 0 else 'gray' for e in efficacy]
        bars = axes[1].bar(defenses, efficacy, color=colors)
        axes[1].set_xlabel('Defense')
        axes[1].set_ylabel('Defense Efficacy')
        axes[1].set_title('Defense Efficacy Score')
        axes[1].set_xticklabels(defenses, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        for bar, eff in zip(bars, efficacy):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{eff:.3f}', ha='center', va='bottom' if eff > 0 else 'top')
        
        # Defense ranking
        sorted_idx = np.argsort(efficacy)[::-1]
        sorted_defenses = [defenses[i] for i in sorted_idx]
        sorted_efficacy = [efficacy[i] for i in sorted_idx]
        
        colors2 = plt.cm.RdYlGn(np.linspace(0, 1, len(sorted_defenses)))
        bars = axes[2].barh(sorted_defenses, sorted_efficacy, color=colors2)
        axes[2].set_xlabel('Defense Efficacy')
        axes[2].set_title('Defense Ranking')
        axes[2].grid(True, alpha=0.3, axis='x')
        
        for bar, eff in zip(bars, sorted_efficacy):
            width = bar.get_width()
            axes[2].text(width + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{eff:.3f}', ha='left', va='center')
        
        plt.suptitle('Defense Evaluation Results', fontsize=16)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
    

    def plot_epsilon_scaling(self, results_by_eps, filename=None):
        """Plot how metrics scale with epsilon"""
        epsilons = sorted(results_by_eps.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        variants = list(results_by_eps[epsilons[0]].keys())
        
        for variant in variants:
            success = [results_by_eps[eps][variant]['success_rate'] for eps in epsilons]
            # Change 'perturbation_norm' to 'l2_norm'
            pert = [results_by_eps[eps][variant]['l2_norm'] for eps in epsilons]
            
            axes[0].plot(epsilons, success, 'o-', linewidth=2, markersize=8, label=variant)
            axes[1].plot(epsilons, pert, 'o-', linewidth=2, markersize=8, label=variant)
        
        # Add ideal scaling line for perturbation
        if epsilons:
            x = np.array([0, max(epsilons) * 1.2])
            y = x * (pert[0] / epsilons[0])
            axes[1].plot(x, y, 'k--', alpha=0.5, label='Ideal Linear')
        
        axes[0].set_xlabel('Epsilon (e)')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_title('Success Rate vs Epsilon')
        axes[0].set_ylim(0, 1.1)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epsilon (e)')
        axes[1].set_ylabel('Perturbation Norm')
        axes[1].set_title('Perturbation Magnitude vs Epsilon')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Effect of Epsilon on Attack Performance', fontsize=16)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
    

    def plot_transferability_heatmap(self, transfer_matrix, source_models, target_models, filename=None):
        """Plot transferability as heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(transfer_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        ax.set_xticks(np.arange(len(target_models)))
        ax.set_yticks(np.arange(len(source_models)))
        ax.set_xticklabels(target_models, rotation=45, ha='right')
        ax.set_yticklabels(source_models)
        
        # Add text annotations
        for i in range(len(source_models)):
            for j in range(len(target_models)):
                text = ax.text(j, i, f'{transfer_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        ax.set_xlabel('Target Model')
        ax.set_ylabel('Source Model')
        ax.set_title('Attack Transferability Heatmap\n(1 = Perfect Transfer)')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_frequency_analysis(self, clean_img, adv_img, freq_processor, filename=None):
        """Plot frequency domain analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Convert to grayscale for frequency analysis
        clean_gray = 0.2989 * clean_img[0] + 0.5870 * clean_img[1] + 0.1140 * clean_img[2]
        adv_gray = 0.2989 * adv_img[0] + 0.5870 * adv_img[1] + 0.1140 * adv_img[2]
        pert_gray = adv_gray - clean_gray
        
        # FFT
        clean_fft = freq_processor.spatial_to_fft(clean_gray.unsqueeze(0).unsqueeze(0))
        adv_fft = freq_processor.spatial_to_fft(adv_gray.unsqueeze(0).unsqueeze(0))
        pert_fft = freq_processor.spatial_to_fft(pert_gray.unsqueeze(0).unsqueeze(0))
        
        clean_mag = torch.log(1 + torch.abs(clean_fft))[0, 0].cpu().numpy()
        adv_mag = torch.log(1 + torch.abs(adv_fft))[0, 0].cpu().numpy()
        pert_mag = torch.log(1 + torch.abs(pert_fft))[0, 0].cpu().numpy()
        
        # Spatial domain
        axes[0, 0].imshow(clean_gray.cpu().numpy(), cmap='gray')
        axes[0, 0].set_title('Clean Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(adv_gray.cpu().numpy(), cmap='gray')
        axes[0, 1].set_title('Adversarial Image')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pert_gray.cpu().numpy(), cmap='RdBu', vmin=-0.03, vmax=0.03)
        axes[0, 2].set_title('Perturbation')
        axes[0, 2].axis('off')
        plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2])
        
        # Frequency domain
        im1 = axes[1, 0].imshow(clean_mag, cmap='viridis')
        axes[1, 0].set_title('Clean - Frequency Magnitude')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0])
        
        im2 = axes[1, 1].imshow(adv_mag, cmap='viridis')
        axes[1, 1].set_title('Adversarial - Frequency Magnitude')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        
        im3 = axes[1, 2].imshow(pert_mag, cmap='hot')
        axes[1, 2].set_title('Perturbation - Frequency Magnitude')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2])
        
        plt.suptitle('Spatial and Frequency Domain Analysis', fontsize=16)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()