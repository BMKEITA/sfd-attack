#!/usr/bin/env python3
"""
Generate publication-quality figures for paper.
Run: python generate_paper_figs.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import json

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set font parameters for paper
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def load_results():
    """Load all result files"""
    results = {}
    
    # Load main results from your JSON
    try:
        with open('results.json', 'r') as f:
            results['main'] = json.load(f)
    except:
        print("Warning: results.json not found")
    
    # Load comparison results
    try:
        with open('comparison_results.json', 'r') as f:
            results['comparison'] = json.load(f)
    except:
        print("Warning: comparison_results.json not found")
    
    # Load CSV if available
    try:
        results['csv'] = pd.read_csv('comparison_results.csv')
    except:
        print("Warning: comparison_results.csv not found")
    
    return results

def fig1_attack_comparison(results, save_dir):
    """Figure 1: Bar chart comparing all attacks"""
    
    if 'csv' not in results:
        print("Skipping fig1: no CSV data")
        return
    
    df = results['csv']
    
    # Sort by perturbation norm for better visualization
    df_sorted = df.sort_values('l2_norm')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Perturbation norms
    x = np.arange(len(df_sorted))
    bars1 = axes[0].bar(x, df_sorted['l2_norm'], color='steelblue')
    axes[0].set_xlabel('Attack Method')
    axes[0].set_ylabel('L2 Perturbation Norm')
    axes[0].set_title('(a) Perturbation Magnitude')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_sorted['attack'], rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, df_sorted['l2_norm']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: SSIM scores
    bars2 = axes[1].bar(x, df_sorted['ssim'], color='forestgreen')
    axes[1].set_xlabel('Attack Method')
    axes[1].set_ylabel('SSIM (higher is better)')
    axes[1].set_title('(b) Perceptual Quality')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_sorted['attack'], rotation=45, ha='right')
    axes[1].set_ylim(0.95, 0.99)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, df_sorted['ssim']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: PSNR
    bars3 = axes[2].bar(x, df_sorted['psnr'], color='coral')
    axes[2].set_xlabel('Attack Method')
    axes[2].set_ylabel('PSNR (dB)')
    axes[2].set_title('(c) Peak Signal-to-Noise Ratio')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df_sorted['attack'], rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, df_sorted['psnr']):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Attack Performance Comparison on CIFAR-10', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig1_attack_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig1_attack_comparison.pdf', bbox_inches='tight')
    plt.show()

def fig2_loss_convergence(results, save_dir):
    """Figure 2: Loss convergence plots for joint hybrid attack"""
    
    if 'main' not in results:
        print("Skipping fig2: no main results")
        return
    
    try:
        joint_stats = results['main']['cifar10']['resnet18']['joint']['stats']
    except:
        print("Could not find joint attack stats")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    iterations = range(1, len(joint_stats['loss_history']) + 1)
    
    # Total loss
    axes[0, 0].plot(iterations, joint_stats['loss_history'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('(a) Total Loss Convergence')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Classification loss
    axes[0, 1].plot(iterations, joint_stats['cls_loss'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Classification Loss')
    axes[0, 1].set_title('(b) Classification Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Frequency loss
    axes[1, 0].plot(iterations, joint_stats['freq_loss'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Frequency Loss')
    axes[1, 0].set_title('(c) HVS-Weighted Frequency Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Perturbation norm
    axes[1, 1].plot(iterations, joint_stats['perturbation_norms'], 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].set_title('(d) Perturbation Norm')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Joint Hybrid Attack Convergence', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig2_loss_convergence.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig2_loss_convergence.pdf', bbox_inches='tight')
    plt.show()

def fig3_ablation_freq_weight(save_dir):
    """Figure 3: Ablation study on frequency weight"""
    
    # Data from your ablation results
    weights = [0.0, 0.3, 0.5, 0.7, 1.0]
    pert_norms = [1.561, 1.456, 1.411, 1.381, 1.346]
    ssim_scores = [0.971, 0.974, 0.975, 0.976, 0.978]
    psnr_values = [33.04, 33.51, 33.75, 34.04, 34.44]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Perturbation norm
    axes[0].plot(weights, pert_norms, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Frequency Weight ?')
    axes[0].set_ylabel('L2 Perturbation Norm')
    axes[0].set_title('(a) Effect on Perturbation Magnitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='?=0.5')
    axes[0].legend()
    
    # SSIM
    axes[1].plot(weights, ssim_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Frequency Weight ?')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('(b) Effect on Perceptual Quality')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    
    # PSNR
    axes[2].plot(weights, psnr_values, 'mo-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Frequency Weight ?')
    axes[2].set_ylabel('PSNR (dB)')
    axes[2].set_title('(c) Effect on Signal-to-Noise Ratio')
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    
    plt.suptitle('Ablation Study: Frequency Weight', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig3_ablation_freq_weight.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig3_ablation_freq_weight.pdf', bbox_inches='tight')
    plt.show()

def fig4_epsilon_analysis(save_dir):
    """Figure 4: Effect of epsilon on attack performance"""
    
    epsilons = [0.01, 0.03, 0.05, 0.10]
    pert_norms = [0.480, 1.411, 2.306, 4.406]
    ssim_scores = [0.997, 0.975, 0.943, 0.821]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Perturbation norm scaling
    axes[0].plot(epsilons, pert_norms, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Perturbation Budget e')
    axes[0].set_ylabel('L2 Perturbation Norm')
    axes[0].set_title('(a) Perturbation Scaling')
    axes[0].grid(True, alpha=0.3)
    
    # Add ideal linear scaling line
    x = np.array([0, 0.12])
    y = x * (pert_norms[1] / epsilons[1])
    axes[0].plot(x, y, 'r--', alpha=0.5, label='Ideal Linear')
    axes[0].legend()
    
    # SSIM degradation
    axes[1].plot(epsilons, ssim_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Perturbation Budget e')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('(b) Perceptual Quality Degradation')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Perturbation Budget', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig4_epsilon_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig4_epsilon_analysis.pdf', bbox_inches='tight')
    plt.show()

def fig5_iteration_analysis(save_dir):
    """Figure 5: Effect of iterations on attack performance"""
    
    iterations = [5, 10, 20, 50]
    pert_norms = [1.270, 1.359, 1.411, 1.428]
    times = [97.7, 193.6, 390.6, 928.4]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Perturbation norm convergence
    axes[0].plot(iterations, pert_norms, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Iterations')
    axes[0].set_ylabel('L2 Perturbation Norm')
    axes[0].set_title('(a) Perturbation Convergence')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=20, color='r', linestyle='--', alpha=0.5, label='20 iterations')
    axes[0].legend()
    
    # Time cost
    axes[1].plot(iterations, times, 'mo-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Iterations')
    axes[1].set_ylabel('Execution Time (seconds)')
    axes[1].set_title('(b) Computational Cost')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=20, color='r', linestyle='--', alpha=0.5)
    
    plt.suptitle('Effect of Iterations on Attack Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig5_iteration_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig5_iteration_analysis.pdf', bbox_inches='tight')
    plt.show()

def fig6_visual_comparison(save_dir):
    """Figure 6: Visual comparison of adversarial examples"""
    
    # This would require actual images - placeholder for now
    # You'll need to load your saved adversarial images
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    methods = ['Clean', 'FGSM', 'PGD', 'Ours']
    
    for i, method in enumerate(methods):
        # Placeholder - replace with actual images
        for j in range(3):
            axes[j, i].text(0.5, 0.5, f'{method}\nSample {j+1}', 
                           ha='center', va='center', fontsize=12)
            axes[j, i].axis('off')
            if j == 0:
                axes[j, i].set_title(method, fontsize=14)
    
    plt.suptitle('Visual Comparison of Adversarial Examples', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig6_visual_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig6_visual_comparison.pdf', bbox_inches='tight')
    plt.show()
    print("\nNote: fig6 requires actual image files. Please update with your saved adversarial images.")

def generate_all_figures():
    """Generate all publication figures"""
    
    print("="*80)
    print("GENERATING PUBLICATION FIGURES")
    print("="*80)
    
    # Create output directory
    save_dir = Path('./paper_figures')
    save_dir.mkdir(exist_ok=True)
    
    # Load results
    results = load_results()
    
    # Generate each figure
    print("\nGenerating Figure 1: Attack Comparison...")
    fig1_attack_comparison(results, save_dir)
    
    print("\nGenerating Figure 2: Loss Convergence...")
    fig2_loss_convergence(results, save_dir)
    
    print("\nGenerating Figure 3: Frequency Weight Ablation...")
    fig3_ablation_freq_weight(save_dir)
    
    print("\nGenerating Figure 4: Epsilon Analysis...")
    fig4_epsilon_analysis(save_dir)
    
    print("\nGenerating Figure 5: Iteration Analysis...")
    fig5_iteration_analysis(save_dir)
    
    print("\nGenerating Figure 6: Visual Comparison...")
    fig6_visual_comparison(save_dir)
    
    print(f"\n All figures saved to {save_dir}")

if __name__ == "__main__":
    generate_all_figures()