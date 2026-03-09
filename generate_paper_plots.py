#!/usr/bin/env python3
"""
Generate all publication-quality plots for the Spatial-Frequency Domain Attack paper.
Run: python generate_paper_plots.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import json
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set font parameters for paper
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

# Color palette
COLORS = {
    'fgsm': '#E24A33',      # red-orange
    'pgd': '#348ABD',        # blue
    'sfa': '#988ED5',        # purple
    'facl': '#777777',       # gray
    'sequential': '#8EBA42', # green
    'joint': '#FFB30E',      # gold
    'adaptive': '#17BECF',   # cyan
    'FGSM': '#E24A33',
    'PGD': '#348ABD',
    'SFA (2025)': '#988ED5',
    'FACL (2024)': '#777777',
    'Sequential': '#8EBA42',
    'Joint Hybrid': '#FFB30E',
    'Adaptive Band': '#17BECF',
}

def load_results():
    """Load all result files from current directory"""
    results = {}
    
    print("\nScanning for result files...")
    
    # Look for result files in current directory
    current_dir = Path('.')
    
    # Find all JSON and CSV files
    json_files = list(current_dir.glob('*.json'))
    csv_files = list(current_dir.glob('*.csv'))
    
    # Also check in subdirectories
    json_files.extend(list(current_dir.glob('**/*.json')))
    csv_files.extend(list(current_dir.glob('**/*.csv')))
    
    # Remove duplicates
    json_files = list(set(json_files))
    csv_files = list(set(csv_files))
    
    print(f"\nFound {len(json_files)} JSON files and {len(csv_files)} CSV files")
    
    # Load JSON files
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results[json_file.stem] = data
            print(f"  ? Loaded {json_file.name} ({json_file.parent})")
        except Exception as e:
            print(f"  ? Could not load {json_file.name}: {e}")
    
    # Load CSV files
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            results[csv_file.stem] = df
            print(f"  ? Loaded {csv_file.name} ({csv_file.parent})")
        except Exception as e:
            print(f"  ? Could not load {csv_file.name}: {e}")
    
    return results

def extract_comparison_data(results):
    """Extract comparison data from either JSON or CSV"""
    
    # Try to get from comparison_results.csv first
    if 'comparison_results' in results and isinstance(results['comparison_results'], pd.DataFrame):
        return results['comparison_results']
    
    # Try to get from comparison_results.json
    if 'comparison_results' in results and isinstance(results['comparison_results'], list):
        return pd.DataFrame(results['comparison_results'])
    
    # Try to get from results.json
    if 'results' in results:
        try:
            data = results['results']
            if 'cifar10' in data and 'resnet18' in data['cifar10']:
                rows = []
                for attack_name, attack_data in data['cifar10']['resnet18'].items():
                    if attack_name not in ['fgsm', 'pgd', 'sfa', 'facl', 'sequential', 'joint', 'adaptive']:
                        continue
                    rows.append({
                        'attack': attack_name,
                        'attack_name': attack_data.get('attack_name', attack_name),
                        'success_rate': attack_data['success_rate'],
                        'l2_norm': attack_data['l2_norm'],
                        'ssim': attack_data['ssim'],
                        'psnr': attack_data['psnr'],
                        'lpips': attack_data['lpips'],
                        'time': attack_data['time']
                    })
                return pd.DataFrame(rows)
        except:
            pass
    
    return None

def extract_ablation_data(results):
    """Extract ablation data from results.json"""
    
    if 'results' in results:
        try:
            data = results['results']
            if 'ablation' in data:
                return data['ablation']
        except:
            pass
    
    return None

def extract_cross_dataset_data(results):
    """Extract cross-dataset data"""
    
    if 'cross_dataset_results' in results and isinstance(results['cross_dataset_results'], pd.DataFrame):
        return results['cross_dataset_results']
    
    return None

def fig1_attack_comparison(results, save_dir):
    """Figure 1: Bar chart comparing all attacks on CIFAR-10"""
    
    df = extract_comparison_data(results)
    if df is None:
        print("  ? Could not find comparison data, skipping fig1")
        return
    
    # Sort by perturbation norm for better visualization
    df_sorted = df.sort_values('l2_norm')
    
    # Create display labels
    df_sorted['display_name'] = df_sorted['attack_name'] if 'attack_name' in df_sorted.columns else df_sorted['attack']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(df_sorted))
    width = 0.6
    
    # Plot 1: Perturbation norms
    colors1 = [COLORS.get(name, '#777777') for name in df_sorted['attack']]
    bars1 = axes[0].bar(x, df_sorted['l2_norm'], width, color=colors1, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Attack Method')
    axes[0].set_ylabel('L2 Perturbation Norm')
    axes[0].set_title('(a) Perturbation Magnitude', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_sorted['display_name'], rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 1.8)
    
    # Add value labels
    for bar, val in zip(bars1, df_sorted['l2_norm']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: SSIM scores
    bars2 = axes[1].bar(x, df_sorted['ssim'], width, color=colors1, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Attack Method')
    axes[1].set_ylabel('SSIM (higher is better)')
    axes[1].set_title('(b) Perceptual Quality', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_sorted['display_name'], rotation=45, ha='right')
    axes[1].set_ylim(0.95, 0.99)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, df_sorted['ssim']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: PSNR
    bars3 = axes[2].bar(x, df_sorted['psnr'], width, color=colors1, edgecolor='black', linewidth=0.5)
    axes[2].set_xlabel('Attack Method')
    axes[2].set_ylabel('PSNR (dB)')
    axes[2].set_title('(c) Peak Signal-to-Noise Ratio', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df_sorted['display_name'], rotation=45, ha='right')
    axes[2].set_ylim(30, 36)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, df_sorted['psnr']):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Attack Performance Comparison on CIFAR-10', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig1_attack_comparison.pdf', bbox_inches='tight')
    plt.savefig(save_dir / 'fig1_attack_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ? Figure 1: Attack comparison generated")

def fig2_loss_convergence(results, save_dir):
    """Figure 2: Loss convergence plots for joint hybrid attack"""
    
    if 'results' not in results:
        print("  ? results.json not found, skipping fig2")
        return
    
    try:
        joint_stats = results['results']['cifar10']['resnet18']['joint']['stats']
    except:
        print("  ? Could not find joint attack stats in results")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    iterations = range(1, len(joint_stats['loss_history']) + 1)
    
    # Total loss
    axes[0, 0].plot(iterations, joint_stats['loss_history'], 'b-', linewidth=2, marker='o', markersize=4, markevery=5)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('(a) Total Loss Convergence', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_locator(MultipleLocator(5))
    
    # Classification loss
    axes[0, 1].plot(iterations, joint_stats['cls_loss'], 'r-', linewidth=2, marker='s', markersize=4, markevery=5)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Classification Loss')
    axes[0, 1].set_title('(b) Classification Loss', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].xaxis.set_major_locator(MultipleLocator(5))
    
    # Frequency loss
    axes[1, 0].plot(iterations, joint_stats['freq_loss'], 'g-', linewidth=2, marker='^', markersize=4, markevery=5)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Frequency Loss')
    axes[1, 0].set_title('(c) HVS-Weighted Frequency Loss', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].xaxis.set_major_locator(MultipleLocator(5))
    
    # Perturbation norm
    axes[1, 1].plot(iterations, joint_stats['perturbation_norms'], 'purple', linewidth=2, marker='d', markersize=4, markevery=5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].set_title('(d) Perturbation Norm', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].xaxis.set_major_locator(MultipleLocator(5))
    
    plt.suptitle('Joint Hybrid Attack Convergence Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig2_loss_convergence.pdf', bbox_inches='tight')
    plt.savefig(save_dir / 'fig2_loss_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ? Figure 2: Loss convergence generated")

def fig3_cross_dataset_comparison(results, save_dir):
    """Figure 3: Cross-dataset performance comparison"""
    
    df = extract_cross_dataset_data(results)
    if df is None:
        print("  ? cross_dataset_results.csv not found, skipping fig3")
        return
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = df['dataset'].unique()
    attacks = df['attack'].unique()
    
    # Bar positions
    x = np.arange(len(datasets))
    width = 0.25
    
    # Colors for different attacks
    color_map = {'FGSM': COLORS['fgsm'], 'PGD': COLORS['pgd'], 'Ours (Joint)': COLORS['joint']}
    
    # Plot 1: Success rates
    for i, attack in enumerate(attacks):
        success_vals = []
        for dataset in datasets:
            val = df[(df['dataset'] == dataset) & (df['attack'] == attack)]['success_rate'].values
            success_vals.append(val[0] if len(val) > 0 else 0)
        bars = axes[0].bar(x + (i-1)*width, success_vals, width, 
                          label=attack, color=color_map.get(attack, '#777777'), 
                          edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, success_vals):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('(a) Attack Success Rate', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: SSIM scores
    for i, attack in enumerate(attacks):
        ssim_vals = []
        for dataset in datasets:
            val = df[(df['dataset'] == dataset) & (df['attack'] == attack)]['ssim'].values
            ssim_vals.append(val[0] if len(val) > 0 else 0)
        bars = axes[1].bar(x + (i-1)*width, ssim_vals, width,
                          label=attack, color=color_map.get(attack, '#777777'),
                          edgecolor='black', linewidth=0.5)
        
        for bar, val in zip(bars, ssim_vals):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    axes[1].set_xlabel('Dataset')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('(b) Perceptual Quality', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylim(0.98, 1.0)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: PSNR
    for i, attack in enumerate(attacks):
        psnr_vals = []
        for dataset in datasets:
            val = df[(df['dataset'] == dataset) & (df['attack'] == attack)]['psnr'].values
            psnr_vals.append(val[0] if len(val) > 0 else 0)
        bars = axes[2].bar(x + (i-1)*width, psnr_vals, width,
                          label=attack, color=color_map.get(attack, '#777777'),
                          edgecolor='black', linewidth=0.5)
        
        for bar, val in zip(bars, psnr_vals):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    axes[2].set_xlabel('Dataset')
    axes[2].set_ylabel('PSNR (dB)')
    axes[2].set_title('(c) Peak Signal-to-Noise Ratio', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(datasets)
    axes[2].set_yscale('linear')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Cross-Dataset Attack Performance', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig3_cross_dataset.pdf', bbox_inches='tight')
    plt.savefig(save_dir / 'fig3_cross_dataset.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ? Figure 3: Cross-dataset comparison generated")

def fig4_ablation_freq_weight(results, save_dir):
    """Figure 4: Ablation study on frequency weight"""
    
    ablation = extract_ablation_data(results)
    if ablation is None:
        print("  ? Could not find ablation data, skipping fig4")
        return
    
    # Extract frequency weight data
    weights = []
    pert_norms = []
    ssim_scores = []
    psnr_values = []
    
    for key, value in ablation.items():
        if key.startswith('freq_weight_'):
            w = float(key.split('_')[2])
            weights.append(w)
            pert_norms.append(value['l2_norm'])
            ssim_scores.append(value['ssim'])
            psnr_values.append(value['psnr'])
    
    if not weights:
        print("  ? No frequency weight data found")
        return
    
    # Sort by weight
    sorted_idx = np.argsort(weights)
    weights = np.array(weights)[sorted_idx]
    pert_norms = np.array(pert_norms)[sorted_idx]
    ssim_scores = np.array(ssim_scores)[sorted_idx]
    psnr_values = np.array(psnr_values)[sorted_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Perturbation norm
    axes[0].plot(weights, pert_norms, 'bo-', linewidth=2, markersize=8, markerfacecolor='white')
    axes[0].set_xlabel('Frequency Weight ?')
    axes[0].set_ylabel('L2 Perturbation Norm')
    axes[0].set_title('(a) Effect on Perturbation Magnitude', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='?=0.5 (optimal)')
    axes[0].legend(loc='best')
    
    # Add value labels
    for i, (w, p) in enumerate(zip(weights, pert_norms)):
        axes[0].annotate(f'{p:.3f}', (w, p), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=8)
    
    # SSIM
    axes[1].plot(weights, ssim_scores, 'go-', linewidth=2, markersize=8, markerfacecolor='white')
    axes[1].set_xlabel('Frequency Weight ?')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('(b) Effect on Perceptual Quality', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[1].set_ylim(0.97, 0.98)
    
    for i, (w, s) in enumerate(zip(weights, ssim_scores)):
        axes[1].annotate(f'{s:.4f}', (w, s), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=8)
    
    # PSNR
    axes[2].plot(weights, psnr_values, 'mo-', linewidth=2, markersize=8, markerfacecolor='white')
    axes[2].set_xlabel('Frequency Weight ?')
    axes[2].set_ylabel('PSNR (dB)')
    axes[2].set_title('(c) Effect on Signal-to-Noise Ratio', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[2].set_ylim(32, 35)
    
    for i, (w, p) in enumerate(zip(weights, psnr_values)):
        axes[2].annotate(f'{p:.2f}', (w, p), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=8)
    
    plt.suptitle('Ablation Study: Frequency Weight', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig4_ablation_freq_weight.pdf', bbox_inches='tight')
    plt.savefig(save_dir / 'fig4_ablation_freq_weight.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ? Figure 4: Frequency weight ablation generated")

def fig5_epsilon_analysis(results, save_dir):
    """Figure 5: Effect of epsilon on attack performance"""
    
    ablation = extract_ablation_data(results)
    if ablation is None:
        print("  ? Could not find ablation data, skipping fig5")
        return
    
    # Extract epsilon data
    epsilons = []
    pert_norms = []
    ssim_scores = []
    
    for key, value in ablation.items():
        if key.startswith('epsilon_'):
            e = float(key.split('_')[1])
            epsilons.append(e)
            pert_norms.append(value['l2_norm'])
            ssim_scores.append(value['ssim'])
    
    if not epsilons:
        print("  ? No epsilon data found")
        return
    
    # Sort by epsilon
    sorted_idx = np.argsort(epsilons)
    epsilons = np.array(epsilons)[sorted_idx]
    pert_norms = np.array(pert_norms)[sorted_idx]
    ssim_scores = np.array(ssim_scores)[sorted_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Perturbation norm scaling
    axes[0].plot(epsilons, pert_norms, 'bo-', linewidth=2, markersize=8, markerfacecolor='white', label='Observed')
    axes[0].set_xlabel('Perturbation Budget e')
    axes[0].set_ylabel('L2 Perturbation Norm')
    axes[0].set_title('(a) Perturbation Scaling', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add ideal linear scaling line
    x_fit = np.array([0, max(epsilons) * 1.2])
    if len(epsilons) > 1:
        slope = pert_norms[1] / epsilons[1]
        y_fit = x_fit * slope
        axes[0].plot(x_fit, y_fit, 'r--', alpha=0.5, linewidth=1.5, label='Ideal Linear')
    axes[0].legend(loc='upper left')
    
    # Add value labels
    for i, (e, p) in enumerate(zip(epsilons, pert_norms)):
        axes[0].annotate(f'{p:.3f}', (e, p), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=8)
    
    # SSIM degradation
    axes[1].plot(epsilons, ssim_scores, 'go-', linewidth=2, markersize=8, markerfacecolor='white')
    axes[1].set_xlabel('Perturbation Budget e')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('(b) Perceptual Quality Degradation', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.8, 1.0)
    
    for i, (e, s) in enumerate(zip(epsilons, ssim_scores)):
        axes[1].annotate(f'{s:.3f}', (e, s), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=8)
    
    plt.suptitle('Effect of Perturbation Budget on Attack Performance', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig5_epsilon_analysis.pdf', bbox_inches='tight')
    plt.savefig(save_dir / 'fig5_epsilon_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ? Figure 5: Epsilon analysis generated")

def fig6_iteration_analysis(results, save_dir):
    """Figure 6: Effect of iterations on attack performance"""
    
    ablation = extract_ablation_data(results)
    if ablation is None:
        print("  ? Could not find ablation data, skipping fig6")
        return
    
    # Extract iteration data
    iterations = []
    pert_norms = []
    times = []
    
    for key, value in ablation.items():
        if key.startswith('iterations_'):
            i = int(key.split('_')[1])
            iterations.append(i)
            pert_norms.append(value['l2_norm'])
            times.append(value['time'])
    
    if not iterations:
        print("  ? No iteration data found")
        return
    
    # Sort by iterations
    sorted_idx = np.argsort(iterations)
    iterations = np.array(iterations)[sorted_idx]
    pert_norms = np.array(pert_norms)[sorted_idx]
    times = np.array(times)[sorted_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Perturbation norm convergence
    axes[0].plot(iterations, pert_norms, 'bo-', linewidth=2, markersize=8, markerfacecolor='white')
    axes[0].set_xlabel('Number of Iterations')
    axes[0].set_ylabel('L2 Perturbation Norm')
    axes[0].set_title('(a) Perturbation Convergence', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=20, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='20 iterations (optimal)')
    axes[0].legend(loc='lower right')
    
    # Add value labels
    for i, (it, p) in enumerate(zip(iterations, pert_norms)):
        axes[0].annotate(f'{p:.3f}', (it, p), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=8)
    
    # Time cost
    axes[1].plot(iterations, times, 'mo-', linewidth=2, markersize=8, markerfacecolor='white')
    axes[1].set_xlabel('Number of Iterations')
    axes[1].set_ylabel('Execution Time (seconds)')
    axes[1].set_title('(b) Computational Cost', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=20, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    
    for i, (it, t) in enumerate(zip(iterations, times)):
        axes[1].annotate(f'{t:.1f}s', (it, t), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=8)
    
    plt.suptitle('Effect of Iterations on Attack Performance', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig6_iteration_analysis.pdf', bbox_inches='tight')
    plt.savefig(save_dir / 'fig6_iteration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ? Figure 6: Iteration analysis generated")

def fig7_radar_chart(results, save_dir):
    """Figure 7: Radar chart comparing attack methods"""
    
    df = extract_comparison_data(results)
    if df is None:
        print("  ? Could not find comparison data, skipping fig7")
        return
    
    # Select key attacks
    attack_order = ['fgsm', 'pgd', 'joint']
    attack_labels = ['FGSM', 'PGD', 'Ours (Joint)']
    
    # Filter data
    df_filtered = df[df['attack'].isin(attack_order)].copy()
    if len(df_filtered) == 0:
        print("  ? No matching attacks found for radar chart")
        return
    
    # Normalize metrics for radar chart
    metrics = ['success_rate', 'ssim', 'psnr', 'l2_norm']
    metric_labels = ['Success Rate', 'SSIM', 'PSNR', 'Low Perturbation']
    
    # For l2_norm, lower is better, so invert
    max_norm = df_filtered['l2_norm'].max()
    df_filtered['l2_norm_inv'] = 1 - (df_filtered['l2_norm'] / max_norm)
    
    # Normalize PSNR to 0-1
    max_psnr = df_filtered['psnr'].max()
    min_psnr = df_filtered['psnr'].min()
    df_filtered['psnr_norm'] = (df_filtered['psnr'] - min_psnr) / (max_psnr - min_psnr + 1e-8)
    
    # Set up radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    colors = [COLORS['fgsm'], COLORS['pgd'], COLORS['joint']]
    
    for i, attack in enumerate(attack_order):
        attack_data = df_filtered[df_filtered['attack'] == attack].iloc[0]
        
        # Get values for each metric
        values = [
            attack_data['success_rate'],
            attack_data['ssim'],
            attack_data['psnr_norm'],
            attack_data['l2_norm_inv'],
        ]
        values += values[:1]  # Close the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=attack_labels[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    
    # Set y limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax.set_title('Attack Performance Radar Chart\n(Higher values are better)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fig7_radar_chart.pdf', bbox_inches='tight')
    plt.savefig(save_dir / 'fig7_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ? Figure 7: Radar chart generated")

def fig8_perceptual_tradeoff(results, save_dir):
    """Figure 8: Scatter plot showing trade-off between perturbation norm and perceptual quality"""
    
    df = extract_comparison_data(results)
    if df is None:
        print("  ? Could not find comparison data, skipping fig8")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    for _, row in df.iterrows():
        attack = row['attack']
        color = COLORS.get(attack, '#777777')
        
        # Determine marker size based on method (ours larger)
        if attack in ['sequential', 'joint', 'adaptive']:
            size = 120
            edgecolor = 'black'
            linewidth = 1.5
        else:
            size = 80
            edgecolor = None
            linewidth = 0.5
        
        # Get display name
        if 'attack_name' in row and pd.notna(row['attack_name']):
            label = row['attack_name']
        else:
            label = attack
        
        scatter = ax.scatter(row['l2_norm'], row['ssim'], s=size, c=color, 
                           edgecolor=edgecolor, linewidth=linewidth, alpha=0.8, zorder=5)
        
        # Add label
        ax.annotate(label, (row['l2_norm'], row['ssim']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, edgecolor='none'))
    
    # Add quadrant lines
    ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.97, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Shade optimal region
    ax.fill_between([0, 1.5], 0.97, 1.0, alpha=0.1, color='green', label='Optimal Region')
    
    ax.set_xlabel('L2 Perturbation Norm (lower is better)', fontsize=12)
    ax.set_ylabel('SSIM (higher is better)', fontsize=12)
    ax.set_title('Perturbation Magnitude vs Perceptual Quality', fontsize=14, fontweight='bold')
    ax.set_xlim(1.3, 1.7)
    ax.set_ylim(0.955, 0.985)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fig8_perceptual_tradeoff.pdf', bbox_inches='tight')
    plt.savefig(save_dir / 'fig8_perceptual_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ? Figure 8: Perceptual trade-off generated")

def generate_all_plots():
    """Generate all publication figures"""
    
    print("="*80)
    print("GENERATING PUBLICATION FIGURES FOR SPATIAL-FREQUENCY ATTACK PAPER")
    print("="*80)
    
    # Create output directory
    save_dir = Path('./paper_figures')
    save_dir.mkdir(exist_ok=True)
    
    # Change to the directory containing your result files
    # You may need to update this path
    results_dir = Path('.')  # Current directory
    
    print(f"\nLooking for result files in: {results_dir.absolute()}")
    
    # Load results
    results = load_results()
    
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    # Generate each figure
    fig1_attack_comparison(results, save_dir)
    fig2_loss_convergence(results, save_dir)
    fig3_cross_dataset_comparison(results, save_dir)
    fig4_ablation_freq_weight(results, save_dir)
    fig5_epsilon_analysis(results, save_dir)
    fig6_iteration_analysis(results, save_dir)
    fig7_radar_chart(results, save_dir)
    fig8_perceptual_tradeoff(results, save_dir)
    
    print(f"\n? All figures saved to {save_dir.absolute()}/")
    print("\nFiles generated:")
    for f in sorted(save_dir.glob('*')):
        print(f"  - {f.name}")

if __name__ == "__main__":
    generate_all_plots()