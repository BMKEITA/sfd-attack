#!/usr/bin/env python3
"""
Transferability analysis across architectures.
Run: python transferability.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt

from config import Config
from data_loader import DataLoaderFactory
from models import ModelFactory
from frequency_processor import HVSFrequencyProcessor
from attacks.hybrid import JointHybridAttack
from attacks.sota import PGD

def run_transferability_experiment():
    """Evaluate attack transferability across architectures"""
    
    print("="*80)
    print("TRANSFERABILITY ANALYSIS ACROSS ARCHITECTURES")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Define architectures to test
    architectures = [
        ('resnet18', 'CNN (ResNet-18)'),
        ('resnet50', 'CNN (ResNet-50)'),
        ('densenet121', 'CNN (DenseNet-121)'),
        ('vit_base', 'ViT-B/16'),
        ('swin_tiny', 'Swin-T'),
        ('mlp_mixer', 'MLP-Mixer'),
    ]
    
    # Load data
    data_loader = DataLoaderFactory()
    test_loader = data_loader.get_dataset('cifar10', train=False, num_samples=50)
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    images, labels = images[:20].to(device), labels[:20].to(device)
    
    # Load all models
    models = {}
    for arch_key, arch_name in architectures:
        print(f"Loading {arch_name}...")
        try:
            model = ModelFactory.get_model(arch_key, num_classes=10, pretrained=False)
            model = model.to(device)
            model.eval()
            models[arch_key] = model
        except Exception as e:
            print(f"  Error loading {arch_key}: {e}")
    
    # Initialize frequency processor
    freq_processor = HVSFrequencyProcessor(device=device)
    
    # Create attacks
    attacks = {
        'PGD': lambda m: PGD(m, epsilon=0.03, iterations=20, device=device),
        'Ours (Joint)': lambda m: JointHybridAttack(m, freq_processor, epsilon=0.03, iterations=20, device=device)
    }
    
    # Get clean accuracies
    clean_accs = {}
    for arch_key, model in models.items():
        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean().item()
            clean_accs[arch_key] = acc
    
    # Generate adversarial examples on source model (ResNet-18)
    source_key = 'resnet18'
    source_model = models[source_key]
    
    adv_examples = {}
    for attack_name, attack_builder in attacks.items():
        print(f"\nGenerating {attack_name} adversarial examples on {source_key}...")
        attack = attack_builder(source_model)
        
        adv_images = attack.attack(images, labels)[0]
        adv_examples[attack_name] = adv_images
        
        # Verify success on source
        with torch.no_grad():
            source_preds = torch.argmax(source_model(adv_images), dim=1)
            source_success = (source_preds != labels).float().mean().item()
        print(f"  Source success: {source_success:.4f}")
    
    # Test transferability to all target models
    results = []
    
    for target_key, target_model in models.items():
        print(f"\nTesting transfer to {target_key}...")
        
        for attack_name, adv_images in adv_examples.items():
            with torch.no_grad():
                target_preds = torch.argmax(target_model(adv_images), dim=1)
                target_success = (target_preds != labels).float().mean().item()
            
            results.append({
                'source': source_key,
                'target': target_key,
                'attack': attack_name,
                'transfer_success': target_success,
                'clean_acc': clean_accs[target_key]
            })
            
            print(f"  {attack_name}: {target_success:.4f}")
    
    # Create transfer matrix
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("TRANSFERABILITY MATRIX")
    print("="*80)
    
    # Pivot for PGD
    pivot_pgd = df[df['attack'] == 'PGD'].pivot(index='source', columns='target', values='transfer_success')
    print("\n--- PGD Transferability ---")
    print(tabulate(pivot_pgd, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Pivot for Ours
    pivot_ours = df[df['attack'] == 'Ours (Joint)'].pivot(index='source', columns='target', values='transfer_success')
    print("\n--- Ours (Joint) Transferability ---")
    print(tabulate(pivot_ours, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Calculate average transfer (excluding self)
    avg_transfer = {}
    for attack_name in ['PGD', 'Ours (Joint)']:
        attack_df = df[df['attack'] == attack_name]
        avg = attack_df[attack_df['source'] != attack_df['target']]['transfer_success'].mean()
        avg_transfer[attack_name] = avg
    
    print("\n--- Average Transferability (excluding self) ---")
    for attack_name, avg in avg_transfer.items():
        print(f"  {attack_name}: {avg:.4f}")
    
    # Save results
    output_dir = Path('./transferability_results')
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / 'transferability.csv', index=False)
    
    # Generate heatmap
    plot_transfer_heatmap(pivot_ours, output_dir / 'transfer_heatmap.png')
    
    # Generate LaTeX table
    latex_table = generate_transfer_latex(pivot_ours)
    with open(output_dir / 'transfer_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\nResults saved to {output_dir}")
    
    return df

def plot_transfer_heatmap(pivot_df, save_path):
    """Plot transferability heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Create mask for diagonal (self-transfer)
    mask = np.zeros_like(pivot_df.values, dtype=bool)
    np.fill_diagonal(mask, True)
    
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                mask=mask, vmin=0, vmax=1,
                cbar_kws={'label': 'Transfer Success Rate'})
    
    plt.title('Attack Transferability Heatmap\n(Ours Joint Hybrid)', fontsize=14)
    plt.xlabel('Target Model')
    plt.ylabel('Source Model')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_transfer_latex(pivot_df):
    """Generate LaTeX table for transferability"""
    
    latex = "\\begin{table}[htbp]\n"
    latex += "\\caption{Attack Transferability Across Architectures (Our Joint Hybrid)}\n"
    latex += "\\label{tab:transfer}\n"
    latex += "\\centering\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{l" + "c" * len(pivot_df.columns) + "}\n"
    latex += "\\toprule\n"
    latex += "Source $\\rightarrow$ Target & " + " & ".join(pivot_df.columns) + " \\\\\n"
    latex += "\\midrule\n"
    
    for source in pivot_df.index:
        row = f"{source} "
        for target in pivot_df.columns:
            val = pivot_df.loc[source, target]
            if source == target:
                row += f"& \\textbf{{{val:.3f}}} "
            else:
                row += f"& {val:.3f} "
        row += "\\\\\n"
        latex += row
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

if __name__ == "__main__":
    run_transferability_experiment()