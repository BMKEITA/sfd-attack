#!/usr/bin/env python3
"""
Comprehensive defense evaluation script.
Run: python defense_experiment.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tabulate import tabulate

from config import Config
from data_loader import DataLoaderFactory
from models import ModelFactory
from frequency_processor import HVSFrequencyProcessor
from attacks.hybrid import JointHybridAttack
from attacks.sota import PGD
from defenses import DefenseEvaluator
from evaluator import AttackEvaluator

def run_defense_experiment():
    """Evaluate attacks against various defenses"""
    
    print("="*80)
    print("COMPREHENSIVE DEFENSE EVALUATION")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = ModelFactory.get_model('resnet18', num_classes=10, pretrained=False)
    model = model.to(device)
    model.eval()
    
    # Load data
    data_loader = DataLoaderFactory()
    test_loader = data_loader.get_dataset('cifar10', train=False, num_samples=50)
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    images, labels = images[:20].to(device), labels[:20].to(device)
    
    # Initialize components
    freq_processor = HVSFrequencyProcessor(device=device)
    defense_eval = DefenseEvaluator(device=device)
    
    # Create attacks
    attacks = {
        'PGD': PGD(model, epsilon=0.03, iterations=20, device=device),
        'Ours (Joint)': JointHybridAttack(model, freq_processor, epsilon=0.03, iterations=20, device=device)
    }
    
    # Defenses to evaluate
    defenses = [
        ('No Defense', 'none', {}),
        ('Gaussian Blur (k=3)', 'gaussian_blur', {'kernel_size': 3, 'sigma': 1.0}),
        ('Gaussian Blur (k=5)', 'gaussian_blur', {'kernel_size': 5, 'sigma': 1.5}),
        ('Median Filter (k=3)', 'median_filter', {'kernel_size': 3}),
        ('Median Filter (k=5)', 'median_filter', {'kernel_size': 5}),
        ('JPEG (Q=75)', 'jpeg_compression', {'quality': 75}),
        ('JPEG (Q=50)', 'jpeg_compression', {'quality': 50}),
        ('Bit Depth (4-bit)', 'bit_depth', {'bits': 4}),
        ('Random Noise (s=0.02)', 'random_noise', {'noise_level': 0.02}),
        ('Low-pass Filter', 'frequency_filter', {'filter_type': 'low_pass', 'cutoff': 0.3}),
        ('High-pass Filter', 'frequency_filter', {'filter_type': 'high_pass', 'cutoff': 0.1}),
    ]
    
    results = []
    
    # Generate adversarial examples for each attack
    adv_examples = {}
    for attack_name, attack in attacks.items():
        print(f"\nGenerating adversarial examples with {attack_name}...")
        evaluator = AttackEvaluator(model, device=device)
        result, adv_images = evaluator.evaluate_attack(attack, images, labels)
        adv_examples[attack_name] = adv_images
        print(f"  Success rate: {result['success_rate']:.4f}")
    
    # Test each defense against each attack
    for defense_name, defense_key, defense_params in defenses:
        print(f"\nTesting defense: {defense_name}")
        
        defense_func = getattr(defense_eval, defense_key, None)
        if defense_func is None and defense_key != 'none':
            print(f"  Warning: Defense {defense_key} not found, skipping")
            continue
        
        for attack_name, adv_images in adv_examples.items():
            if defense_key == 'none':
                defended_adv = adv_images
                defended_preds = torch.argmax(model(defended_adv), dim=1)
                defended_success = (defended_preds != labels).float().mean().item()
            else:
                try:
                    defended_adv = defense_func(adv_images, **defense_params)
                    defended_preds = torch.argmax(model(defended_adv), dim=1)
                    defended_success = (defended_preds != labels).float().mean().item()
                except Exception as e:
                    print(f"  Error with {defense_name} on {attack_name}: {e}")
                    defended_success = 1.0
            
            # Original attack success
            orig_preds = torch.argmax(model(adv_images), dim=1)
            orig_success = (orig_preds != labels).float().mean().item()
            
            defense_efficacy = defended_success - orig_success
            
            results.append({
                'defense': defense_name,
                'attack': attack_name,
                'orig_success': orig_success,
                'defended_success': defended_success,
                'defense_efficacy': defense_efficacy
            })
            
            print(f"  {attack_name}: {orig_success:.4f} ? {defended_success:.4f} "
                  f"(?={defense_efficacy:+.4f})")
    
    # Create summary
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("DEFENSE EVALUATION SUMMARY")
    print("="*80)
    
    # Pivot for better visualization
    pivot = df.pivot(index='defense', columns='attack', values='defended_success')
    
    print("\n--- Defended Attack Success Rates ---")
    print(tabulate(pivot, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Save results
    output_dir = Path('./defense_results')
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / 'defense_results.csv', index=False)
    
    # Generate LaTeX table
    latex_table = generate_defense_latex(pivot)
    with open(output_dir / 'defense_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\nResults saved to {output_dir}")
    
    return df

def generate_defense_latex(pivot_df):
    """Generate LaTeX table for defense results"""
    
    latex = "\\begin{table}[htbp]\n"
    latex += "\\caption{Defense Effectiveness Against Different Attacks}\n"
    latex += "\\label{tab:defense}\n"
    latex += "\\centering\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{lcc}\n"
    latex += "\\toprule\n"
    latex += "Defense & PGD & Ours (Joint) \\\\\n"
    latex += "\\midrule\n"
    
    for defense in pivot_df.index:
        latex += f"{defense} & "
        latex += f"{pivot_df.loc[defense, 'PGD']:.4f} & "
        latex += f"{pivot_df.loc[defense, 'Ours (Joint)']:.4f} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

if __name__ == "__main__":
    run_defense_experiment()