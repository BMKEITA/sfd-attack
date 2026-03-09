#!/usr/bin/env python3
"""
Cross-dataset evaluation script.
Run: python cross_dataset_experiment.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from tabulate import tabulate

from config import Config
from data_loader import DataLoaderFactory
from models import ModelFactory, SimpleCNN
from frequency_processor import HVSFrequencyProcessor
from attacks.hybrid import JointHybridAttack
from attacks.sota import PGD, FGSM
from evaluator import AttackEvaluator

def run_cross_dataset_experiment():
    """Run attacks on multiple datasets"""
    
    print("="*80)
    print("CROSS-DATASET GENERALIZATION EXPERIMENT")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize components
    data_loader = DataLoaderFactory()
    
    # Datasets to evaluate - use only available datasets
    datasets = [
        ('cifar10', 'CIFAR-10', 10, (32, 32), True),  # Always available
        ('gtsrb', 'GTSRB', 43, (32, 32), False),      # May need download
        ('imagenet_subset', 'ImageNet-100', 100, (224, 224), False),  # Needs manual download
    ]
    
    # Attacks to evaluate
    attacks_to_test = [
        ('FGSM', lambda m, d, f=None: FGSM(m, epsilon=0.03, device=d)),
        ('PGD', lambda m, d, f=None: PGD(m, epsilon=0.03, iterations=20, device=d)),
        ('Ours (Joint)', lambda m, d, f: JointHybridAttack(m, f, epsilon=0.03, iterations=20, device=d)),
    ]
    
    results = []
    
    for dataset_key, dataset_name, num_classes, img_size, auto_download in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*60}")
        
        # Try to load dataset
        try:
            test_loader = data_loader.get_dataset(dataset_key, train=False, num_samples=100)
            print(f"  Successfully loaded {dataset_name}")
        except Exception as e:
            print(f"  Could not load {dataset_name}: {e}")
            print(f"  Skipping {dataset_name}...")
            continue
        
        # Load model for this dataset
        try:
            if dataset_key == 'imagenet_subset':
                # For ImageNet, use pretrained model
                model = ModelFactory.get_model('resnet18', num_classes=100, pretrained=True)
            else:
                # For small datasets, use simple CNN
                model = SimpleCNN(num_classes=num_classes)
            
            model = model.to(device)
            model.eval()
            print(f"  Model loaded successfully")
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue
        
        # Initialize frequency processor with correct image size
        freq_processor = HVSFrequencyProcessor(image_size=img_size, device=device)
        
        # Get a fixed batch for consistent evaluation
        try:
            test_iter = iter(test_loader)
            images, labels = next(test_iter)
            images, labels = images[:20].to(device), labels[:20].to(device)
        except Exception as e:
            print(f"  Error loading batch: {e}")
            continue
        
        # Get clean accuracy
        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            clean_acc = (preds == labels).float().mean().item()
        
        print(f"  Clean accuracy: {clean_acc:.4f}")
        
        # Evaluate each attack
        for attack_name, attack_builder in attacks_to_test:
            print(f"\n  Testing {attack_name}...")
            
            try:
                if attack_name == 'Ours (Joint)':
                    attack = attack_builder(model, device, freq_processor)
                else:
                    attack = attack_builder(model, device, None)
                
                evaluator = AttackEvaluator(model, device=device)
                
                start_time = time.time()
                result, adv_images = evaluator.evaluate_attack(attack, images, labels, batch_size=20)
                elapsed = time.time() - start_time
                
                result_row = {
                    'dataset': dataset_name,
                    'attack': attack_name,
                    'clean_acc': clean_acc,
                    'success_rate': result['success_rate'],
                    'pert_norm': result['l2_norm'],
                    'ssim': result['ssim'],
                    'psnr': result['psnr'],
                    'lpips': result['lpips'],
                    'time': elapsed
                }
                
                results.append(result_row)
                
                print(f"    Success: {result['success_rate']:.4f}")
                print(f"    Pert Norm: {result['l2_norm']:.4f}")
                print(f"    SSIM: {result['ssim']:.4f}")
                print(f"    PSNR: {result['psnr']:.2f} dB")
                
            except Exception as e:
                print(f"    Error in {attack_name}: {e}")
                continue
    
    if not results:
        print("\nNo results obtained. Please check dataset availability.")
        return None
    
    # Create summary table
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("CROSS-DATASET RESULTS SUMMARY")
    print("="*80)
    
    # Pivot table for better visualization
    try:
        pivot_success = df.pivot(index='attack', columns='dataset', values='success_rate')
        pivot_pert = df.pivot(index='attack', columns='dataset', values='pert_norm')
        pivot_ssim = df.pivot(index='attack', columns='dataset', values='ssim')
        
        print("\n--- Attack Success Rates ---")
        print(tabulate(pivot_success, headers='keys', tablefmt='grid', floatfmt='.4f'))
        
        print("\n--- Perturbation Norms ---")
        print(tabulate(pivot_pert, headers='keys', tablefmt='grid', floatfmt='.4f'))
        
        print("\n--- SSIM Scores ---")
        print(tabulate(pivot_ssim, headers='keys', tablefmt='grid', floatfmt='.4f'))
    except:
        print("\n--- Raw Results ---")
        print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Save results
    output_dir = Path('./cross_dataset_results')
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / 'cross_dataset_results.csv', index=False)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(df)
    with open(output_dir / 'cross_dataset_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\nResults saved to {output_dir}")
    
    return df

def generate_latex_table(df):
    """Generate LaTeX table for paper"""
    
    latex = "\\begin{table}[htbp]\n"
    latex += "\\caption{Attack Performance Across Different Datasets}\n"
    latex += "\\label{tab:cross_dataset}\n"
    latex += "\\centering\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{lccccc}\n"
    latex += "\\toprule\n"
    latex += "Dataset & Attack & Success & Pert Norm & SSIM & PSNR \\\\\n"
    latex += "\\midrule\n"
    
    for _, row in df.iterrows():
        latex += f"{row['dataset']} & {row['attack']} & "
        latex += f"{row['success_rate']:.4f} & {row['pert_norm']:.4f} & "
        latex += f"{row['ssim']:.4f} & {row['psnr']:.2f} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

if __name__ == "__main__":
    run_cross_dataset_experiment()