"""
Comprehensive evaluation pipeline for attacks and defenses.
Author: Your Name
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import json

from metrics import PerceptualMetrics, AttackMetrics
from defenses import DefenseEvaluator

class AttackEvaluator:
    """Comprehensive attack evaluation"""
    
    def __init__(self, model, device=None, logger=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger
        self.perceptual_metrics = PerceptualMetrics(device)
        self.defense_evaluator = DefenseEvaluator(device)
        self.model.to(self.device)
        self.model.eval()
    
    
    def evaluate_attack(self, attack, images, labels, batch_size=10):
        """Evaluate a single attack"""
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Ensure we're using the right number of samples
        if images.size(0) > batch_size:
            images = images[:batch_size]
            labels = labels[:batch_size]
        
        start_time = time.time()
        adv_images, stats = attack.attack(images, labels)
        elapsed_time = time.time() - start_time
        
        # Ensure adv_images has same shape as images
        assert adv_images.shape == images.shape, f"Shape mismatch: {adv_images.shape} vs {images.shape}"
        
        with torch.no_grad():
            clean_outputs = self.model(images)
            adv_outputs = self.model(adv_images)
            
            clean_preds = torch.argmax(clean_outputs, dim=1)
            adv_preds = torch.argmax(adv_outputs, dim=1)
            
            clean_probs = F.softmax(clean_outputs, dim=1)
            adv_probs = F.softmax(adv_outputs, dim=1)
        
        success_rate = (adv_preds != labels).float().mean().item()
        
        # Safe confidence drop calculation
        clean_conf = torch.gather(clean_probs, 1, labels.unsqueeze(1)).squeeze()
        adv_conf = torch.gather(adv_probs, 1, labels.unsqueeze(1)).squeeze()
        confidence_drop = (clean_conf - adv_conf).mean().item()
        
        perturbations = adv_images - images
        
        # Detach tensors before passing to perceptual metrics
        clean_img_detached = images[0].detach()
        adv_img_detached = adv_images[0].detach()
        pert_detached = perturbations.detach()
        
        # Compute perceptual metrics for first image only (for speed)
        perceptual = self.perceptual_metrics.compute_all(
            clean_img_detached, 
            adv_img_detached, 
            pert_detached
        )
        
        convergence = 0
        if stats and 'success_rate' in stats and len(stats['success_rate']) > 0:
            convergence = AttackMetrics.convergence_speed(stats.get('success_rate', []))
        
        results = {
            'attack_name': getattr(attack, 'variant_name', type(attack).__name__),
            'success_rate': success_rate,
            'confidence_drop': confidence_drop,
            'convergence_iter': convergence,
            'time': elapsed_time,
            **perceptual,
            'stats': stats
        }
        
        return results, adv_images

    def evaluate_transferability(self, attack, source_model, target_model, images, labels):
        """Evaluate attack transferability across models"""
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        source_model.eval()
        target_model.eval()
        
        adv_images, _ = attack.attack(images, labels)
        
        with torch.no_grad():
            source_preds = torch.argmax(source_model(adv_images), dim=1)
            target_preds = torch.argmax(target_model(adv_images), dim=1)
        
        source_success = AttackMetrics.success_rate(source_preds, labels)
        target_success = AttackMetrics.success_rate(target_preds, labels)
        transfer_rate = target_success / (source_success + 1e-8)
        
        return {
            'source_success': source_success,
            'target_success': target_success,
            'transfer_rate': transfer_rate
        }
    
    def evaluate_defenses(self, attack, images, labels, defenses):
        """Evaluate attack against multiple defenses"""
        adv_images, _ = attack.attack(images, labels)
        
        results = {}
        for defense in defenses:
            if self.logger:
                self.logger._write_log(f"    Testing defense: {defense['name']}")
            
            defense_func = getattr(self.defense_evaluator, defense['key'], None)
            if defense_func:
                result = self.defense_evaluator.test_defense(
                    model=self.model,
                    clean_images=images,
                    adv_images=adv_images,
                    labels=labels,
                    defense_name=defense['name'],
                    defense_func=defense_func,
                    **defense['params']
                )
                results[defense['name']] = result
        
        return results
    
    def batch_evaluate(self, attack, dataloader, num_batches=None):
        """Evaluate attack on entire dataset"""
        attack_successes = []
        times = []
        
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if num_batches and i >= num_batches:
                break
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            start_time = time.time()
            adv_images, _ = attack.attack(images, labels)
            elapsed_time = time.time() - start_time
            
            with torch.no_grad():
                adv_preds = torch.argmax(self.model(adv_images), dim=1)
                success = (adv_preds != labels).float().mean().item()
            
            attack_successes.append(success)
            times.append(elapsed_time / images.size(0))
        
        return {
            'avg_success': np.mean(attack_successes),
            'std_success': np.std(attack_successes),
            'avg_time_per_sample': np.mean(times)
        }


class ComparisonEvaluator:
    """Compare multiple attacks and generate tables"""
    
    def __init__(self, save_dir=None):
        self.save_dir = Path(save_dir) if save_dir else Path('./comparison_results')
        self.save_dir.mkdir(exist_ok=True)
        self.results = []
    
    def add_result(self, attack_name, dataset, model, metrics):
        """Add evaluation result"""
        result = {
            'attack': attack_name,
            'dataset': dataset,
            'model': model,
            **metrics
        }
        self.results.append(result)
    
    def save_results(self, filename='comparison_results.json'):
        """Save results to JSON"""
        with open(self.save_dir / filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save as CSV
        df = pd.DataFrame(self.results)
        df.to_csv(self.save_dir / filename.replace('.json', '.csv'), index=False)
    
    def generate_latex_table(self, metrics=['success_rate', 'ssim', 'psnr', 'time']):
        """Generate LaTeX table for paper"""
        df = pd.DataFrame(self.results)
        
        attacks = df['attack'].unique()
        datasets = df['dataset'].unique()
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\caption{Attack Performance Comparison}\n"
        latex += "\\label{tab:attack_comparison}\n"
        latex += "\\centering\n"
        latex += "\\small\n"
        
        # Build table header
        header = "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
        header += "\\toprule\n"
        header += "Attack & " + " & ".join([m.replace('_', ' ').title() for m in metrics]) + " \\\\\n"
        header += "\\midrule\n"
        
        latex += header
        
        for attack in attacks:
            attack_data = df[df['attack'] == attack].iloc[0]
            row = f"{attack} "
            for metric in metrics:
                value = attack_data.get(metric, 0)
                if isinstance(value, float):
                    row += f"& {value:.4f} "
                else:
                    row += f"& {value} "
            row += "\\\\\n"
            latex += row
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        # Save to file
        with open(self.save_dir / 'comparison_table.tex', 'w') as f:
            f.write(latex)
        
        return latex