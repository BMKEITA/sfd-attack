"""
Main experiment pipeline for comprehensive evaluation.
Author: Your Name
"""

import torch
import numpy as np
import random
from pathlib import Path
import json
import pickle

from config import Config
from data_loader import DataLoaderFactory
from models import ModelFactory, SimpleCNN
from frequency_processor import HVSFrequencyProcessor
from attacks.hybrid import SequentialHybridAttack, JointHybridAttack, AdaptiveBandAttack
from attacks.sota import FGSM, PGD, SFA, FACL
from evaluator import AttackEvaluator, ComparisonEvaluator
from visualization import PublicationVisualizer
from defenses import DefenseEvaluator
from metrics import PerceptualMetrics
from trainer import ModelTrainer

class Experiment:
    """Main experiment class for comprehensive evaluation"""
    
    def __init__(self, name=None):
        Config.print_summary()
        
        self.name = name or Config.EXPERIMENT_ID
        self.device = Config.DEVICE
        self.results_dir = Config.get_experiment_path()
        
        # Set random seeds
        self._set_seed(Config.SEED)
        
        # Initialize components
        self.data_loader = DataLoaderFactory()
        self.visualizer = PublicationVisualizer(save_dir=self.results_dir / 'figures')
        self.evaluator = None
        self.comparison = ComparisonEvaluator(save_dir=self.results_dir)
        
        print(f"\nExperiment directory: {self.results_dir}")
    
    def _set_seed(self, seed):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def run(self):
        """Run complete experiment pipeline"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE EXPERIMENT")
        print("="*80)
        
        # Step 1: Load data
        print("\n1. Loading datasets...")
        datasets = self._load_datasets()
        
        # Step 2: Load/train models
        print("\n2. Loading models...")
        models = self._load_models()
        
        # Step 3: Initialize frequency processor
        print("\n3. Initializing frequency processor...")
        freq_processor = HVSFrequencyProcessor(image_size=(32, 32), device=self.device)
        
        # Step 4: Create attacks
        print("\n4. Creating attacks...")
        attacks = self._create_attacks(models['cifar10']['resnet18'], freq_processor)
        
        # Step 5: Evaluate on CIFAR-10
        print("\n5. Evaluating on CIFAR-10...")
        cifar10_results = self._evaluate_dataset(
            datasets['cifar10'],
            models['cifar10'],
            attacks,
            freq_processor
        )
        
        # Step 6: Transferability analysis
        print("\n6. Analyzing transferability...")
        transfer_results = self._analyze_transferability(
            datasets['cifar10'],
            models['cifar10'],
            attacks
        )
        
        # Step 7: Defense evaluation
        print("\n7. Evaluating defenses...")
        defense_results = self._evaluate_defenses(
            datasets['cifar10'],
            models['cifar10']['resnet18'],
            attacks['joint']
        )
        
        # Step 8: Ablation studies
        print("\n8. Running ablation studies...")
        ablation_results = self._run_ablation(
            datasets['cifar10'],
            models['cifar10']['resnet18'],
            freq_processor
        )
        
        # Step 9: Save all results
        print("\n9. Saving results...")
        self._save_all_results({
            'cifar10': cifar10_results,
            'transfer': transfer_results,
            'defenses': defense_results,
            'ablation': ablation_results
        })
        
        # Step 10: Generate visualizations
        print("\n10. Generating visualizations...")
        self._generate_visualizations(
            cifar10_results,
            defense_results,
            transfer_results,
            ablation_results
        )
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"Results saved in: {self.results_dir}")
        print("="*80)
        
        return {
            'cifar10': cifar10_results,
            'transfer': transfer_results,
            'defenses': defense_results,
            'ablation': ablation_results
        }
    
    def _load_datasets(self):
        """Load all datasets"""
        datasets = {}
        
        # CIFAR-10
        print("  Loading CIFAR-10...")
        train_loader = self.data_loader.get_dataset('cifar10', train=True)
        test_loader = self.data_loader.get_dataset('cifar10', train=False, num_samples=1000)
        datasets['cifar10'] = {'train': train_loader, 'test': test_loader}
        
        # Get sample batch for consistent evaluation
        sample_images, sample_labels = self.data_loader.get_sample_batch('cifar10', num_samples=50)
        datasets['cifar10']['samples'] = (sample_images, sample_labels)
        
        # ImageNet (validation subset)
        try:
            print("  Loading ImageNet...")
            imagenet_loader = self.data_loader.get_dataset('imagenet', train=False, num_samples=500)
            datasets['imagenet'] = {'test': imagenet_loader}
        except:
            print("  ImageNet not available, skipping...")
        
        return datasets
    
    def _load_models(self):
        """Load all models"""
        models = {'cifar10': {}}
        
        # Load pretrained models for CIFAR-10
        for arch in Config.DEFAULT_MODELS:
            print(f"  Loading {arch}...")
            try:
                model = ModelFactory.get_model(arch, num_classes=10, pretrained=True)
                model = model.to(self.device)
                model.eval()
                models['cifar10'][arch] = model
            except Exception as e:
                print(f"  Error loading {arch}: {e}")
        
        # Train a simple CNN if needed
        if 'resnet18' not in models['cifar10']:
            print("  Training simple CNN...")
            trainer = ModelTrainer(device=self.device)
            model, history = trainer.train_model(epochs=10)
            models['cifar10']['cnn'] = model
        
        return models
    
    def _create_attacks(self, model, freq_processor):
        """Create all attack instances"""
        attacks = {}
        
        # Your variants
        attacks['sequential'] = SequentialHybridAttack(
            model, freq_processor, epsilon=0.03, iterations=20
        )
        attacks['joint'] = JointHybridAttack(
            model, freq_processor, epsilon=0.03, iterations=20, freq_weight=0.5
        )
        attacks['adaptive'] = AdaptiveBandAttack(
            model, freq_processor, epsilon=0.03, iterations=20, top_k=2
        )
        
        # SOTA baselines
        attacks['fgsm'] = FGSM(model, epsilon=0.03)
        attacks['pgd'] = PGD(model, epsilon=0.03, iterations=20)
        attacks['sfa'] = SFA(model, freq_processor, epsilon=0.03, iterations=20)
        attacks['facl'] = FACL(model, freq_processor, epsilon=0.03, iterations=20)
        
        return attacks
    
    def _evaluate_dataset(self, dataset, models, attacks, freq_processor):
        """Evaluate all attacks on a dataset"""
        images, labels = dataset['samples']
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n  Evaluating on {model_name}...")
            evaluator = AttackEvaluator(model, device=self.device)
            
            model_results = {}
            for attack_name, attack in attacks.items():
                print(f"    Testing {attack_name}...")
                result, adv_images = evaluator.evaluate_attack(attack, images, labels)
                model_results[attack_name] = result
                
                # Add to comparison
                self.comparison.add_result(
                    attack_name=attack_name,
                    dataset='cifar10',
                    model=model_name,
                    metrics={k: v for k, v in result.items() if k != 'stats'}
                )
            
            results[model_name] = model_results
        
        return results

    def _analyze_transferability(self, dataset, models, attacks):
        """Analyze attack transferability across models"""
        images, labels = dataset['samples']
        
        model_names = list(models.keys())
        transfer_matrix = np.zeros((len(attacks), len(model_names), len(model_names)))
        
        results = {}
        
        for a_idx, (attack_name, attack) in enumerate(attacks.items()):
            print(f"\n  Analyzing {attack_name} transferability...")
            attack_results = {}
            
            for s_idx, source_name in enumerate(model_names):
                source_model = models[source_name]
                evaluator = AttackEvaluator(source_model, device=self.device)
                
                source_result, adv_images = evaluator.evaluate_attack(attack, images, labels)
                
                for t_idx, target_name in enumerate(model_names):
                    if source_name == target_name:
                        transfer_matrix[a_idx, s_idx, t_idx] = 1.0
                        continue
                    
                    target_model = models[target_name]
                    target_model.eval()
                    
                    with torch.no_grad():
                        target_preds = torch.argmax(target_model(adv_images), dim=1)
                        target_success = (target_preds != labels.to(self.device)).float().mean().item()
                    
                    transfer_matrix[a_idx, s_idx, t_idx] = target_success
            
            attack_results['matrix'] = transfer_matrix[a_idx].tolist()
            attack_results['avg_transfer'] = np.mean([
                transfer_matrix[a_idx, s_idx, t_idx] 
                for s_idx in range(len(model_names))
                for t_idx in range(len(model_names))
                if s_idx != t_idx
            ])
            
            results[attack_name] = attack_results
        
        return results
    
    def _evaluate_defenses(self, dataset, model, attack):
        """Evaluate attack against defenses"""
        images, labels = dataset['samples'][:20]  # Use subset for speed
        
        evaluator = AttackEvaluator(model, device=self.device)
        defense_results = evaluator.evaluate_defenses(
            attack, images, labels, Config.DEFENSES
        )
        
        return defense_results
    
    def _run_ablation(self, dataset, model, freq_processor):
        """Run ablation studies"""
        images, labels = dataset['samples'][:10]
        
        results = {}
        
        # Ablate frequency weight
        print("  Ablating frequency weight...")
        freq_weights = [0.0, 0.3, 0.5, 0.7, 1.0]
        for w in freq_weights:
            attack = JointHybridAttack(
                model, freq_processor, epsilon=0.03, iterations=20, freq_weight=w
            )
            evaluator = AttackEvaluator(model, device=self.device)
            result, _ = evaluator.evaluate_attack(attack, images, labels)
            results[f'freq_weight_{w}'] = result
        
        # Ablate epsilon
        print("  Ablating epsilon...")
        epsilons = [0.01, 0.03, 0.05, 0.1]
        for eps in epsilons:
            attack = JointHybridAttack(
                model, freq_processor, epsilon=eps, iterations=20, freq_weight=0.5
            )
            evaluator = AttackEvaluator(model, device=self.device)
            result, _ = evaluator.evaluate_attack(attack, images, labels)
            results[f'epsilon_{eps}'] = result
        
        # Ablate iterations
        print("  Ablating iterations...")
        iters_list = [5, 10, 20, 50]
        for iters in iters_list:
            attack = JointHybridAttack(
                model, freq_processor, epsilon=0.03, iterations=iters, freq_weight=0.5
            )
            evaluator = AttackEvaluator(model, device=self.device)
            result, _ = evaluator.evaluate_attack(attack, images, labels)
            results[f'iterations_{iters}'] = result
        
        return results
    
    def _save_all_results(self, all_results):
        """Save all results to disk"""
        # Save as JSON
        with open(self.results_dir / 'results.json', 'w') as f:
            # Convert numpy arrays to lists
            json_results = self._convert_to_serializable(all_results)
            json.dump(json_results, f, indent=2)
        
        # Save as pickle
        with open(self.results_dir / 'results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        # Save comparison table
        self.comparison.save_results('comparison_results.json')
        
        print(f"  Results saved to {self.results_dir}")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def _generate_visualizations(self, cifar10_results, defense_results, 
                                 transfer_results, ablation_results):
        """Generate all visualizations"""
        
        # Attack progress plots
        progress_stats = {}
        for model_name, model_results in cifar10_results.items():
            for attack_name, attack_result in model_results.items():
                if 'stats' in attack_result:
                    progress_stats[f"{model_name}_{attack_name}"] = attack_result['stats']
        
        if progress_stats:
            self.visualizer.plot_attack_progress(
                progress_stats, 
                title="Attack Progress Comparison",
                filename="attack_progress.png"
            )
        
        # Variant comparison
        variant_results = {}
        for model_name, model_results in cifar10_results.items():
            if model_name == 'resnet18':
                for attack_name, attack_result in model_results.items():
                    if attack_name in ['sequential', 'joint', 'adaptive']:
                        variant_results[attack_name] = attack_result
        
        if variant_results:
            self.visualizer.plot_variant_comparison(
                variant_results,
                filename="variant_comparison.png"
            )
        
        # Defense comparison
        if defense_results:
            self.visualizer.plot_defense_comparison(
                defense_results,
                filename="defense_comparison.png"
            )
        
        # Epsilon scaling from ablation
        epsilon_results = {}
        for k, v in ablation_results.items():
            if k.startswith('epsilon_'):
                eps = float(k.split('_')[1])
                epsilon_results[eps] = {'joint': v}
        
        if epsilon_results:
            self.visualizer.plot_epsilon_scaling(
                epsilon_results,
                filename="epsilon_scaling.png"
            )


# ==================== MAIN ====================

if __name__ == "__main__":
    experiment = Experiment()
    results = experiment.run()