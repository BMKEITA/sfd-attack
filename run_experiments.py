#!/usr/bin/env python3
"""
Script to run all experiments for the spatial-frequency attack paper.
Author: Your Name
"""

import argparse
import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from experiment import Experiment
    from config import Config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nMake sure all required files exist:")
    print("  - config.py")
    print("  - experiment.py")
    print("  - trainer.py")
    print("  - data_loader.py")
    print("  - models.py")
    print("  - frequency_processor.py")
    print("  - attacks/hybrid.py")
    print("  - attacks/sota.py")
    print("  - evaluator.py")
    print("  - visualization.py")
    print("  - defenses.py")
    print("  - metrics.py")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run spatial-frequency attack experiments')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'imagenet', 'gtsrb', 'medical'],
                        help='Dataset to use')
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--no-vis', action='store_true', help='Skip visualizations')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    
    # Update config
    Config.DEFAULT_DATASET = args.dataset
    
    if args.quick:
        Config.NUM_SAMPLES = 10
        Config.EPSILONS = [0.03]
        Config.ITERATIONS = 10
        Config.DEFAULT_MODELS = ['resnet18']
        print("\n  Running QUICK mode with reduced samples and iterations")
    
    print("="*80)
    print("SPATIAL-FREQUENCY DOMAIN ADVERSARIAL ATTACK")
    print("Comprehensive Experiment Suite")
    print("="*80)
    
    try:
        experiment = Experiment(name=args.name)
        results = experiment.run()
        
        print("\n" + "="*80)
        print(" ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print(f" Results saved in: {experiment.results_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"\n Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()