# Spatial-Frequency Domain Hybrid Adversarial Attacks with HVS-Aware Constraints

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of the paper "Spatial-Frequency Domain Hybrid Adversarial Attacks with HVS-Aware Constraints".

##  Overview

This repository contains the complete implementation of our novel hybrid adversarial attack that simultaneously optimizes perturbations in both spatial and frequency domains, guided by Human Visual System (HVS) constraints modeled via the Contrast Sensitivity Function (CSF).

### Key Features:
- **Three attack variants**: Sequential Filtering & Injection, Joint Hybrid Loss Optimization, Adaptive Frequency Band Selection
- **HVS-aware frequency constraints** for stealthy perturbations
- **Comprehensive evaluation** on CIFAR-10, GTSRB, and ImageNet-100
- **Extensive ablation studies** on frequency weight, epsilon, and iterations
- **Publication-quality visualization** tools

##  Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sfd-attack.git
cd sfd-attack

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
# Run a quick test
python test_attack.py

# Run comprehensive experiments
python run_experiments.py --quick

# Generate publication figures
python generate_paper_figs.py

