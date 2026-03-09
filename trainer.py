"""
Model training utilities for CIFAR-10 and other datasets.
Author: Your Name
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models import SimpleCNN
from config import Config

class ModelTrainer:
    """Trainer for image classification models"""
    
    def __init__(self, device=None, logger=None):
        self.device = device or Config.DEVICE
        self.logger = logger
    
    def train_model(self, model=None, train_loader=None, test_loader=None, 
                   epochs=15, lr=0.001, save_path=None):
        """Train a model on the given dataset"""
        
        if model is None:
            model = SimpleCNN(num_classes=10).to(self.device)
        
        if train_loader is None:
            from data_loader import DataLoaderFactory
            data_loader = DataLoaderFactory()
            train_loader = data_loader.get_dataset('cifar10', train=True)
            test_loader = data_loader.get_dataset('cifar10', train=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': []
        }
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({'loss': train_loss/(total/inputs.size(0)), 
                                  'acc': 100.*correct/total})
            
            epoch_loss = train_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            # Test phase
            test_acc = self.test_model(model, test_loader)
            history['test_acc'].append(test_acc)
            
            scheduler.step()
            
            if self.logger:
                self.logger._write_log(
                    f"Epoch {epoch+1}: Loss={epoch_loss:.3f}, "
                    f"Train Acc={epoch_acc:.1f}%, Test Acc={test_acc:.1f}%"
                )
            else:
                print(f"Epoch {epoch+1}: Loss={epoch_loss:.3f}, "
                      f"Train Acc={epoch_acc:.1f}%, Test Acc={test_acc:.1f}%")
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'history': history,
                'epochs': epochs,
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")
        
        model.eval()
        return model, history
    
    def test_model(self, model, test_loader):
        """Test model accuracy on test set"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total
    
    def train_ensemble(self, num_models=3, train_loader=None, test_loader=None, 
                      epochs=15, save_dir=None):
        """Train an ensemble of models"""
        models = []
        histories = []
        
        for i in range(num_models):
            print(f"\nTraining model {i+1}/{num_models}")
            model = SimpleCNN(num_classes=10).to(self.device)
            model, history = self.train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=epochs,
                save_path=f"{save_dir}/model_{i}.pth" if save_dir else None
            )
            models.append(model)
            histories.append(history)
        
        return models, histories
    
    def train_adversarial(self, model=None, train_loader=None, test_loader=None,
                         attack=None, epsilon=0.03, alpha=0.007, epochs=15):
        """Adversarial training"""
        if model is None:
            model = SimpleCNN(num_classes=10).to(self.device)
        
        if attack is None:
            from attacks.sota import PGD
            attack = PGD(model, epsilon=epsilon, alpha=alpha, iterations=10)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'adv_acc': []
        }
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Adv Train]')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Generate adversarial examples
                adv_inputs = attack.generate(inputs, targets)
                
                # Train on both clean and adversarial
                optimizer.zero_grad()
                outputs_clean = model(inputs)
                outputs_adv = model(adv_inputs)
                
                loss = (criterion(outputs_clean, targets) + 
                       criterion(outputs_adv, targets)) / 2
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs_adv.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({'loss': train_loss/(total/inputs.size(0)), 
                                  'adv_acc': 100.*correct/total})
            
            # Test on adversarial examples
            adv_acc = self.test_adversarial(model, test_loader, attack)
            history['adv_acc'].append(adv_acc)
            
            print(f"Epoch {epoch+1}: Adv Test Acc={adv_acc:.1f}%")
        
        model.eval()
        return model, history
    
    def test_adversarial(self, model, test_loader, attack):
        """Test model accuracy on adversarial examples"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                adv_inputs = attack.generate(inputs, targets)
                outputs = model(adv_inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total
    
    def load_model(self, model_path, model_class=None, num_classes=10):
        """Load a trained model"""
        if model_class is None:
            model_class = SimpleCNN
        
        model = model_class(num_classes=num_classes).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model = None
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0
        
        return self.early_stop