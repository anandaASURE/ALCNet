"""
Analysis and visualization tools for ALCNet compression.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


class CompressionAnalyzer:
    """Analyzer for ALCNet compression patterns.
       Author - Ananda Jana , IISER TVM , Kerala , India

    Args:
        model: Trained ALCNet model
        device (str): Device to use
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
    
    def compute_feature_survival(self, dataloader, max_batches=10):
        """Compute survival probability for input features.
        
        Args:
            dataloader: DataLoader with input data
            max_batches (int): Maximum number of batches to process
        
        Returns:
            survival_probs: Dictionary mapping layer to survival probabilities
        """
        self.model.eval()
        all_scores = {i: [] for i in range(len(self.model.layers) - 1)}
        
        with torch.no_grad():
            for batch_idx, (batch_x, _) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                batch_x = batch_x.to(self.device)
                if batch_x.dim() > 2:
                    batch_x = batch_x.view(batch_x.size(0), -1)
                
                a = batch_x
                for i, layer in enumerate(self.model.layers[:-1]):
                    if hasattr(layer, 'forward'):
                        if isinstance(layer, type(self.model.layers[0])):
                            a, s, _ = layer(a)
                            all_scores[i].append(s.cpu())
        
        # Compute average survival probabilities
        survival_probs = {}
        for layer_idx, scores_list in all_scores.items():
            if scores_list:
                avg_scores = torch.cat(scores_list, dim=0).mean(dim=0)
                survival_probs[f'layer_{layer_idx+1}'] = avg_scores.numpy()
        
        return survival_probs
    
    def plot_compression_evolution(self, history):
        """Plot how compression ratios evolve during training.
        
        Args:
            history: Training history from trainer
        """
        compression_data = history['compression_ratios']
        if not compression_data:
            print("No compression ratio data available")
            return
        
        # Extract layer-wise ratios
        layers = list(compression_data[0].keys())
        epochs = range(len(compression_data))
        
        plt.figure(figsize=(10, 6))
        for layer in layers:
            ratios = [compression_data[e][layer] for e in epochs]
            plt.plot(epochs, ratios, label=layer, marker='o', markersize=3)
        
        plt.xlabel('Epoch')
        plt.ylabel('Compression Ratio (ρ)')
        plt.title('Evolution of Learned Compression Ratios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_training_curves(self, history):
        """Plot training and validation curves.
        
        Args:
            history: Training history from trainer
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        epochs = range(len(history['train_loss']))
        ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
        if history['val_loss']:
            ax1.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
        if history['val_acc']:
            ax2.plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_layer_comparison(self, compression_ratios_dict):
        """Compare learned compression ratios across different tasks.
        
        Args:
            compression_ratios_dict: Dict mapping task names to compression ratios
        """
        tasks = list(compression_ratios_dict.keys())
        layers = list(compression_ratios_dict[tasks[0]].keys())
        
        x = np.arange(len(layers))
        width = 0.8 / len(tasks)
        
        plt.figure(figsize=(12, 6))
        for i, task in enumerate(tasks):
            ratios = [compression_ratios_dict[task][layer] for layer in layers]
            plt.bar(x + i * width, ratios, width, label=task)
        
        plt.xlabel('Layer')
        plt.ylabel('Compression Ratio (ρ)')
        plt.title('Learned Compression Ratios Across Tasks')
        plt.xticks(x + width * (len(tasks) - 1) / 2, layers)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return plt.gcf()
    
    def get_summary_stats(self, history):
        """Get summary statistics from training history.
        
        Returns:
            Dict with summary statistics
        """
        stats = {
            'final_train_acc': history['train_acc'][-1] if history['train_acc'] else None,
            'final_val_acc': history['val_acc'][-1] if history['val_acc'] else None,
            'best_val_acc': max(history['val_acc']) if history['val_acc'] else None,
            'final_compression_ratios': history['compression_ratios'][-1] if history['compression_ratios'] else None,
            'avg_compression': None
        }
        
        if stats['final_compression_ratios']:
            ratios = list(stats['final_compression_ratios'].values())
            stats['avg_compression'] = np.mean(ratios)
        
        return stats
