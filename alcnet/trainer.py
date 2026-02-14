"""
Trainer for ALCNet models with built-in loss handling.
Author - Ananda Jana , IISER TVM , Kerala , India

"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class ALCNetTrainer:
    """Trainer for ALCNet models.
    
    Args:
        model: ALCNet model instance
        optimizer: PyTorch optimizer
        lambda_sparse (float): Weight for sparsity loss (default: 0.001)
        lambda_ratio (float): Weight for ratio loss (default: 0.01)
        device (str): Device to use ('cuda' or 'cpu')
    """
    
    def __init__(self, model, optimizer, lambda_sparse=0.001, lambda_ratio=0.01, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lambda_sparse = lambda_sparse
        self.lambda_ratio = lambda_ratio
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'compression_ratios': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc='Training', leave=False):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Flatten if needed
            if batch_x.dim() > 2:
                batch_x = batch_x.view(batch_x.size(0), -1)
            
            # Forward pass
            output, sparsity_loss, ratio_loss = self.model(batch_x)
            
            # Compute total loss
            ce_loss = F.cross_entropy(output, batch_y)
            loss = ce_loss + self.lambda_sparse * sparsity_loss + self.lambda_ratio * ratio_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Flatten if needed
                if batch_x.dim() > 2:
                    batch_x = batch_x.view(batch_x.size(0), -1)
                
                # Forward pass
                output, sparsity_loss, ratio_loss = self.model(batch_x)
                
                # Compute loss
                ce_loss = F.cross_entropy(output, batch_y)
                loss = ce_loss + self.lambda_sparse * sparsity_loss + self.lambda_ratio * ratio_loss
                
                # Track metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def fit(self, train_loader, val_loader=None, epochs=50, verbose=True):
        """Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs (int): Number of epochs to train
            verbose (bool): Whether to print progress
        """
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            # Track compression ratios
            ratios = self.model.get_compression_ratios()
            self.history['compression_ratios'].append(ratios)
            
            # Print progress
            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%"
                if val_loader is not None:
                    msg += f" - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                if (epoch + 1) % 10 == 0:
                    msg += f"\n  Compression ratios: {ratios}"
                print(msg)
        
        return self.history
    
    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
