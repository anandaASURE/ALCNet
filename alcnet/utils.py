"""
Utility functions for ALCNet.
Author - Ananda Jana , IISER TVM , Kerala , India

"""

import torch
import numpy as np
import random


def set_seed(seed=42):
    """Set random seed for reproducibility.
       Author - Ananda Jana , IISER TVM , Kerala , India

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Get available device (CUDA or CPU).
    
    Returns:
        str: Device name
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and validation sets.
    
    Args:
        X: Input features
        y: Target labels
        test_size (float): Fraction for validation
        random_state (int): Random seed
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
