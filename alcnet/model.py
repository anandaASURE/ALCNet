"""
Core ALCNet model implementation with learnable compression ratios.

Author - Ananda Jana , IISER TVM , Kerala , India

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ALCNetLayer(nn.Module):
    """
    Author - Ananda Jana , IISER TVM , Kerala , India

    Single ALCNet layer with learnable compression ratio.
    
    Args:
        input_size (int): Size of input features
        output_size (int): Size of output features
        initial_ratio (float): Initial compression ratio (default: 0.5)
        initial_alpha (float): Initial sharpness parameter (default: 1.0)
    """
    
    def __init__(self, input_size, output_size, initial_ratio=0.5, initial_alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        
        # Learnable compression ratio in logit space for stability
        self.rho_logit = nn.Parameter(
            torch.tensor(math.log(initial_ratio / (1 - initial_ratio + 1e-8)))
        )
    
    def get_compression_ratio(self):
        """Get current compression ratio."""
        return torch.sigmoid(self.rho_logit).item()
    
    def forward(self, x):
        """Forward pass with soft filtering.
            Author - Ananda Jana , IISER TVM , Kerala , India

        Returns:
            a: Filtered activations
            s: Selection scores
            ratio_loss: Loss encouraging target compression
        """
        # Linear transformation + ReLU
        h = F.relu(self.linear(x))
        
        # Compute median reference
        median_h = torch.median(h, dim=-1, keepdim=True)[0]
        
        # Compute selection scores (soft filter)
        s = torch.sigmoid(self.alpha * (h - median_h))
        
        # Apply filter
        a = h * s
        
        # Compute ratio loss
        target_ratio = torch.sigmoid(self.rho_logit)
        actual_ratio = s.mean()
        ratio_loss = (actual_ratio - target_ratio) ** 2
        
        return a, s, ratio_loss


class ALCNet(nn.Module):
    """Adaptive Layer Condensation Network with learnable compression ratios.
       Author - Ananda Jana , IISER TVM , Kerala , India

    Args:
        layer_sizes (list): List of layer sizes [input, hidden1, ..., output]
        initial_ratios (list, optional): Initial compression ratios for each layer
        initial_alpha (float): Initial sharpness parameter (default: 1.0)
    
    Example:
        >>> model = ALCNet([784, 256, 128, 64, 10])
        >>> output, sparsity_loss, ratio_loss = model(x)
    """
    
    def __init__(self, layer_sizes, initial_ratios=None, initial_alpha=1.0):
        super().__init__()
        
        # Compute initial ratios from geometric decay if not provided
        if initial_ratios is None:
            initial_ratios = []
            for i in range(len(layer_sizes) - 2):
                ratio = layer_sizes[i + 1] / layer_sizes[i]
                initial_ratios.append(ratio)
        
        # Create layers with learnable compression
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            if i < len(layer_sizes) - 2:  # Not the output layer
                layer = ALCNetLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    initial_ratios[i] if i < len(initial_ratios) else 0.5,
                    initial_alpha
                )
            else:  # Output layer (no filtering)
                layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
        
        self.num_filterable_layers = len(layer_sizes) - 2
    
    def get_compression_ratios(self):
        """Get current compression ratios for all layers."""
        ratios = {}
        for i, layer in enumerate(self.layers[:-1]):
            if isinstance(layer, ALCNetLayer):
                ratios[f'layer_{i+1}'] = layer.get_compression_ratio()
        return ratios
    
    def forward(self, x):
        """Forward pass through network.
           Author - Ananda Jana , IISER TVM , Kerala , India

        Returns:
            output: Network output (logits)
            sparsity_loss: Overall sparsity loss
            ratio_loss: Compression ratio matching loss
        """
        a = x
        sparsity_loss = 0
        ratio_losses = []
        
        # Forward through all layers except last
        for layer in self.layers[:-1]:
            if isinstance(layer, ALCNetLayer):
                a, s, ratio_loss = layer(a)
                sparsity_loss += s.mean()
                ratio_losses.append(ratio_loss)
            else:
                a = F.relu(layer(a))
        
        # Last layer (no filtering)
        output = self.layers[-1](a)
        
        # Average losses
        if self.num_filterable_layers > 0:
            sparsity_loss = sparsity_loss / self.num_filterable_layers
            ratio_loss = sum(ratio_losses) / len(ratio_losses) if ratio_losses else torch.tensor(0.0)
        else:
            ratio_loss = torch.tensor(0.0)
        
        return output, sparsity_loss, ratio_loss
    
    def predict(self, x):
        """Prediction without auxiliary losses."""
        output, _, _ = self.forward(x)
        return F.softmax(output, dim=-1)
