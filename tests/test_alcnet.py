"""
Basic tests for ALCNet package.
"""

import torch
import pytest
from alcnet import ALCNet, ALCNetLayer, ALCNetTrainer, CompressionAnalyzer


def test_alcnet_layer_creation():
    """Test ALCNetLayer creation."""
    layer = ALCNetLayer(784, 256, initial_ratio=0.5)
    assert isinstance(layer, torch.nn.Module)
    assert layer.get_compression_ratio() > 0


def test_alcnet_forward():
    """Test ALCNet forward pass."""
    model = ALCNet([784, 256, 128, 64, 10])
    x = torch.randn(32, 784)
    output, sparsity, ratio_loss = model(x)
    
    assert output.shape == (32, 10)
    assert isinstance(sparsity.item(), float)
    assert isinstance(ratio_loss.item(), float)


def test_compression_ratios():
    """Test getting compression ratios."""
    model = ALCNet([784, 256, 128, 64, 10])
    ratios = model.get_compression_ratios()
    
    assert isinstance(ratios, dict)
    assert len(ratios) == 3  # 3 hidden layers


def test_predict():
    """Test prediction method."""
    model = ALCNet([784, 256, 128, 64, 10])
    x = torch.randn(32, 784)
    predictions = model.predict(x)
    
    assert predictions.shape == (32, 10)
    assert torch.allclose(predictions.sum(dim=1), torch.ones(32), atol=1e-5)


def test_trainer_creation():
    """Test trainer creation."""
    model = ALCNet([784, 256, 128, 64, 10])
    optimizer = torch.optim.Adam(model.parameters())
    trainer = ALCNetTrainer(model, optimizer)
    
    assert trainer.model is not None
    assert trainer.optimizer is not None


def test_analyzer_creation():
    """Test analyzer creation."""
    model = ALCNet([784, 256, 128, 64, 10])
    analyzer = CompressionAnalyzer(model)
    
    assert analyzer.model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
