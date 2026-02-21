<img width="180" height="177" alt="Screenshot 2026-02-21 162500" src="https://github.com/user-attachments/assets/f883486d-7a83-44c0-88ba-8950cc2460c3" />



# ALCN - Adaptive Layer Condensation Networks
### Author - Ananda Jana  
[![PyPI version](https://badge.fury.io/py/alcnet.svg)](https://badge.fury.io/py/alcnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Learning Dynamic Compression Ratios for Hierarchical Feature Selection**

ALCNet/ALCN is a neural network architecture where compression ratios at each layer are learned as trainable parameters rather than fixed before training. This enables task-adaptive architecture optimization without expensive neural architecture search.

<img width="555" height="375" alt="image" src="https://github.com/user-attachments/assets/65994f91-87e9-4b8a-9f3c-3fcf14c7bdd7" />

## Key Features

- **Learnable Compression Ratios**: Target sparsity at each layer adapts during training
- **Task-Adaptive**: Simple tasks learn aggressive compression, complex tasks preserve features
- **Differentiable**: End-to-end training with backpropagation
- **No NAS Required**: Eliminates expensive architecture search
- **Interpretable**: Learned compression ratios reveal task complexity

- 
## Installation

```bash
pip install alcnet
```

## Quick Start

```python
import torch
from alcnet import ALCNet, ALCNetTrainer
import torch.optim as optim

# Create model with learnable compression ratios
model = ALCNet([784, 256, 128, 64, 10])

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = ALCNetTrainer(model, optimizer, lambda_sparse=0.001, lambda_ratio=0.01)

# Train on your data
history = trainer.fit(train_loader, val_loader, epochs=50)

# Check learned compression ratios
print(model.get_compression_ratios())
```

## How It Works

Unlike traditional networks with fixed layer sizes, ALCNet learns optimal compression at each layer:

1. **Standard Approach**: You decide `[784, 256, 128, 64, 10]` before training
2. **ALCNet Approach**: Network learns how much to compress at each layer based on task

### Architecture

```
Input Layer (784) 
    ↓  [Learnable ρ⁽¹⁾]
Hidden Layer (256)
    ↓  [Learnable ρ⁽²⁾]
Hidden Layer (128)
    ↓  [Learnable ρ⁽³⁾]
Hidden Layer (64)
    ↓
Output (10)
```

Each ρ⁽ⁱ⁾ ∈ (0,1) controls compression intensity at layer i.

## Example Results

Simple tasks may learn aggressive early compression:
```
layer_1: ρ = 0.20 (keep 20% of neurons)
layer_2: ρ = 0.15
layer_3: ρ = 0.10
```

Complex tasks preserve more features:
```
layer_1: ρ = 0.70 (keep 70% of neurons)
layer_2: ρ = 0.60
layer_3: ρ = 0.40
```

## Advanced Usage

### Analysis and Visualization

```python
from alcnet import CompressionAnalyzer

analyzer = CompressionAnalyzer(model)

# Plot compression evolution during training
analyzer.plot_compression_evolution(history)

# Compute feature survival probabilities
survival = analyzer.compute_feature_survival(dataloader)

# Get summary statistics
stats = analyzer.get_summary_stats(history)
```

### Custom Configuration

```python
# Specify initial compression ratios
initial_ratios = [0.5, 0.4, 0.3]
model = ALCNet([784, 256, 128, 64, 10], initial_ratios=initial_ratios)

# Adjust loss weights
trainer = ALCNetTrainer(
    model, 
    optimizer,
    lambda_sparse=0.005,  # Sparsity weight
    lambda_ratio=0.05     # Ratio matching weight
)
```

<img width="740" height="1086" alt="Screenshot 2026-02-15 002455" src="https://github.com/user-attachments/assets/a391af6a-692e-45df-a59d-e78ca84f53fe" />

## Examples

See the `examples/` directory:
- `mnist_example.py`: Complete MNIST classification example
- More examples coming soon

## Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.9.0
- NumPy
- matplotlib (optional, for visualization)
- scikit-learn (optional, for data utilities)

## Citation

If you use ALCNet in your research, please cite:

```bibtex
@article{jana2026adaptive,
  author       = {Jana, Ananda},
  title        = {Adaptive Layer Condensation Networks: Learning Dynamic Compression Ratios for Hierarchical Feature Selection},
  journal      = {TechRxiv},
  year         = {2026},
  month        = feb,
  doi          = {10.36227/techrxiv.177162166.68955501/v1},
  note         = {Preprint}
}
```

## Paper

The full paper describing the theory, implementation, and experimental validation of ALCNet is available at: [(https://doi.org/10.36227/techrxiv.177162166.68955501/v1)]

A. Jana, “Adaptive Layer Condensation Networks: Learning Dynamic Compression Ratios for Hierarchical Feature Selection,” TechRxiv, Feb. 20, 2026. doi:10.36227/techrxiv.177162166.68955501/v1

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Ananda Jana**  
Indian Institute of Science Education and Research Thiruvananthapuram (IISER TVM), Kerala, India

## Acknowledgments

This work explores learnable compression mechanisms in neural network architectures through independent research into adaptive feature selection and hierarchical compression strategies.
