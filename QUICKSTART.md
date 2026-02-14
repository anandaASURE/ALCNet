# ALCNet Quick Start

## Installation

```bash
pip install alcnet
```

## Basic Usage

### 1. Simple Classification

```python
import torch
from alcnet import ALCNet, ALCNetTrainer
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Your data
X_train = torch.randn(1000, 784)  # 1000 samples, 784 features
y_train = torch.randint(0, 10, (1000,))  # 10 classes

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create model
model = ALCNet([784, 256, 128, 64, 10])

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create trainer
trainer = ALCNetTrainer(model, optimizer)

# Train
history = trainer.fit(train_loader, epochs=20)

# Check learned compression
print(model.get_compression_ratios())
```

### 2. With Validation

```python
# Add validation data
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32)

# Train with validation
history = trainer.fit(train_loader, val_loader, epochs=50)
```

### 3. Analyze Results

```python
from alcnet import CompressionAnalyzer

analyzer = CompressionAnalyzer(model)

# Get statistics
stats = analyzer.get_summary_stats(history)
print(f"Best accuracy: {stats['best_val_acc']:.2f}%")
print(f"Average compression: {stats['avg_compression']:.3f}")

# Plot training curves
analyzer.plot_training_curves(history)

# Plot compression evolution
analyzer.plot_compression_evolution(history)
```

### 4. Save and Load

```python
# Save model
trainer.save_model('my_model.pth')

# Load model later
model_new = ALCNet([784, 256, 128, 64, 10])
optimizer_new = optim.Adam(model_new.parameters())
trainer_new = ALCNetTrainer(model_new, optimizer_new)
trainer_new.load_model('my_model.pth')
```

### 5. Make Predictions

```python
# Predict on new data
X_test = torch.randn(100, 784)
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(dim=1)
```

## Customization

### Custom Compression Ratios

```python
# Specify initial compression ratios
initial_ratios = [0.4, 0.3, 0.2]
model = ALCNet([784, 256, 128, 64, 10], initial_ratios=initial_ratios)
```

### Adjust Loss Weights

```python
trainer = ALCNetTrainer(
    model,
    optimizer,
    lambda_sparse=0.005,  # Higher = more sparsity
    lambda_ratio=0.05     # Higher = stricter ratio matching
)
```

### Use GPU

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = ALCNetTrainer(model, optimizer, device=device)
```

## Complete MNIST Example

```python
from torchvision import datasets, transforms

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# Create and train model
model = ALCNet([784, 256, 128, 64, 10])
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = ALCNetTrainer(model, optimizer, device='cuda')

history = trainer.fit(train_loader, test_loader, epochs=20)

print("\nFinal Compression Ratios:")
for layer, ratio in model.get_compression_ratios().items():
    print(f"{layer}: {ratio:.4f}")
```

## Tips

1. **Start with default settings** - they work well for most tasks
2. **Monitor compression ratios** - they reveal task complexity
3. **Try different layer sizes** - architecture affects learning
4. **Use validation data** - essential for hyperparameter tuning
5. **Experiment with loss weights** - balance accuracy vs compression

## Common Patterns

### Aggressive Compression (Simple Tasks)
```python
trainer = ALCNetTrainer(model, optimizer, lambda_sparse=0.01, lambda_ratio=0.05)
```

### Gentle Compression (Complex Tasks)
```python
trainer = ALCNetTrainer(model, optimizer, lambda_sparse=0.0001, lambda_ratio=0.001)
```

### High-Dimensional Data
```python
# Use more aggressive initial compression
initial_ratios = [0.2, 0.15, 0.1]
model = ALCNet([10000, 512, 256, 64, 10], initial_ratios=initial_ratios)
```

## Next Steps

- Read the full [README.md](README.md) for more details
- See [examples/mnist_example.py](examples/mnist_example.py) for complete code

