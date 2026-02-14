"""
Example usage of ALCNet on MNIST dataset.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from alcnet import ALCNet, ALCNetTrainer, CompressionAnalyzer, set_seed


def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create ALCNet model
    layer_sizes = [784, 256, 128, 64, 10]
    model = ALCNet(layer_sizes)
    print(f"\nModel architecture: {layer_sizes}")
    print(f"Initial compression ratios: {model.get_compression_ratios()}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    trainer = ALCNetTrainer(
        model=model,
        optimizer=optimizer,
        lambda_sparse=0.001,
        lambda_ratio=0.01,
        device=device
    )
    
    # Train the model
    print("\nTraining ALCNet...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=20,
        verbose=True
    )
    
    # Print final compression ratios
    print("\n" + "="*60)
    print("Final learned compression ratios:")
    final_ratios = model.get_compression_ratios()
    for layer, ratio in final_ratios.items():
        print(f"  {layer}: {ratio:.4f}")
    
    # Analyze results
    analyzer = CompressionAnalyzer(model, device=device)
    stats = analyzer.get_summary_stats(history)
    
    print("\nTraining Summary:")
    print(f"  Final Train Accuracy: {stats['final_train_acc']:.2f}%")
    print(f"  Final Test Accuracy: {stats['final_val_acc']:.2f}%")
    print(f"  Best Test Accuracy: {stats['best_val_acc']:.2f}%")
    print(f"  Average Compression: {stats['avg_compression']:.4f}")
    print("="*60)
    
    # Save model
    trainer.save_model('alcnet_mnist.pth')
    print("\nModel saved to 'alcnet_mnist.pth'")
    
    # Generate plots
    try:
        import matplotlib.pyplot as plt
        
        # Plot training curves
        fig1 = analyzer.plot_training_curves(history)
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("Training curves saved to 'training_curves.png'")
        
        # Plot compression evolution
        fig2 = analyzer.plot_compression_evolution(history)
        plt.savefig('compression_evolution.png', dpi=150, bbox_inches='tight')
        print("Compression evolution saved to 'compression_evolution.png'")
        
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")


if __name__ == "__main__":
    main()
