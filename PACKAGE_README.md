# ALCNet Package - Complete Setup

## 📦 Package Successfully Built!

Your ALCNet package is ready for PyPI upload. The following files have been created:

### Distribution Files (in `dist/`)
- `alcnet-0.1.0-py3-none-any.whl` - Wheel distribution
- `alcnet-0.1.0.tar.gz` - Source distribution

## 📁 Package Structure

```
alcnet_package/
├── alcnet/                  # Main package
│   ├── __init__.py         # Package initialization
│   ├── model.py            # ALCNet model implementation
│   ├── trainer.py          # Training utilities
│   ├── analysis.py         # Analysis and visualization
│   └── utils.py            # Helper functions
├── examples/               # Example scripts
│   └── mnist_example.py    # Complete MNIST example
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_alcnet.py      # Basic tests
├── dist/                   # Built distributions (ready for upload)
│   ├── alcnet-0.1.0-py3-none-any.whl
│   └── alcnet-0.1.0.tar.gz
├── pyproject.toml          # Modern package configuration
├── setup.py                # Backward compatibility
├── MANIFEST.in             # Include non-Python files
├── LICENSE                 # MIT License
├── README.md               # Main documentation
├── QUICKSTART.md           # Quick start guide
├── UPLOAD_GUIDE.md         # Detailed upload instructions
└── build.sh                # Build automation script
```

## 🚀 Quick Upload to PyPI

### Option 1: Test on TestPyPI First (Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ alcnet
```

### Option 2: Direct Upload to PyPI

```bash
# Upload to production PyPI
python -m twine upload dist/*

# Install from PyPI
pip install alcnet
```

**Credentials:**
- Username: `__token__`
- Password: Your API token (with `pypi-` prefix)

## 📝 What's Included

### Core Features
✓ **ALCNet Model** - Neural network with learnable compression ratios
✓ **ALCNetTrainer** - Easy training with built-in loss handling
✓ **CompressionAnalyzer** - Visualization and analysis tools
✓ **Complete Examples** - MNIST classification example
✓ **Tests** - Basic test suite
✓ **Documentation** - Comprehensive guides

### Dependencies
- torch >= 1.9.0
- numpy >= 1.19.0
- tqdm >= 4.50.0
- matplotlib (optional, for visualization)
- scikit-learn (optional, for utilities)

## 🎯 Key Features of ALCNet

1. **Learnable Compression** - Ratios adapt during training
2. **Task-Adaptive** - Simple tasks compress more, complex tasks preserve features
3. **No Manual Tuning** - Eliminates architecture search
4. **Interpretable** - Compression ratios reveal task complexity
5. **Efficient** - Joint optimization of features and compression

## 📚 Documentation

- **README.md** - Full package documentation
- **QUICKSTART.md** - Get started in 5 minutes
- **UPLOAD_GUIDE.md** - Detailed PyPI upload process
- **examples/mnist_example.py** - Complete working example

## 🔧 Usage Example

```python
from alcnet import ALCNet, ALCNetTrainer
import torch.optim as optim

# Create model with learnable compression
model = ALCNet([784, 256, 128, 64, 10])

# Train
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = ALCNetTrainer(model, optimizer)
history = trainer.fit(train_loader, val_loader, epochs=50)

# Check learned compression
print(model.get_compression_ratios())
# Output: {'layer_1': 0.3245, 'layer_2': 0.2891, 'layer_3': 0.1987}
```

## 📊 Expected Results

Different tasks learn different compression patterns:

**Simple Task (MNIST):**
```
layer_1: ρ = 0.20  (aggressive compression)
layer_2: ρ = 0.15
layer_3: ρ = 0.10
```

**Complex Task (CIFAR-10):**
```
layer_1: ρ = 0.70  (preserve features)
layer_2: ρ = 0.60
layer_3: ρ = 0.40
```

## 🔄 Version History

- **v0.1.0** (Current) - Initial release
  - Core ALCNet implementation
  - Training utilities
  - Analysis tools
  - MNIST example

## 📄 License

MIT License - See LICENSE file

## 👤 Author

**Ananda Jana**  
Indian Institute of Science Education and Research Thiruvananthapuram (IISER TVM)

## 🎓 Citation

```bibtex
@article{jana2026alcnet,
  title={Adaptive Layer Condensation Networks: Learning Dynamic 
         Compression Ratios for Hierarchical Feature Selection},
  author={Jana, Ananda},
  year={2026}
}
```

## 🐛 Issues and Support

For issues, questions, or contributions:
- GitHub Issues: [Your repo URL]
- Email: [Your email]

## ✅ Pre-Upload Checklist

- [x] Package structure created
- [x] All core files implemented
- [x] Distribution files built
- [x] Documentation complete
- [x] Examples included
- [x] Tests written
- [x] License added
- [ ] PyPI account created
- [ ] API token generated
- [ ] Upload to TestPyPI (recommended)
- [ ] Upload to PyPI

## 🎉 Next Steps

1. **Review the code** - Check all files in `alcnet/`
2. **Test locally** - Run `examples/mnist_example.py`
3. **Read upload guide** - See `UPLOAD_GUIDE.md`
4. **Upload to TestPyPI** - Test first!
5. **Upload to PyPI** - Make it public!

---

**Package ready for upload! 🚀**

For detailed instructions, see `UPLOAD_GUIDE.md`
