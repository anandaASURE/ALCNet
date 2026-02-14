# ALCNet PyPI Upload Guide

## Prerequisites

1. **PyPI Account**: Create accounts at:
   - TestPyPI: https://test.pypi.org/account/register/
   - PyPI: https://pypi.org/account/register/

2. **API Tokens**: Generate API tokens:
   - TestPyPI: https://test.pypi.org/manage/account/#api-tokens
   - PyPI: https://pypi.org/manage/account/token/

3. **Install Required Tools**:
```bash
pip install build twine --upgrade
```

## Step-by-Step Upload Process

### 1. Build the Package

Navigate to the package directory:
```bash
cd alcnet_package
```

Build distribution files:
```bash
python -m build
```

This creates two files in `dist/`:
- `alcnet-0.1.0.tar.gz` (source distribution)
- `alcnet-0.1.0-py3-none-any.whl` (built distribution)

### 2. Test Upload to TestPyPI (Recommended)

Upload to TestPyPI first to test:
```bash
python -m twine upload --repository testpypi dist/*
```

When prompted:
- Username: `__token__`
- Password: Your TestPyPI API token (including `pypi-` prefix)

Test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ alcnet
```

### 3. Upload to PyPI

Once verified on TestPyPI, upload to real PyPI:
```bash
python -m twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: Your PyPI API token (including `pypi-` prefix)

### 4. Install and Test

After successful upload:
```bash
pip install alcnet
```

Test the installation:
```python
from alcnet import ALCNet
model = ALCNet([784, 256, 128, 64, 10])
print(model.get_compression_ratios())
```

## Updating the Package

For subsequent releases:

1. Update version in `pyproject.toml`
2. Rebuild: `python -m build`
3. Upload: `python -m twine upload dist/*`

**Important**: PyPI does not allow re-uploading the same version. Always increment the version number.

## Version Numbering

Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible API changes
- MINOR: New functionality, backward compatible
- PATCH: Bug fixes, backward compatible

Examples:
- `0.1.0` → `0.1.1` (bug fix)
- `0.1.0` → `0.2.0` (new features)
- `0.1.0` → `1.0.0` (major release)

## Troubleshooting

### Issue: "File already exists"
**Solution**: Increment version number in `pyproject.toml` and rebuild.

### Issue: "Invalid credentials"
**Solution**: 
- Ensure username is `__token__` (not your username)
- Include full API token with `pypi-` prefix
- Check token hasn't expired

### Issue: "Package name already taken"
**Solution**: Choose a different package name in `pyproject.toml`.

### Issue: Build fails
**Solution**: 
- Ensure all required files are present
- Check `pyproject.toml` syntax
- Verify Python version compatibility

## Files Generated

After successful upload, your package will have:
- PyPI page: https://pypi.org/project/alcnet/
- Install command: `pip install alcnet`
- Documentation: Rendered from README.md

## Security Notes

- Never commit API tokens to version control
- Store tokens securely (use environment variables or password manager)
- Regenerate tokens if compromised
- Use separate tokens for TestPyPI and PyPI

## Additional Resources

- Python Packaging Guide: https://packaging.python.org/
- Twine Documentation: https://twine.readthedocs.io/
- PyPI Help: https://pypi.org/help/
