"""
Setup script for ALCNet package.
Note: This is for backward compatibility. The package is primarily configured via pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    name="alcnet",
    packages=find_packages(),
    python_requires=">=3.7",
)
