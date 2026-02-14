"""
ALCNet - Adaptive Layer Condensation Networks
Learning Dynamic Compression Ratios for Hierarchical Feature Selection


Author: Ananda Jana
Institution: Indian Institute of Science Education and Research 
            Thiruvananthapuram (IISER TVM), Kerala, India

"""

from .model import ALCNet, ALCNetLayer
from .trainer import ALCNetTrainer
from .analysis import CompressionAnalyzer
from .utils import set_seed

__version__ = "0.1.1"
__author__ = "Ananda Jana"
__all__ = ["ALCNet", "ALCNetLayer", "ALCNetTrainer", "CompressionAnalyzer", "set_seed"]
