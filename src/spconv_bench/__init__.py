"""Benchmark suite for 3D sparse-convolution libraries on PILArNet-M.

Compares speed (forward / forward+backward latency) and GPU memory
(peak allocated / reserved) across spconv, torchsparse++ and WarpConvNet
using a shared, library-agnostic ResNet-style sparse CNN and identical inputs.
"""

__version__ = "0.1.0"
