"""MinkowskiEngine implementation of the benchmark network + adapter.

MinkowskiEngine coordinates are ``(N, 1+D)`` with the *first* column the batch
index -- identical to the neutral :class:`~spconv_bench.data.Batch` layout
(``[batch, x, y, z]``), so no permutation is needed. A stride-1
``MinkowskiConvolution`` keeps the input coordinates (submanifold-equivalent);
stride-2 downsamples. Residuals use ``SparseTensor.__add__`` (coordinate-aware).

NOTE: MinkowskiEngine's last release (0.5.4) predates CUDA 12 / torch 2.x, so it
is benchmarked in its own environment on torch 1.10 + CUDA 11.3. The sparse-conv
kernels are MinkowskiEngine's own CUDA, so the measurement still reflects the
library; the torch/CUDA difference is documented in the README.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import MinkowskiEngine as ME

from spconv_bench.bench import Adapter
from spconv_bench.data import Batch
from spconv_bench.networks.spec import NetworkSpec

D = 3  # spatial dimensions


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(ch, ch, kernel_size=k, stride=1, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(ch)
        self.conv2 = ME.MinkowskiConvolution(ch, ch, kernel_size=k, stride=1, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(ch)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class Stage(nn.Module):
    def __init__(self, in_ch: int, ch: int, k: int, n_blocks: int, downsample: bool):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = ME.MinkowskiConvolution(
                in_ch, ch, kernel_size=k, stride=2, dimension=D
            )
            self.down_bn = ME.MinkowskiBatchNorm(ch)
            self.down_relu = ME.MinkowskiReLU(inplace=True)
        self.blocks = nn.ModuleList([ResBlock(ch, k) for _ in range(n_blocks)])

    def forward(self, x):
        if self.downsample is not None:
            x = self.down_relu(self.down_bn(self.downsample(x)))
        for blk in self.blocks:
            x = blk(x)
        return x


class MinkowskiNet(nn.Module):
    def __init__(self, spec: NetworkSpec):
        super().__init__()
        C = spec.stage_channels
        k = spec.kernel_size
        self.stem = ME.MinkowskiConvolution(
            spec.in_channels, C[0], kernel_size=k, stride=1, dimension=D
        )
        self.stem_bn = ME.MinkowskiBatchNorm(C[0])
        self.relu = ME.MinkowskiReLU(inplace=True)
        stages = []
        for i, ch in enumerate(C):
            in_ch = C[i - 1] if i > 0 else C[0]
            stages.append(Stage(in_ch, ch, k, spec.blocks_per_stage, downsample=(i > 0)))
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        x = self.relu(self.stem_bn(self.stem(x)))
        for stage in self.stages:
            x = stage(x)
        return x


class MinkowskiAdapter(Adapter):
    name = "minkowski"

    def library_version(self) -> str:
        return f"MinkowskiEngine {getattr(ME, '__version__', 'unknown')}"

    def make_model(self, spec: NetworkSpec, device: torch.device) -> nn.Module:
        return MinkowskiNet(spec).to(device)

    def make_input(self, batch: Batch, in_channels: int, device: torch.device):
        coords = torch.from_numpy(batch.coords).to(dtype=torch.int32).contiguous()
        feats = torch.from_numpy(batch.feat_slice(in_channels)).to(
            dtype=torch.float32
        ).contiguous()
        return ME.SparseTensor(
            features=feats.to(device), coordinates=coords.to(device)
        )

    def forward(self, model: nn.Module, inp) -> torch.Tensor:
        return model(inp).F


def get_adapter() -> Adapter:
    return MinkowskiAdapter()
