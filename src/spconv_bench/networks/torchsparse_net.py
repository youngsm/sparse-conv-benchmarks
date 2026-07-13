"""torchsparse++ implementation of the benchmark network + adapter.

torchsparse coordinate convention: ``coords[:, 0]`` is the batch index and
``coords[:, 1:4]`` are the x/y/z voxel indices -- identical to the neutral
:class:`~spconv_bench.data.Batch` format, so no permutation is needed. A
stride-1 ``Conv3d`` with an odd kernel auto-pads and preserves coordinates
(submanifold); stride-2 downsamples.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn

from spconv_bench.bench import Adapter
from spconv_bench.data import Batch
from spconv_bench.networks.spec import NetworkSpec


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int):
        super().__init__()
        self.conv1 = spnn.Conv3d(ch, ch, k, stride=1, bias=False)
        self.bn1 = spnn.BatchNorm(ch)
        self.conv2 = spnn.Conv3d(ch, ch, k, stride=1, bias=False)
        self.bn2 = spnn.BatchNorm(ch)
        self.relu = spnn.ReLU(inplace=True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        # torchsparse residual via SparseTensor.__add__ (matches its SparseResBlock).
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class Stage(nn.Module):
    def __init__(self, in_ch: int, ch: int, k: int, n_blocks: int, downsample: bool):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = spnn.Conv3d(in_ch, ch, k, stride=2, bias=False)
            self.down_bn = spnn.BatchNorm(ch)
            self.down_relu = spnn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([ResBlock(ch, k) for _ in range(n_blocks)])

    def forward(self, x: SparseTensor) -> SparseTensor:
        if self.downsample is not None:
            x = self.down_relu(self.down_bn(self.downsample(x)))
        for blk in self.blocks:
            x = blk(x)
        return x


class TorchSparseNet(nn.Module):
    def __init__(self, spec: NetworkSpec):
        super().__init__()
        C = spec.stage_channels
        k = spec.kernel_size
        self.stem = spnn.Conv3d(spec.in_channels, C[0], k, stride=1, bias=False)
        self.stem_bn = spnn.BatchNorm(C[0])
        self.relu = spnn.ReLU(inplace=True)
        stages = []
        for i, ch in enumerate(C):
            in_ch = C[i - 1] if i > 0 else C[0]
            stages.append(Stage(in_ch, ch, k, spec.blocks_per_stage, downsample=(i > 0)))
        self.stages = nn.ModuleList(stages)

    def forward(self, x: SparseTensor) -> SparseTensor:
        x = self.relu(self.stem_bn(self.stem(x)))
        for stage in self.stages:
            x = stage(x)
        return x


class TorchSparseAdapter(Adapter):
    name = "torchsparse"

    def library_version(self) -> str:
        import importlib.metadata as md

        try:
            return f"torchsparse {md.version('torchsparse')}"
        except md.PackageNotFoundError:
            return getattr(torchsparse, "__version__", "unknown")

    def make_model(self, spec: NetworkSpec, device: torch.device) -> nn.Module:
        return TorchSparseNet(spec).to(device)

    def make_input(self, batch: Batch, in_channels: int, device: torch.device):
        coords = torch.from_numpy(batch.coords).to(device=device, dtype=torch.int32)
        feats = torch.from_numpy(batch.feat_slice(in_channels)).to(
            device=device, dtype=torch.float32
        )
        return SparseTensor(coords=coords.contiguous(), feats=feats.contiguous())

    def forward(self, model: nn.Module, inp) -> torch.Tensor:
        return model(inp).feats


def get_adapter() -> Adapter:
    return TorchSparseAdapter()
