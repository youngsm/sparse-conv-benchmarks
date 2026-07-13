"""WarpConvNet implementation of the benchmark network + adapter.

WarpConvNet (NVIDIA, by the MinkowskiEngine author) represents a batch of
sparse voxels as a :class:`Voxels` geometry with *offset-based* batching: the
coordinate tensor is spatial-only ``(N, 3)`` and per-sample boundaries are given
by an ``offsets`` tensor ``[0, n0, n0+n1, ...]``. ``SparseConv3d`` with
``stride=1, generative=False`` is submanifold (coords preserved); ``stride=2``
downsamples. Norm/activation modules consume and return a ``Geometry``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import warpconvnet  # noqa: F401  (initializes NVIDIA Warp on import)
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.normalizations import BatchNorm
from warpconvnet.nn.modules.activations import ReLU

from spconv_bench.bench import Adapter
from spconv_bench.data import Batch
from spconv_bench.networks.spec import NetworkSpec


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int):
        super().__init__()
        self.conv1 = SparseConv3d(ch, ch, k, stride=1, bias=False)
        self.bn1 = BatchNorm(ch)
        self.conv2 = SparseConv3d(ch, ch, k, stride=1, bias=False)
        self.bn2 = BatchNorm(ch)
        self.relu = ReLU()

    def forward(self, x):
        # WarpConvNet residual: geometry-level add (coordinate-aware), matching
        # the library's own ResBlocks (e.g. SparseConvNeXtBlock3d: `return h + x`).
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class Stage(nn.Module):
    def __init__(self, in_ch: int, ch: int, k: int, n_blocks: int, downsample: bool):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = SparseConv3d(in_ch, ch, k, stride=2, bias=False)
            self.down_bn = BatchNorm(ch)
            self.down_relu = ReLU()
        self.blocks = nn.ModuleList([ResBlock(ch, k) for _ in range(n_blocks)])

    def forward(self, x):
        if self.downsample is not None:
            x = self.down_relu(self.down_bn(self.downsample(x)))
        for blk in self.blocks:
            x = blk(x)
        return x


class WarpConvNet(nn.Module):
    def __init__(self, spec: NetworkSpec):
        super().__init__()
        C = spec.stage_channels
        k = spec.kernel_size
        self.stem = SparseConv3d(spec.in_channels, C[0], k, stride=1, bias=False)
        self.stem_bn = BatchNorm(C[0])
        self.relu = ReLU()
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


class WarpConvNetAdapter(Adapter):
    name = "warpconvnet"

    def library_version(self) -> str:
        import importlib.metadata as md

        try:
            return f"warpconvnet {md.version('warpconvnet')}"
        except md.PackageNotFoundError:
            return "unknown"

    def make_model(self, spec: NetworkSpec, device: torch.device) -> nn.Module:
        return WarpConvNet(spec).to(device)

    def make_input(self, batch: Batch, in_channels: int, device: torch.device):
        b_idx = batch.coords[:, 0].astype(np.int64)
        xyz = np.ascontiguousarray(batch.coords[:, 1:4]).astype(np.int32)
        counts = np.bincount(b_idx, minlength=batch.batch_size)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        coords = torch.from_numpy(xyz).to(device=device, dtype=torch.int32)
        feats = torch.from_numpy(batch.feat_slice(in_channels)).to(
            device=device, dtype=torch.float32
        )
        return Voxels(
            coords.contiguous(), feats.contiguous(),
            offsets=torch.from_numpy(offsets), device=str(device),
        )

    def forward(self, model: nn.Module, inp) -> torch.Tensor:
        return model(inp).features


def get_adapter() -> Adapter:
    return WarpConvNetAdapter()
