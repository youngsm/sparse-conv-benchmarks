"""spconv implementation of the benchmark network + adapter."""

from __future__ import annotations

import torch
import torch.nn as nn

import spconv.pytorch as spconv

from spconv_bench.bench import Adapter
from spconv_bench.data import Batch
from spconv_bench.networks.spec import NetworkSpec


class ResBlock(nn.Module):
    """Two submanifold convs with a residual add (coords preserved)."""

    def __init__(self, ch: int, k: int, indice_key: str):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(ch, ch, k, bias=False, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = spconv.SubMConv3d(ch, ch, k, bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + identity))
        return out


class Stage(nn.Module):
    """Optional stride-2 downsample (regular sparse conv) then residual blocks."""

    def __init__(self, in_ch: int, ch: int, k: int, n_blocks: int,
                 stage_idx: int, downsample: bool):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = spconv.SparseConv3d(
                in_ch, ch, k, stride=2, padding=1, bias=False,
                indice_key=f"sp{stage_idx}",
            )
            self.down_bn = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList(
            [ResBlock(ch, k, indice_key=f"subm{stage_idx}") for _ in range(n_blocks)]
        )

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
            x = x.replace_feature(self.relu(self.down_bn(x.features)))
        for blk in self.blocks:
            x = blk(x)
        return x


class SpconvNet(nn.Module):
    def __init__(self, spec: NetworkSpec):
        super().__init__()
        C = spec.stage_channels
        k = spec.kernel_size
        self.stem = spconv.SubMConv3d(
            spec.in_channels, C[0], k, bias=False, indice_key="subm0"
        )
        self.stem_bn = nn.BatchNorm1d(C[0])
        self.relu = nn.ReLU(inplace=True)
        stages = []
        for i, ch in enumerate(C):
            in_ch = C[i - 1] if i > 0 else C[0]
            stages.append(
                Stage(in_ch, ch, k, spec.blocks_per_stage, stage_idx=i, downsample=(i > 0))
            )
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        x = self.stem(x)
        x = x.replace_feature(self.relu(self.stem_bn(x.features)))
        for stage in self.stages:
            x = stage(x)
        return x


class SpconvAdapter(Adapter):
    name = "spconv"

    def library_version(self) -> str:
        import importlib.metadata as md

        for dist in ("spconv-cu121", "spconv-cu124", "spconv-cu126",
                     "spconv-cu120", "spconv-cu118", "spconv"):
            try:
                return f"{dist} {md.version(dist)}"
            except md.PackageNotFoundError:
                continue
        return getattr(spconv, "__version__", "unknown")

    def make_model(self, spec: NetworkSpec, device: torch.device) -> nn.Module:
        return SpconvNet(spec).to(device)

    def make_input(self, batch: Batch, in_channels: int, device: torch.device):
        coords = torch.from_numpy(batch.coords).to(device=device, dtype=torch.int32)
        feats = torch.from_numpy(batch.feat_slice(in_channels)).to(
            device=device, dtype=torch.float32
        )
        return spconv.SparseConvTensor(
            feats.contiguous(), coords.contiguous(),
            list(batch.spatial_shape), batch.batch_size,
        )

    def forward(self, model: nn.Module, inp) -> torch.Tensor:
        return model(inp).features


def get_adapter() -> Adapter:
    return SpconvAdapter()
