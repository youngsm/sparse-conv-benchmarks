"""Library-agnostic description of the benchmark network.

Every library adapter (spconv / torchsparse / warpconvnet) builds *the same*
architecture from a :class:`NetworkSpec`, so differences in the measured speed
and memory reflect the libraries, not the model.

Architecture (a standard 3D sparse ResNet encoder):

    stem:   SubMConv(in_channels -> stage_channels[0], k=3, stride=1)
    stage0: blocks x ResBlock(stage_channels[0])                 # full res
    stage1: SparseConv(stride=2) -> blocks x ResBlock(...)       # /2
    stage2: SparseConv(stride=2) -> blocks x ResBlock(...)       # /4
    ...

A ``ResBlock`` is the classic two-conv residual unit built from *submanifold*
convolutions (coordinates preserved, so the residual add is well defined):

    y = x
    y = ReLU(BN(SubMConv(y)))
    y = BN(SubMConv(y))
    out = ReLU(y + x)

Submanifold convolutions keep the active-site set fixed; the strided
``SparseConv`` at the start of each stage is a *generative* (regular) sparse
convolution that downsamples. Together they exercise the two operations that
dominate real sparse-CNN workloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class NetworkSpec:
    """Parameters defining the benchmark network."""

    name: str
    in_channels: int = 1
    stage_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    blocks_per_stage: int = 2
    kernel_size: int = 3
    #: whether stage 0 runs at full input resolution (no leading downsample)
    stem_stride: int = 1

    @property
    def num_stages(self) -> int:
        return len(self.stage_channels)

    def num_downsamples(self) -> int:
        """Number of stride-2 downsampling convolutions (one per stage after 0)."""
        return max(0, self.num_stages - 1)

    def describe(self) -> str:
        chans = "-".join(str(c) for c in self.stage_channels)
        return (
            f"{self.name}: in={self.in_channels} channels[{chans}] "
            f"blocks/stage={self.blocks_per_stage} k={self.kernel_size}"
        )


#: A small, medium and large network so we can see how each library scales with
#: model width/depth (all share the same shape, only channels/blocks grow).
DEFAULT_SPECS = {
    "small": NetworkSpec(
        name="small",
        in_channels=1,
        stage_channels=[16, 32, 64, 128],
        blocks_per_stage=1,
    ),
    "medium": NetworkSpec(
        name="medium",
        in_channels=1,
        stage_channels=[32, 64, 128, 256],
        blocks_per_stage=2,
    ),
    "large": NetworkSpec(
        name="large",
        in_channels=1,
        stage_channels=[64, 128, 256, 512],
        blocks_per_stage=3,
    ),
}
