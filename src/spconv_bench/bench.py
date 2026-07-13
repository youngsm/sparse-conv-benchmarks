"""Library-agnostic benchmark harness (speed + GPU memory).

Each sparse-conv library provides an :class:`Adapter` that knows how to build
the shared network, upload a batch to the GPU, and run a forward pass returning
a dense feature tensor. The harness then measures, with CUDA events:

* forward-only latency (training-mode forward, ``no_grad``),
* forward+backward latency (loss = output.sum(); ``loss.backward()``),

and, with :func:`torch.cuda.max_memory_*`, the peak allocated/reserved memory
for each.

Fairness notes
--------------
* The library sparse tensor is built once (uploaded to the GPU) and reused
  across iterations, so we measure *steady-state* forward/backward throughput:
  the convolution rulebook / kernel map is constructed during warmup and then
  reused, exactly as it is amortized across steps in real training and
  inference. This is also required to treat WarpConvNet fairly -- it autotunes
  its GEMM algorithm per problem shape and caches the result, which only
  amortizes when the input is not rebuilt every call.
* Every library builds the identical architecture (see ``networks/spec.py``)
  and consumes byte-identical inputs (see ``data.py``).
* Peak memory is measured on a clean allocator state after warmup, so it
  reflects the steady-state forward / forward+backward footprint.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from spconv_bench.data import Batch
from spconv_bench.networks.spec import NetworkSpec


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class Timing:
    mean_ms: float
    std_ms: float
    median_ms: float
    min_ms: float
    p90_ms: float
    n_iters: int

    @classmethod
    def from_samples(cls, times_ms: List[float]) -> "Timing":
        s = sorted(times_ms)
        n = len(s)
        p90 = s[min(n - 1, int(round(0.9 * (n - 1))))]
        return cls(
            mean_ms=float(statistics.fmean(s)),
            std_ms=float(statistics.pstdev(s)) if n > 1 else 0.0,
            median_ms=float(statistics.median(s)),
            min_ms=float(s[0]),
            p90_ms=float(p90),
            n_iters=n,
        )


@dataclass
class Memory:
    peak_alloc_mb: float
    peak_reserved_mb: float


@dataclass
class BenchResult:
    library: str
    library_version: str
    torch_version: str
    cuda_version: str
    device_name: str
    spec_name: str
    in_channels: int
    batch_size: int
    n_voxels: int
    n_params: int
    forward: Optional[Timing] = None
    forward_backward: Optional[Timing] = None
    mem_forward: Optional[Memory] = None
    mem_forward_backward: Optional[Memory] = None
    throughput_kvox_s: float = 0.0  # forward+backward
    ok: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Adapter interface
# --------------------------------------------------------------------------- #
class Adapter:
    """Base class each library implements.

    ``make_input`` uploads a batch to the GPU *once* and returns an opaque
    holder; ``forward`` must (cheaply) wrap those GPU tensors into the library's
    sparse-tensor type and run the model, so that rulebook construction is
    re-timed each iteration.
    """

    name: str = "base"

    def library_version(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def make_model(self, spec: NetworkSpec, device: torch.device) -> torch.nn.Module:
        raise NotImplementedError

    def make_input(self, batch: Batch, in_channels: int, device: torch.device) -> Any:
        raise NotImplementedError

    def forward(self, model: torch.nn.Module, inp: Any) -> torch.Tensor:
        """Run the model; return a dense ``(M, C)`` feature tensor for the loss."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Timing / memory primitives
# --------------------------------------------------------------------------- #
def _time_cuda(fn: Callable[[], Any], n_warmup: int, n_iters: int,
               device: torch.device) -> List[float]:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize(device)
    times: List[float] = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))
    return times


def _peak_mem(fn: Callable[[], Any], device: torch.device) -> Memory:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return Memory(
        peak_alloc_mb=torch.cuda.max_memory_allocated(device) / 1e6,
        peak_reserved_mb=torch.cuda.max_memory_reserved(device) / 1e6,
    )


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def benchmark(
    adapter: Adapter,
    spec: NetworkSpec,
    batch: Batch,
    device: torch.device,
    in_channels: Optional[int] = None,
    n_warmup: int = 10,
    n_iters: int = 30,
) -> BenchResult:
    """Benchmark one (library, spec, batch) triple."""
    in_channels = in_channels or spec.in_channels
    cuda_ver = torch.version.cuda or "cpu"
    res = BenchResult(
        library=adapter.name,
        library_version=adapter.library_version(),
        torch_version=torch.__version__,
        cuda_version=cuda_ver,
        device_name=torch.cuda.get_device_name(device),
        spec_name=spec.name,
        in_channels=in_channels,
        batch_size=batch.batch_size,
        n_voxels=batch.n_voxels,
        n_params=0,
    )

    try:
        model = adapter.make_model(spec, device)
        model.train()
        res.n_params = sum(p.numel() for p in model.parameters())
        inp = adapter.make_input(batch, in_channels, device)

        def fwd() -> torch.Tensor:
            return adapter.forward(model, inp)

        def fwd_bwd() -> torch.Tensor:
            model.zero_grad(set_to_none=True)
            out = adapter.forward(model, inp)
            loss = out.float().sum()
            loss.backward()
            return loss

        # correctness sanity check
        with torch.no_grad():
            out = fwd()
        if not torch.isfinite(out.float().sum()):
            raise RuntimeError("non-finite output on sanity forward")

        # forward-only latency
        with torch.no_grad():
            res.forward = Timing.from_samples(
                _time_cuda(fwd, n_warmup, n_iters, device)
            )
        # forward+backward latency
        res.forward_backward = Timing.from_samples(
            _time_cuda(fwd_bwd, n_warmup, n_iters, device)
        )
        # peak memory (measured separately, on clean allocator state)
        with torch.no_grad():
            res.mem_forward = _peak_mem(fwd, device)
        res.mem_forward_backward = _peak_mem(fwd_bwd, device)

        if res.forward_backward.mean_ms > 0:
            res.throughput_kvox_s = (
                batch.n_voxels / (res.forward_backward.mean_ms / 1e3) / 1e3
            )
    except Exception as e:  # noqa: BLE001 - record failure, keep other configs
        res.ok = False
        res.error = f"{type(e).__name__}: {e}"

    return res
