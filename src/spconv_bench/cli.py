"""Command-line entry point: run the benchmark for one library and dump JSON.

Example::

    spconv-bench --library spconv --split val --voxel-size 1 \
        --specs small medium large --batch-sizes 1 4 8 \
        --out results/spconv.json
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch

from spconv_bench import data as data_mod
from spconv_bench.bench import benchmark
from spconv_bench.networks.spec import DEFAULT_SPECS

ADAPTER_MODULES = {
    "spconv": "spconv_bench.networks.spconv_net",
    "torchsparse": "spconv_bench.networks.torchsparse_net",
    "warpconvnet": "spconv_bench.networks.warpconvnet_net",
}


def get_adapter(name: str):
    mod = importlib.import_module(ADAPTER_MODULES[name])
    return mod.get_adapter()


def build_batch(events, batch_size: int):
    """Deterministic batch = first `batch_size` events (wrapping if needed)."""
    n = len(events)
    chosen = [events[i % n] for i in range(batch_size)]
    return data_mod.make_batch(chosen)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sparse-conv library benchmark")
    p.add_argument("--library", required=True, choices=list(ADAPTER_MODULES))
    p.add_argument("--split", default="val")
    p.add_argument("--voxel-size", type=float, default=1.0)
    p.add_argument("--specs", nargs="+", default=["small", "medium", "large"],
                   choices=list(DEFAULT_SPECS))
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-iters", type=int, default=30)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    adapter = get_adapter(args.library)

    events = data_mod.load_events(
        split=args.split, voxel_size=args.voxel_size, rebuild=args.rebuild_cache
    )
    print(f"[{args.library}] loaded {len(events)} events from split={args.split} "
          f"(voxel_size={args.voxel_size})")

    meta = {
        "library": args.library,
        "library_version": adapter.library_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": ".".join(map(str, torch.cuda.get_device_capability(device))),
        "hostname": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "voxel_size": args.voxel_size,
        "in_channels": args.in_channels,
        "n_warmup": args.n_warmup,
        "n_iters": args.n_iters,
    }

    results = []
    for spec_name in args.specs:
        spec = DEFAULT_SPECS[spec_name]
        for bs in args.batch_sizes:
            batch = build_batch(events, bs)
            res = benchmark(
                adapter, spec, batch, device,
                in_channels=args.in_channels,
                n_warmup=args.n_warmup, n_iters=args.n_iters,
            )
            results.append(res.to_dict())
            if res.ok:
                print(f"  {spec_name:6s} bs={bs:<3d} vox={res.n_voxels:<7d} "
                      f"params={res.n_params/1e6:5.2f}M | "
                      f"fwd {res.forward.median_ms:7.2f}ms  "
                      f"fwd+bwd {res.forward_backward.median_ms:7.2f}ms | "
                      f"mem(f/f+b) {res.mem_forward.peak_alloc_mb:6.0f}/"
                      f"{res.mem_forward_backward.peak_alloc_mb:6.0f} MB")
            else:
                print(f"  {spec_name:6s} bs={bs:<3d} FAILED: {res.error}")
            del batch
            gc.collect()
            torch.cuda.empty_cache()

    out_path = Path(args.out) if args.out else Path("results") / f"{args.library}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"[{args.library}] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
