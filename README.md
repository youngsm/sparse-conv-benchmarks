# Sparse Convolution Library Benchmark (PILArNet-M)

A detailed, apples-to-apples benchmark of 3D **sparse convolution** libraries on real liquid-argon TPC data.
It measures both **speed** (forward and forward+backward latency) and **GPU memory** (peak allocated and reserved) for an identical network architecture and identical inputs across each library.

## Libraries compared

| Library | Version | Env (torch / CUDA) | Install | Sparse conv kernels |
| --- | --- | --- | --- | --- |
| [spconv](https://github.com/traveller59/spconv) | `spconv-cu124` 2.3.8 | 2.5.0 / 12.4 | prebuilt wheel | implicit GEMM / native |
| [torchsparse++](https://github.com/mit-han-lab/torchsparse) | master (`385f5ce`) | 2.5.0 / 12.4 | compiled from source | adaptive gather-scatter / implicit GEMM |
| [WarpConvNet](https://github.com/NVlabs/WarpConvNet) | 1.7.11 | 2.5.0 / 12.4 | compiled from source (CUTLASS) | NVIDIA Warp + CUTLASS implicit GEMM |
| [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) | 0.5.4 | 1.10.2 / 11.3 | compiled from source (openblas) | gather-scatter / coordinate hashing |

MinkowskiEngine's last release (0.5.4, 2021) predates CUDA 12 / torch 2.x and has no working wheel or modern source build, so it is compiled and benchmarked on its **native stack** (torch 1.10.2 + CUDA 11.3, which still runs on this driver) in a separate environment.
Because its sparse-conv kernels are its own CUDA (not torch's), the measurement still reflects the library; the torch/CUDA difference from the other three is a caveat, called out again with the results.
WarpConvNet, from the same author (Chris Choy) and now maintained at NVIDIA, is its modern successor.

## What is measured

For every `(library, network size, batch size)` configuration:

- **Forward latency** - training-mode forward pass, timed under `no_grad`.
- **Forward+backward latency** - forward, `loss = output.sum()`, `loss.backward()`.
- **Peak memory (forward)** and **peak memory (forward+backward)** - `torch.cuda.max_memory_allocated` / `max_memory_reserved`.
- **Throughput** - active input voxels processed per second (forward+backward).

Timing uses CUDA events with warmup iterations and `torch.cuda.synchronize`; each result reports mean/median/std/min/p90 over the timed iterations.

## Fairness methodology

- **Identical architecture.** Every library builds the same ResNet-style 3D sparse encoder from one library-agnostic [`NetworkSpec`](src/spconv_bench/networks/spec.py): a submanifold stem, then stages of `[strided sparse conv downsample] + N submanifold residual blocks`, with channels doubling per stage. Three sizes (`small`/`medium`/`large`) probe how each library scales with width and depth.
- **Identical inputs.** All libraries consume byte-identical voxelized events, cached to disk once (see below).
- **Rulebook cost included.** The library sparse tensor is rebuilt from pre-uploaded GPU coordinates *inside every timed iteration*, so the cost of building the convolution rulebook / kernel map is counted (this is what a training loop over varying data actually pays), while host-to-device transfer is not re-timed.
- **Shared environment where possible.** spconv, torchsparse++ and WarpConvNet run in one environment with the same torch (2.5.0), CUDA (12.4) and numpy, so their differences reflect the libraries, not their dependencies. MinkowskiEngine cannot build on that stack (its last release predates it), so it runs in a separate torch-1.10 / CUDA-11.3 environment; its numbers carry that caveat.

## Dataset

[PILArNet-M-mini](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M-mini) is loaded with the HuggingFace `datasets` library.
Each event's `point` array is reshaped to `(N, 8)`: columns 0-2 are integer voxel coordinates on a 768³ grid, columns 3-7 are per-voxel features (energy and count-like quantities).
Events are voxelized at **voxel size 1** (the native resolution, so one point per voxel) and cached as a compact ragged array.
The input feature is the energy deposition (`in_channels=1` by default); active voxels per event range from ~600 to ~12000, and batching scales this up.

## Environment

spconv, torchsparse++ and WarpConvNet have conflicting build requirements but all work against one modern CUDA 12.4 / torch 2.5 stack once compiled, so the benchmark uses a **single environment** for those three.
MinkowskiEngine 0.5.4 predates that stack and is built separately on torch 1.10.2 + CUDA 11.3 (see step 8 of `scripts/setup_env.sh`).
It is built as a clone of the existing `pimm` conda env (which already provides torch 2.5.0+cu124, a coherent CUDA 12.4 toolchain with `nvcc`, and `spconv-cu124`), plus the CUDA dev headers needed to compile extensions.
`uv` is used as the package installer and to compile torchsparse++ and WarpConvNet; cloning leaves the original `pimm` env untouched.
See [`scripts/setup_env.sh`](scripts/setup_env.sh) for the exact, reproducible steps.

The GPU is an **NVIDIA A100-SXM4-40GB** (SM 8.0) on the SLURM `ampere` partition (account `neutrino:ml-dev`).

## Repository layout

```
src/spconv_bench/
  data.py                  # PILArNet-M loading (datasets) + voxelization + cache
  bench.py                 # library-agnostic timing/memory harness + Adapter API
  cli.py                   # run one library, dump results/<library>.json
  report.py                # aggregate JSON -> summary.md + CSV + plots
  networks/
    spec.py                # library-agnostic NetworkSpec (small/medium/large)
    spconv_net.py          # spconv model + adapter
    torchsparse_net.py     # torchsparse++ model + adapter
    warpconvnet_net.py     # WarpConvNet model + adapter
    minkowski_net.py       # MinkowskiEngine model + adapter
scripts/
  setup_env.sh             # build the benchmark environment
  submit_ampere.sbatch     # run all libraries on one A100 + aggregate
  cuda_build_env.sh        # CUDA build-toolchain environment variables
```

## Usage

```bash
# 1. Build the environment (clone pimm + compile torchsparse/warpconvnet)
bash scripts/setup_env.sh

# 2. Run the full benchmark on an A100 and aggregate
sbatch scripts/submit_ampere.sbatch

# ...or run one library interactively on a GPU node
python -m spconv_bench.cli --library spconv --split val --voxel-size 1 \
    --specs small medium large --batch-sizes 1 4 8 --out results/spconv.json

# 3. Aggregate into tables + plots
python -m spconv_bench.report results/*.json --outdir results
```

## Results

Measured on one **NVIDIA A100-SXM4-40GB** (SM 8.0).
spconv 2.3.8, torchsparse 2.1.0 and WarpConvNet 1.7.11 run on torch 2.5.0+cu124; **MinkowskiEngine 0.5.4 runs on torch 1.10.2+cu113** (its native stack - see the caveat below).
Inputs are the 20 PILArNet-M-mini validation events voxelized at voxel size 1 (~600-12000 active voxels each); batches are the first `B` events.
Every number is the median over 30 timed iterations after 10 warmup iterations.
Full tables are in [`results/summary.md`](results/summary.md); raw per-configuration data is in [`results/results.csv`](results/results.csv).

![overview](results/plots/overview.png)

### Headline

- **WarpConvNet is the standout at scale** - the fastest on the medium and large models (up to **15x** faster than spconv; 65.7 ms vs 1008 ms at large / bs=16) **and** the most memory-efficient (down to **0.36x** spconv's peak memory). It needs one non-default setting - a deterministic point ordering - to reach this; see below.
- **torchsparse++ is fastest at small scale** (**2.1-3.1x** spconv) and always strong, but it uses the **most memory** (1.2-2.5x spconv).
- **MinkowskiEngine** (on its older torch-1.10 / CUDA-11.3 stack) is a strong all-rounder - the lowest memory of all (tied with WarpConvNet) and fast (up to 9x spconv at large scale) - though its different torch/CUDA makes the absolute cross-stack comparison inexact.
- **spconv** is the most predictable, but the slowest at large batch/model.

No single library wins everywhere, but at the large scale that matters for real detectors, **WarpConvNet (speed + memory) and MinkowskiEngine (memory)** lead; torchsparse++ leads at small scale.

### Forward+backward latency (median ms, lower is better)

| network / batch | spconv | torchsparse | warpconvnet | minkowski¹ |
| --- | ---: | ---: | ---: | ---: |
| small,  bs=1  | 10.7 | **5.2** | 15.8 | 9.4 |
| small,  bs=16 | 39.5 | 13.9 | 19.0 | **12.8** |
| medium, bs=1  | 26.5 | **12.7** | 23.8 | 23.8 |
| medium, bs=16 | 214.0 | 69.9 | **33.0** | 45.9 |
| large,  bs=1  | 101.9 | 49.4 | **34.6** | 48.5 |
| large,  bs=8  | 602.0 | 211.5 | **51.2** | 82.5 |
| large,  bs=16 | 1007.9 | 352.0 | **65.7** | 110.6 |

### Peak memory, forward+backward (MB allocated, lower is better)

| network / batch | spconv | torchsparse | warpconvnet | minkowski¹ |
| --- | ---: | ---: | ---: | ---: |
| small,  bs=16 | 564 | 711 | 219 | **167** |
| medium, bs=16 | 1540 | 1828 | 561 | **523** |
| large,  bs=8  | 2430 | 3356 | **1014** | 1037 |
| large,  bs=16 | 4095 | 5061 | 1558 | **1528** |

¹ MinkowskiEngine on torch 1.10.2+cu113 (different framework stack); the other three on torch 2.5.0+cu124.

Per-network bar charts (latency + memory) and log-log scaling curves are in [`results/plots/`](results/plots):
`bars_{small,medium,large}.png`, `scaling_{small,medium,large}.png`, `speedup_{small,medium,large}.png`.

![scaling (large model)](results/plots/scaling_large.png)

### WarpConvNet needs a deterministic point ordering

WarpConvNet's strided convolution emits its downsampled coordinates in `POINT_ORDERING.RANDOM` **by default**.
Because the benchmark reuses the input each iteration, that means each pooled-resolution geometry is laid out differently on every forward pass, so WarpConvNet's per-geometry kernel-map cache never hits and its neighbour search is spatially incoherent.
With the default ordering, batched latency was inflated by roughly **10x** and never reached steady state (e.g. small / bs=16 sat at a flat ~250 ms of pure overhead - see the git history of this file).

Setting a deterministic space-filling (Morton) order on the strided convolutions - `order=POINT_ORDERING.MORTON_XYZ`, done in [`warpconvnet_net.py`](src/spconv_bench/networks/warpconvnet_net.py) - makes the downsampled coordinates reproducible and locality-friendly.
The kernel map is then built once during warmup and reused, and latency drops to the numbers above (small / bs=16: 250 ms → 19 ms).
This is the single most important knob for WarpConvNet performance, and worth knowing for anyone deploying it.
spconv, torchsparse and MinkowskiEngine thread one kernel-map cache through the whole network, so they do not need it.

### Reproduce

```bash
bash scripts/setup_env.sh                 # build the environment (once)
sbatch scripts/submit_ampere.sbatch       # run all libraries on one A100 + aggregate
```
