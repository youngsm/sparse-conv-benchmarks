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

- **torchsparse++ is the fastest of the torch-2.5 libraries** - **2.1-3.1x** faster than spconv on forward+backward - but it uses the **most memory** (1.2-2.5x spconv).
- **WarpConvNet is the most memory-efficient** of the torch-2.5 libraries - down to **0.37x** spconv's peak memory - and its raw GEMM kernels are fast (competitive at batch 1), but its batched latency is limited by per-forward kernel-map construction (see below).
- **spconv** sits in the middle on speed and memory, with the most predictable scaling.
- **MinkowskiEngine** (on its older torch-1.10 / CUDA-11.3 stack) is surprisingly strong on this workload - the **lowest large-model latency** and memory comparable to WarpConvNet - but the cross-stack comparison is not apples-to-apples (different torch/CUDA), so read its absolute speed against the others with that caveat.

### Forward+backward latency (median ms, lower is better)

| network / batch | spconv | torchsparse | warpconvnet | minkowski¹ |
| --- | ---: | ---: | ---: | ---: |
| small,  bs=1  | 13.9 | 6.3 | 19.7 | 8.5 |
| small,  bs=16 | 41.2 | 14.3 | 253.5 | 12.7 |
| medium, bs=1  | 29.1 | 13.7 | 30.1 | 24.0 |
| medium, bs=16 | 217.1 | 70.4 | 414.1 | 45.9 |
| large,  bs=1  | 104.2 | 49.9 | 44.0 | 49.4 |
| large,  bs=8  | 603.4 | 212.4 | 457.5 | 83.0 |
| large,  bs=16 | 1004.3 | 352.8 | 569.0 | 111.7 |

### Peak memory, forward+backward (MB allocated, lower is better)

| network / batch | spconv | torchsparse | warpconvnet | minkowski¹ |
| --- | ---: | ---: | ---: | ---: |
| small,  bs=16 | 564 | 711 | 219 | 169 |
| medium, bs=16 | 1540 | 1828 | 570 | 523 |
| large,  bs=8  | 2430 | 3356 | 1012 | 1037 |
| large,  bs=16 | 4095 | 5061 | 1559 | 1528 |

¹ MinkowskiEngine on torch 1.10.2+cu113 (different framework stack); the other three on torch 2.5.0+cu124.

Per-network bar charts (latency + memory) and log-log scaling curves are in [`results/plots/`](results/plots):
`bars_{small,medium,large}.png`, `scaling_{small,medium,large}.png`, `speedup_{small,medium,large}.png`.

![scaling (large model)](results/plots/scaling_large.png)

### How to read the WarpConvNet latency

WarpConvNet is fast at batch size 1 but shows a large jump at batch size ≥ 2 (visible in the bars above).
This is a real, explainable effect of how the libraries cache the convolution kernel map (the sparse analogue of an im2col index), not a bug in the benchmark:

- spconv and torchsparse thread **one kernel-map cache through the entire network** (it lives on the input tensor), so when the same geometry is seen again the maps at *every* resolution are reused.
- WarpConvNet caches kernel maps **per geometry object**; the input-resolution map is reused, but each strided downsample produces a fresh geometry whose kernel maps are rebuilt on every forward pass.

The benchmark measures steady state (input built once, so kernel maps and WarpConvNet's per-shape GEMM autotuning are amortized over the timed iterations - see the fairness notes in `src/spconv_bench/bench.py`).
In that regime the per-forward kernel-map construction for the pooled resolutions dominates WarpConvNet's batched latency; its raw GEMM kernels are ~0.1 ms each.
As the model/batch grows and GEMM work dominates that fixed cost, WarpConvNet closes the gap and overtakes spconv (large, bs=8/16: 1.3-1.8x faster than spconv).
In a training loop over *varying* geometry, every library rebuilds its kernel maps each step, which narrows this gap.

### Reproduce

```bash
bash scripts/setup_env.sh                 # build the environment (once)
sbatch scripts/submit_ampere.sbatch       # run all libraries on one A100 + aggregate
```
