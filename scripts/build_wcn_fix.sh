#!/usr/bin/env bash
# Build WarpConvNet from the fix/cute-grouped-sm90-cpasync-race branch for
# sm_80 (A100) + sm_90 (H200), using the CUDA 12.8 toolkit (envs/cudatk128).
# The release 1.7.11's Hopper CuTe kernels are numerically broken (NaN); this
# branch fixes the cp.async race so the auto-tuned kernels are correct+fast.
set -eo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA=/sdf/group/neutrino/youngsam/miniforge3/bin/conda
BENCH=/sdf/group/neutrino/youngsam/.conda/envs/pimm-bench
PY="$BENCH/bin/python"
CUDATK="$PROJ/envs/cudatk128"
SRC="$PROJ/scratch/WarpConvNet-fix"
BRANCH=fix/cute-grouped-sm90-cpasync-race

source "$(dirname "$CONDA")/../etc/profile.d/conda.sh"
conda activate "$BENCH"

# 1. Clone (recursive for CUTLASS submodule) if not already present.
if [[ ! -d "$SRC/.git" ]]; then
    echo ">>> cloning $BRANCH -> $SRC"
    rm -rf "$SRC"
    git clone --recursive --branch "$BRANCH" \
        https://github.com/NVlabs/WarpConvNet.git "$SRC"
else
    echo ">>> reusing clone at $SRC"
    git -C "$SRC" submodule update --init --recursive
fi
echo ">>> HEAD: $(git -C "$SRC" rev-parse --short HEAD)"
echo ">>> CUTLASS: $(ls "$SRC"/**/cutlass/include 2>/dev/null | head -1 || echo MISSING)"

# 2. Point the build at the CUDA 12.8 toolkit (compiles Hopper CuTe kernels;
#    12.4's ptxas rejects them). libcuda stub for -lcuda under lib/stubs.
export CUDA_HOME="$CUDATK"
export PATH="$CUDATK/bin:$PATH"
TARGETS="$CUDATK/targets/x86_64-linux"
export CPATH="$TARGETS/include:${CPATH:-}"
export LIBRARY_PATH="$TARGETS/lib:$TARGETS/lib/stubs:$CUDATK/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$TARGETS/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0 9.0"
export MAX_JOBS="${MAX_JOBS:-12}"

echo ">>> nvcc: $(nvcc --version | tail -1)"
echo ">>> building warpconvnet (arch $TORCH_CUDA_ARCH_LIST, CUDA 12.8)"
uv pip install --python "$PY" --no-build-isolation --no-cache --reinstall "$SRC"

echo ">>> build DONE; import check on GPU node is separate"
