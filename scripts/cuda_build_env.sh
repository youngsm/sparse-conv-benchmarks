#!/usr/bin/env bash
# Configure the CUDA 12.1 build toolchain for compiling the sparse-conv
# libraries that ship as source (warpconvnet, torchsparse) against the uv
# environment's torch==2.4.1+cu121.
#
# The toolchain (nvcc, CUDA headers, driver stub, google-sparsehash) is provided
# by a conda prefix used ONLY as a compiler / CUDA_HOME. Every Python package and
# the runtime itself live in the uv environment. Source this before building:
#
#     source scripts/cuda_build_env.sh
#
set -u

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="$PROJ/envs/cudatk121"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc" >&2
    echo "Build the toolchain first (see scripts/setup_env.sh)." >&2
    return 1 2>/dev/null || exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"

# conda CUDA lays libraries under lib/ (+ targets/.../lib); the CUDA driver stub
# libcuda.so lives in a stubs/ subdir and is needed for `-lcuda` at link time.
_libdirs=(
    "$CUDA_HOME/lib"
    "$CUDA_HOME/lib/stubs"
    "$CUDA_HOME/lib64"
    "$CUDA_HOME/lib64/stubs"
    "$CUDA_HOME/targets/x86_64-linux/lib"
    "$CUDA_HOME/targets/x86_64-linux/lib/stubs"
)
_lp=""
for d in "${_libdirs[@]}"; do
    [[ -d "$d" ]] && _lp="${_lp:+$_lp:}$d"
done
export LIBRARY_PATH="${_lp}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${_lp}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# google-sparsehash headers (torchsparse) live in the conda include dir
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

# Target only the A100 (sm_80) to keep compiles fast and match the benchmark GPU.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-16}"

echo "CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version | tail -2
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS"
