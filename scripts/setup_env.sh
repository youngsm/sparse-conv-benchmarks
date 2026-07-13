#!/usr/bin/env bash
# Build the benchmark environment.
#
# Design: all three libraries run in ONE environment so the comparison uses an
# identical torch / CUDA / numpy. We start from a clone of the existing `pimm`
# conda env (torch 2.5.0+cu124, a coherent CUDA 12.4 toolchain incl. nvcc, and
# spconv-cu124 already present), add the CUDA *dev* headers needed to compile
# extensions, then use `uv` as the installer for the orchestration package and
# to compile torchsparse++ and WarpConvNet against that CUDA 12.4 toolchain.
#
# Cloning keeps the user's working `pimm` env untouched. Run from anywhere:
#     bash scripts/setup_env.sh
# no `set -u` -- conda activation scripts reference unbound vars
set -eo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA=/sdf/group/neutrino/youngsam/miniforge3/bin/conda
SRC_ENV=/sdf/home/y/youngsam/sw/dune/.conda/envs/pimm
BENCH=/sdf/home/y/youngsam/sw/dune/.conda/envs/pimm-bench
PY="$BENCH/bin/python"
TS_COMMIT=385f5ce8718fcae93540511b7f5832f4e71fd835   # torchsparse master (torchsparse++)
WCN_VERSION=1.7.11

# 1. Clone pimm (skip if the clone already exists). pimm already contains a
#    complete, coherent CUDA 12.4 toolchain: nvcc, the cuda*-dev headers (under
#    targets/x86_64-linux/include), conda gcc/gxx, the libcuda stub, cccl
#    (cub/thrust), and google-sparsehash -- so no extra CUDA packages are needed.
if [[ ! -x "$PY" ]]; then
    echo ">>> cloning $SRC_ENV -> $BENCH"
    "$CONDA" create -y --clone "$SRC_ENV" -p "$BENCH"
fi

# 2. Activate the env (sets conda gcc/gxx + CUDA_HOME) and point the build at the
#    CUDA headers/libs. conda keeps CUDA headers under targets/x86_64-linux; the
#    libcuda *stub* (for WarpConvNet's `-lcuda`) is under lib/stubs, and
#    sparsehash headers (for torchsparse) are under include/.
source "$(dirname "$CONDA")/../etc/profile.d/conda.sh"
conda activate "$BENCH"
TARGETS="$BENCH/targets/x86_64-linux"
export CUDA_HOME="$BENCH"
export CPATH="$TARGETS/include:$BENCH/include:${CPATH:-}"
export LIBRARY_PATH="$TARGETS/lib:$TARGETS/lib/stubs:$BENCH/lib:$BENCH/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$TARGETS/lib:$BENCH/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"   # A100 (SM 8.0)
export MAX_JOBS="${MAX_JOBS:-16}"

# 4. Orchestration package (editable) + Warp runtime, installed with uv.
echo ">>> installing orchestration package + warp-lang (uv)"
uv pip install --python "$PY" tabulate warp-lang
uv pip install --python "$PY" -e "$PROJ"

# 5. torchsparse++ (compiled from source; needs nvcc + sparsehash).
#    FORCE_CUDA=1 is required because torchsparse's setup.py otherwise gates CUDA
#    kernel compilation on torch.cuda.is_available(), which is False on a GPU-less
#    build/login node -- yielding a CPU-only backend that fails at runtime.
echo ">>> building torchsparse++ @ $TS_COMMIT"
FORCE_CUDA=1 uv pip install --python "$PY" --no-build-isolation \
    "git+https://github.com/mit-han-lab/torchsparse.git@${TS_COMMIT}"

# 6. WarpConvNet (compiled from PyPI sdist; bundles CUTLASS).
echo ">>> building warpconvnet==$WCN_VERSION"
uv pip install --python "$PY" --no-build-isolation "warpconvnet==${WCN_VERSION}"

# 7. Sanity check.
echo ">>> import check"
"$PY" - <<'PY'
import torch, spconv, torchsparse, warpconvnet
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("spconv / torchsparse / warpconvnet import OK")
PY
echo ">>> main env DONE"

# ---------------------------------------------------------------------------
# 8. MinkowskiEngine (separate env). Its last release (0.5.4) predates CUDA 12 /
#    torch 2.x, so it is built on its native stack -- torch 1.10.2 + CUDA 11.3
#    (which still runs on the driver here). This is a different torch/CUDA than
#    the main env; the sparse-conv kernels are MinkowskiEngine's own, and the
#    difference is noted in the README.
# ---------------------------------------------------------------------------
ME_ENV=/sdf/home/y/youngsam/sw/dune/.conda/envs/me-bench
if [[ ! -x "$ME_ENV/bin/python" ]]; then
    echo ">>> creating MinkowskiEngine env (python 3.9 + CUDA 11.3 + openblas)"
    "$CONDA" create -y -p "$ME_ENV" -c nvidia/label/cuda-11.3.1 -c conda-forge \
        python=3.9 cuda-toolkit openblas libopenblas ninja
fi
conda activate "$ME_ENV"
export CUDA_HOME="$ME_ENV"
export CPATH="$ME_ENV/include:${CPATH:-}"
export LIBRARY_PATH="$ME_ENV/lib:$ME_ENV/lib/stubs:${LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-16}"
uv pip install --python "$ME_ENV/bin/python" "torch==1.10.2+cu113" \
    --index-url https://download.pytorch.org/whl/cu113
uv pip install --python "$ME_ENV/bin/python" "numpy==1.23.5"
uv pip install --python "$ME_ENV/bin/python" --no-deps -e "$PROJ"
# MinkowskiEngine (compiled from source with openblas + forced CUDA)
tmp_me="$(mktemp -d)"; git clone --depth 1 https://github.com/NVIDIA/MinkowskiEngine.git "$tmp_me/ME"
( cd "$tmp_me/ME" && "$ME_ENV/bin/python" setup.py install --force_cuda --blas=openblas \
    --cuda_home="$ME_ENV" --blas_include_dirs="$ME_ENV/include" \
    --blas_library_dirs="$ME_ENV/lib" )
echo ">>> MinkowskiEngine env DONE"
echo ">>> DONE"
