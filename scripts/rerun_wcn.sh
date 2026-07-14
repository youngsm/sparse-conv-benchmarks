#!/usr/bin/env bash
# Re-run ONLY the warpconvnet benchmark (fix-branch build) into $OUTDIR, with the
# default AUTO GEMM-algorithm selection. Invoke under srun on the target GPU:
#   OUTDIR=results       bash scripts/rerun_wcn.sh   # A100
#   OUTDIR=results/h200  bash scripts/rerun_wcn.sh   # H200
set -o pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
CONDA_SH=/sdf/group/neutrino/youngsam/miniforge3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate /sdf/group/neutrino/youngsam/.conda/envs/pimm-bench

OUTDIR="${OUTDIR:-results}"
SPECS="${SPECS:-small medium large}"
BATCHES="${BATCHES:-1 2 4 8 16}"
export WARPCONVNET_BENCHMARK_CACHE_DIR="${WARPCONVNET_BENCHMARK_CACHE_DIR:-$PROJ/.warpconvnet_cache_fix}"
mkdir -p "$OUTDIR"

echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
python - <<'PY'
import warpconvnet, torch
print("warpconvnet from", warpconvnet.__file__)
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
PY

python -m spconv_bench.cli --library warpconvnet --precision bf16 \
    --split val --voxel-size 1 \
    --specs $SPECS --batch-sizes $BATCHES --n-warmup 20 --n-iters 30 \
    --out "$OUTDIR/warpconvnet.json"
echo "WCN_RERUN_DONE $OUTDIR"
