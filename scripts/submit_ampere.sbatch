#!/usr/bin/env bash
#SBATCH --account=neutrino:ml-dev
#SBATCH --partition=ampere
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=spconv-bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.out
#
# Run the full sparse-conv benchmark (spconv / torchsparse++ / WarpConvNet) on
# one A100, then aggregate results. Submit from the repo root:
#
#     sbatch scripts/submit_ampere.sbatch
#
# NOTE: no `set -u` -- conda's activation scripts reference unbound vars.
set -o pipefail

PROJ="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJ"
mkdir -p logs results

# Benchmark environment: a clone of the `pimm` conda env (torch 2.5.0+cu124,
# CUDA 12.4 toolchain) with torchsparse + warpconvnet compiled in. See
# scripts/setup_env.sh for how it is built.
CONDA_SH=/sdf/group/neutrino/youngsam/miniforge3/etc/profile.d/conda.sh
BENCH_ENV=/sdf/home/y/youngsam/sw/dune/.conda/envs/pimm-bench
source "$CONDA_SH"
conda activate "$BENCH_ENV"

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

# WarpConvNet: keep its default "auto" GEMM-algorithm selection (it autotunes
# per problem shape and caches the winner). The harness reuses the input across
# iterations, so the autotune runs during warmup and is then amortized. Persist
# the tuning cache on the shared FS so it survives across jobs.
export WARPCONVNET_BENCHMARK_CACHE_DIR="${WARPCONVNET_BENCHMARK_CACHE_DIR:-$PROJ/.warpconvnet_cache}"

SPECS="${SPECS:-small medium large}"
BATCHES="${BATCHES:-1 2 4 8 16}"
SPLIT="${SPLIT:-val}"
VOX="${VOX:-1}"
PRECISION="${PRECISION:-bf16}"   # fp32 | tf32 | bf16 | fp16

for lib in spconv torchsparse warpconvnet; do
    echo "======================== $lib ($PRECISION) ========================"
    python -m spconv_bench.cli --library "$lib" --precision "$PRECISION" \
        --split "$SPLIT" --voxel-size "$VOX" \
        --specs $SPECS --batch-sizes $BATCHES --n-warmup 20 --n-iters 30 \
        --out "results/${lib}.json" || echo "!!! $lib FAILED"
done

# MinkowskiEngine 0.5.4 has no CUDA-12/torch-2.x wheel or clean local build, so
# it runs inside the DeepLearnPhysics larcv container, which ships ME built
# against torch 2.5.1+cu121 -- the same modern stack as the other three.
SPINE_SIF="${SPINE_SIF:-/sdf/group/neutrino/images/develop.sif}"
if [[ -f "$SPINE_SIF" ]]; then
    echo "======================== minkowski ($PRECISION, container) ========================"
    apptainer exec --nv --bind "$PROJ:/work" "$SPINE_SIF" bash -c \
        "cd /work && export OMP_NUM_THREADS=16 && PYTHONPATH=/work/src python3 -m spconv_bench.cli \
         --library minkowski --precision $PRECISION --split $SPLIT --voxel-size $VOX \
         --specs $SPECS --batch-sizes $BATCHES --n-warmup 20 --n-iters 30 \
         --out /work/results/minkowski.json" || echo "!!! minkowski FAILED"
fi

echo "======================== aggregate ========================"
python -m spconv_bench.report results/*.json --outdir results || true

echo "DONE"
