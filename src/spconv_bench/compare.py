"""Cross-cutting comparison plots: latency/memory vs model size, and A100 vs H200.

The per-device report (``report.py``) plots each metric against batch size, one
figure per (device, model). This module adds the views that span those axes:

  * latency / memory as a function of **model size** (params), per GPU
  * a **GPU comparison** (A100 vs H200) per library
  * a comprehensive **scaling grid** (metric x model, A100 solid / H200 dashed)

Usage::

    python -m spconv_bench.compare --a100 results --h200 results/h200 \
        --outdir results/comparison
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from spconv_bench.report import COLORS, LIB_ORDER, SPEC_ORDER, load

GPU_STYLE = {"A100": "-", "H200": "--"}
GPU_MARKER = {"A100": "o", "H200": "s"}


def gather(dirs: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for gpu, d in dirs.items():
        paths = sorted(glob.glob(str(Path(d) / "*.json")))
        if not paths:
            continue
        df = load(paths)
        df = df[df["ok"]].copy()
        df["gpu"] = gpu
        frames.append(df)
    if not frames:
        raise SystemExit("no result JSONs found")
    return pd.concat(frames, ignore_index=True)


def _libs(df):
    return [l for l in LIB_ORDER if l in set(df["library"])]


def _specs(df):
    return [s for s in SPEC_ORDER if s in set(df["spec"])]


def _gpus(df):
    return [g for g in ["A100", "H200"] if g in set(df["gpu"])]


def _params_by_spec(df) -> Dict[str, float]:
    return {s: df[df["spec"] == s]["n_params"].iloc[0] / 1e6 for s in _specs(df)}


def make_plots(df: pd.DataFrame, outdir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    outdir.mkdir(parents=True, exist_ok=True)
    libs, specs, gpus = _libs(df), _specs(df), _gpus(df)
    pmap = _params_by_spec(df)
    px = [pmap[s] for s in specs]
    written: List[Path] = []

    def color(l):
        return COLORS.get(l)

    # ---- 1 & 2. latency / memory vs MODEL SIZE, one panel per GPU, at fixed bs
    for value, ylabel, fname, title in [
        ("fwdbwd_ms", "Fwd+bwd latency (ms)", "latency_vs_modelsize.png",
         "Latency vs model size"),
        ("mem_fwdbwd_mb", "Peak memory fwd+bwd (MB)", "memory_vs_modelsize.png",
         "Memory vs model size"),
    ]:
        for bs in [8]:
            fig, axes = plt.subplots(1, len(gpus), figsize=(6 * len(gpus), 4.6),
                                     squeeze=False, sharey=True)
            for j, gpu in enumerate(gpus):
                ax = axes[0][j]
                for lib in libs:
                    ys = [
                        df[(df.gpu == gpu) & (df.library == lib) & (df.spec == s)
                           & (df.batch_size == bs)][value].mean()
                        for s in specs
                    ]
                    ax.plot(px, ys, "o-", color=color(lib), label=lib, lw=2, ms=7)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xticks(px)
                ax.set_xticklabels([f"{s}\n{pmap[s]:.1f}M" for s in specs])
                ax.set_xlabel("model (parameters)")
                if j == 0:
                    ax.set_ylabel(ylabel)
                ax.set_title(f"{gpu}")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(fontsize=9)
            fig.suptitle(f"{title}  (batch size {bs}, bf16)")
            fig.tight_layout()
            out = outdir / fname
            fig.savefig(out, dpi=140)
            plt.close(fig)
            written.append(out)

    # ---- 3. A100 vs H200 grouped bars per library (headline: large, bs=16) ----
    if len(gpus) >= 2:
        spec, bs = ("large" if "large" in specs else specs[-1]), (
            16 if 16 in set(df.batch_size) else sorted(df.batch_size)[-1])
        panels = [("fwdbwd_ms", "Fwd+bwd latency (ms)"),
                  ("mem_fwdbwd_mb", "Peak memory fwd+bwd (MB)")]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        x = np.arange(len(libs))
        w = 0.8 / len(gpus)
        gpu_col = {"A100": "#4C72B0", "H200": "#C44E52"}
        for ax, (value, ylabel) in zip(axes, panels):
            for k, gpu in enumerate(gpus):
                vals = [
                    df[(df.gpu == gpu) & (df.library == lib) & (df.spec == spec)
                       & (df.batch_size == bs)][value].mean()
                    for lib in libs
                ]
                bars = ax.bar(x + k * w, vals, width=w, label=gpu, color=gpu_col.get(gpu))
                for b, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                                ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x + w * (len(gpus) - 1) / 2)
            ax.set_xticklabels(libs, rotation=15)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel.split(" (")[0])
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend()
        fig.suptitle(f"A100 vs H200  ({spec} model, batch {bs}, bf16)")
        fig.tight_layout()
        out = outdir / "gpu_compare.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(out)

    # ---- 4. comprehensive scaling grid: metric x model, A100 solid/H200 dashed
    metrics = [("fwdbwd_ms", "Fwd+bwd latency (ms)"),
               ("mem_fwdbwd_mb", "Peak memory fwd+bwd (MB)")]
    fig, axes = plt.subplots(len(metrics), len(specs),
                             figsize=(5 * len(specs), 4.4 * len(metrics)), squeeze=False)
    for r, (value, ylabel) in enumerate(metrics):
        for c, spec in enumerate(specs):
            ax = axes[r][c]
            for lib in libs:
                for gpu in gpus:
                    s = df[(df.gpu == gpu) & (df.library == lib) & (df.spec == spec)]
                    s = s.sort_values("batch_size")
                    if s.empty:
                        continue
                    ax.plot(s["batch_size"], s[value], GPU_STYLE.get(gpu, "-"),
                            marker=GPU_MARKER.get(gpu, "o"), color=color(lib),
                            lw=1.8, ms=5, alpha=0.9)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(sorted(df["batch_size"].unique()))
            ax.set_xticklabels(sorted(df["batch_size"].unique()))
            if r == len(metrics) - 1:
                ax.set_xlabel("batch size")
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 0:
                ax.set_title(f"{spec}  ({_params_by_spec(df)[spec]:.1f}M params)")
            ax.grid(True, which="both", alpha=0.3)
    # combined legend: colors = libraries, linestyle = GPU
    handles = [Line2D([0], [0], color=color(l), lw=2, label=l) for l in libs]
    handles += [Line2D([0], [0], color="0.3", lw=2, ls=GPU_STYLE[g],
                       marker=GPU_MARKER[g], label=g) for g in gpus]
    fig.legend(handles=handles, loc="upper center", ncol=len(libs) + len(gpus),
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Scaling with batch size, model, and GPU (bf16)", y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = outdir / "scaling_grid.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    written.append(out)

    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a100", default="results", help="dir with A100 result JSONs")
    ap.add_argument("--h200", default="results/h200", help="dir with H200 result JSONs")
    ap.add_argument("--outdir", default="results/comparison")
    args = ap.parse_args(argv)

    df = gather({"A100": args.a100, "H200": args.h200})
    for p in make_plots(df, Path(args.outdir)):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
