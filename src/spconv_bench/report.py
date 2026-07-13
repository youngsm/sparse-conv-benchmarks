"""Aggregate per-library result JSON into comprehensive tables and plots.

Usage::

    python -m spconv_bench.report results/*.json --outdir results

Produces:
  - results.csv            : tidy long-form table of every measurement
  - summary.md             : markdown tables (latency, memory, throughput,
                             speedup vs spconv, memory ratio vs spconv)
  - plots/bars_<spec>.png  : grouped bars (latency + memory) vs batch size
  - plots/scaling_<spec>.png : latency & memory vs active voxels (log-log)
  - plots/speedup_<spec>.png : speedup vs spconv baseline
  - plots/overview.png     : one figure summarizing latency + memory per spec
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

SPEC_ORDER = ["small", "medium", "large"]
LIB_ORDER = ["spconv", "torchsparse", "warpconvnet", "minkowski"]
COLORS = {
    "spconv": "#4C72B0",
    "torchsparse": "#DD8452",
    "warpconvnet": "#55A868",
    "minkowski": "#C44E52",
}
BASELINE = "spconv"


def load(paths: List[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        for r in d["results"]:
            row = dict(
                library=r["library"],
                lib_version=r["library_version"],
                torch=r["torch_version"],
                cuda=r["cuda_version"],
                device=r["device_name"],
                spec=r["spec_name"],
                batch_size=r["batch_size"],
                n_voxels=r["n_voxels"],
                n_params=r["n_params"],
                ok=r["ok"],
                error=r.get("error", ""),
            )
            if r["ok"]:
                row.update(
                    fwd_ms=r["forward"]["median_ms"],
                    fwd_ms_std=r["forward"]["std_ms"],
                    fwdbwd_ms=r["forward_backward"]["median_ms"],
                    fwdbwd_ms_std=r["forward_backward"]["std_ms"],
                    mem_fwd_mb=r["mem_forward"]["peak_alloc_mb"],
                    mem_fwdbwd_mb=r["mem_forward_backward"]["peak_alloc_mb"],
                    mem_fwdbwd_reserved_mb=r["mem_forward_backward"]["peak_reserved_mb"],
                    kvox_s=r["throughput_kvox_s"],
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _lib_cat(df: pd.DataFrame) -> pd.DataFrame:
    libs = [l for l in LIB_ORDER if l in set(df["library"])]
    libs += [l for l in df["library"].unique() if l not in libs]
    df = df.copy()
    df["library"] = pd.Categorical(df["library"], categories=libs, ordered=True)
    return df


def _sort_index(table: pd.DataFrame) -> pd.DataFrame:
    return table.reindex(
        sorted(
            table.index,
            key=lambda t: (SPEC_ORDER.index(t[0]) if t[0] in SPEC_ORDER else 9, t[1]),
        )
    )


def pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    ok = _lib_cat(df[df["ok"]])
    table = ok.pivot_table(
        index=["spec", "batch_size", "n_voxels"],
        columns="library",
        values=value,
        observed=True,
    )
    return _sort_index(table)


def ratio_table(df: pd.DataFrame, value: str, invert: bool = False) -> Optional[pd.DataFrame]:
    """Ratio of each library to the spconv baseline for `value`.

    invert=False -> baseline/lib  (speedup: >1 means faster than spconv)
    invert=True  -> lib/baseline  (memory:  <1 means less memory than spconv)
    """
    t = pivot(df, value)
    if BASELINE not in t.columns:
        return None
    base = t[BASELINE]
    out = pd.DataFrame(index=t.index)
    for lib in t.columns:
        out[lib] = (base / t[lib]) if not invert else (t[lib] / base)
    return out


def markdown_summary(df: pd.DataFrame) -> str:
    ok = df[df["ok"]]
    lines = ["# Sparse-convolution library benchmark on PILArNet-M\n"]
    if not ok.empty:
        meta = ok.iloc[0]
        lines.append(f"- **Device:** {meta['device']}")
        lines.append(f"- **torch / CUDA:** {meta['torch']} / cu{meta['cuda']}")
        vers = ok.groupby("library", observed=True)["lib_version"].first()
        for lib in [l for l in LIB_ORDER if l in vers.index]:
            lines.append(f"- **{lib}:** {vers[lib]}")
        params = ok.groupby("spec", observed=True)["n_params"].first()
        pstr = ", ".join(
            f"{s}={params[s]/1e6:.1f}M" for s in SPEC_ORDER if s in params.index
        )
        lines.append(f"- **Model sizes (params):** {pstr}")
        lines.append("")
        lines.append(
            "Rows are `(network, batch size, active input voxels)`; columns are libraries.\n"
        )

    metrics = [
        ("fwdbwd_ms", "Forward+backward latency (median ms) - lower is better"),
        ("fwd_ms", "Forward latency (median ms) - lower is better"),
        ("mem_fwdbwd_mb", "Peak memory, forward+backward (MB allocated) - lower is better"),
        ("mem_fwd_mb", "Peak memory, forward-only (MB allocated) - lower is better"),
        ("mem_fwdbwd_reserved_mb", "Peak memory, forward+backward (MB reserved) - lower is better"),
        ("kvox_s", "Throughput (k active voxels / s, fwd+bwd) - higher is better"),
    ]
    for value, title in metrics:
        if value not in df.columns:
            continue
        t = pivot(df, value)
        if t.empty:
            continue
        lines.append(f"## {title}\n")
        lines.append(t.round(2).to_markdown())
        lines.append("")

    sp = ratio_table(df, "fwdbwd_ms", invert=False)
    if sp is not None:
        lines.append(f"## Speedup vs {BASELINE} (forward+backward) - >1 is faster than {BASELINE}\n")
        lines.append(sp.round(2).to_markdown())
        lines.append("")
    mr = ratio_table(df, "mem_fwdbwd_mb", invert=True)
    if mr is not None:
        lines.append(f"## Memory vs {BASELINE} (forward+backward) - <1 is less memory than {BASELINE}\n")
        lines.append(mr.round(2).to_markdown())
        lines.append("")

    failures = df[~df["ok"]]
    if not failures.empty:
        lines.append("## Failures\n")
        for _, r in failures.iterrows():
            lines.append(f"- {r['library']} / {r['spec']} bs={r['batch_size']}: {r['error']}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _specs_present(ok: pd.DataFrame) -> List[str]:
    return [s for s in SPEC_ORDER if s in set(ok["spec"])] + [
        s for s in ok["spec"].unique() if s not in SPEC_ORDER
    ]


def make_plots(df: pd.DataFrame, outdir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = _lib_cat(df[df["ok"]])
    if ok.empty:
        return []
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    libs = list(ok["library"].cat.categories)
    specs = _specs_present(ok)

    def color(lib):
        return COLORS.get(lib, None)

    # ---- 1. grouped bars: latency + memory vs batch size, per spec ----------
    panels = [
        ("fwdbwd_ms", "Fwd+bwd latency (ms)"),
        ("mem_fwdbwd_mb", "Peak memory fwd+bwd (MB)"),
    ]
    for spec in specs:
        sub = ok[ok["spec"] == spec]
        bss = sorted(sub["batch_size"].unique())
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2))
        for ax, (value, ylabel) in zip(axes, panels):
            x = np.arange(len(bss))
            w = 0.8 / max(1, len(libs))
            for j, lib in enumerate(libs):
                ys = [
                    sub[(sub["batch_size"] == b) & (sub["library"] == lib)][value].mean()
                    for b in bss
                ]
                ax.bar(x + j * w, ys, width=w, label=lib, color=color(lib))
            ax.set_xticks(x + w * (len(libs) - 1) / 2)
            ax.set_xticklabels([str(b) for b in bss])
            ax.set_xlabel("batch size")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel.split(" (")[0])
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend()
        fig.suptitle(f"PILArNet-M sparse CNN - {spec} model")
        fig.tight_layout()
        out = plotdir / f"bars_{spec}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        written.append(out)

    # ---- 2. scaling curves: latency & memory vs active voxels (log-log) ------
    for spec in specs:
        sub = ok[ok["spec"] == spec].sort_values("n_voxels")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for ax, (value, ylabel) in zip(
            axes,
            [("fwdbwd_ms", "Fwd+bwd latency (ms)"), ("mem_fwdbwd_mb", "Peak memory fwd+bwd (MB)")],
        ):
            for lib in libs:
                s = sub[sub["library"] == lib]
                if s.empty:
                    continue
                ax.plot(s["n_voxels"], s[value], "o-", label=lib, color=color(lib))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("active input voxels")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel.split(" (")[0] + " vs problem size")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
        fig.suptitle(f"Scaling - {spec} model")
        fig.tight_layout()
        out = plotdir / f"scaling_{spec}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        written.append(out)

    # ---- 3. speedup vs spconv, per spec -------------------------------------
    sp = ratio_table(df, "fwdbwd_ms", invert=False)
    if sp is not None:
        for spec in specs:
            rows = sp[sp.index.get_level_values("spec") == spec]
            if rows.empty:
                continue
            bss = [idx[1] for idx in rows.index]
            fig, ax = plt.subplots(figsize=(7, 4.2))
            x = np.arange(len(bss))
            w = 0.8 / max(1, len(libs))
            for j, lib in enumerate(libs):
                if lib not in rows.columns:
                    continue
                ax.bar(x + j * w, rows[lib].values, width=w, label=lib, color=color(lib))
            ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
            ax.set_xticks(x + w * (len(libs) - 1) / 2)
            ax.set_xticklabels([str(b) for b in bss])
            ax.set_xlabel("batch size")
            ax.set_ylabel(f"speedup vs {BASELINE} (fwd+bwd)")
            ax.set_title(f"Speedup vs {BASELINE} - {spec} model  (>1 is faster)")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            out = plotdir / f"speedup_{spec}.png"
            fig.savefig(out, dpi=130)
            plt.close(fig)
            written.append(out)

    # ---- 4. overview: latency + memory across all specs ---------------------
    fig, axes = plt.subplots(len(specs), 2, figsize=(12, 3.8 * len(specs)), squeeze=False)
    for i, spec in enumerate(specs):
        sub = ok[ok["spec"] == spec]
        bss = sorted(sub["batch_size"].unique())
        for k, (value, ylabel) in enumerate(
            [("fwdbwd_ms", "Fwd+bwd latency (ms)"), ("mem_fwdbwd_mb", "Peak memory fwd+bwd (MB)")]
        ):
            ax = axes[i][k]
            x = np.arange(len(bss))
            w = 0.8 / max(1, len(libs))
            for j, lib in enumerate(libs):
                ys = [
                    sub[(sub["batch_size"] == b) & (sub["library"] == lib)][value].mean()
                    for b in bss
                ]
                ax.bar(x + j * w, ys, width=w, label=lib, color=color(lib))
            ax.set_xticks(x + w * (len(libs) - 1) / 2)
            ax.set_xticklabels([str(b) for b in bss])
            ax.set_ylabel(ylabel)
            if i == len(specs) - 1:
                ax.set_xlabel("batch size")
            if i == 0 and k == 0:
                ax.legend()
            ax.set_title(f"{spec} - {ylabel.split(' (')[0]}")
            ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("PILArNet-M sparse CNN benchmark - overview")
    fig.tight_layout()
    out = plotdir / "overview.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    written.append(out)

    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="result JSON files")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    df = load(args.paths)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df.to_csv(outdir / "results.csv", index=False)
    summary = markdown_summary(df)
    (outdir / "summary.md").write_text(summary)
    print(summary)

    if not args.no_plots:
        for p in make_plots(df, outdir):
            print(f"wrote {p}")
    print(f"wrote {outdir/'summary.md'} and {outdir/'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
