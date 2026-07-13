"""PILArNet-M data pipeline.

Loads the PILArNet-M(-mini) dataset with the HuggingFace ``datasets`` library,
voxelizes each event at a configurable voxel size, and caches the result as a
compact ragged array on the shared filesystem so that *every* library benchmark
consumes byte-for-byte identical inputs.

PILArNet-M ``point`` layout (flat float array, reshaped to ``(N, 8)``):

===  ===========================================================
col  meaning
===  ===========================================================
0-2  x, y, z voxel coordinates (integers on a 768^3 grid)
3    energy deposition (well-behaved, ~[0.01, 18.5])
4    energy deposition (near-duplicate of col 3)
5    count-like feature (large dynamic range)
6    count-like feature (large dynamic range)
7    count-like feature
===  ===========================================================

At voxel size 1 the coordinates are already unique (one point per voxel), so
voxelization is an identity + shift; the general path (voxel_size > 1) merges
collisions by averaging their features.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

REPO_ID = "DeepLearnPhysics/PILArNet-M-mini"
POINT_NCOLS = 8
COORD_COLS: Tuple[int, int, int] = (0, 1, 2)
# feature columns, energy first so `in_channels=1` picks the well-behaved one
FEATURE_COLS: Tuple[int, ...] = (3, 4, 5, 6, 7)
GRID_SIZE = 768  # native PILArNet-M resolution per axis

_PROJ = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _PROJ / "data" / "cache"


def _ensure_hf_home() -> None:
    """Point the HF cache at the shared FS unless the user already set it."""
    os.environ.setdefault("HF_HOME", str(_PROJ / "data" / "hf_cache"))


@dataclass
class Event:
    """A single voxelized event."""

    coords: np.ndarray  # (N, 3) int32, non-negative voxel indices
    feats: np.ndarray  # (N, F) float32 features
    n_points_raw: int  # points before voxelization

    @property
    def n_voxels(self) -> int:
        return self.coords.shape[0]


@dataclass
class Batch:
    """A library-neutral batch: coords carry a leading batch index."""

    coords: np.ndarray  # (M, 4) int32: [batch_idx, x, y, z]
    feats: np.ndarray  # (M, F) float32
    batch_size: int
    spatial_shape: Tuple[int, int, int]

    @property
    def n_voxels(self) -> int:
        return self.coords.shape[0]

    def feat_slice(self, in_channels: int) -> np.ndarray:
        """First ``in_channels`` feature columns (contiguous float32)."""
        return np.ascontiguousarray(self.feats[:, :in_channels])


def _reshape_point(flat) -> np.ndarray:
    arr = np.asarray(flat, dtype=np.float32)
    if arr.size % POINT_NCOLS != 0:
        raise ValueError(f"point array size {arr.size} not divisible by {POINT_NCOLS}")
    return arr.reshape(-1, POINT_NCOLS)


def voxelize(
    points: np.ndarray,
    voxel_size: float = 1.0,
    feature_cols: Sequence[int] = FEATURE_COLS,
    reduce: str = "mean",
) -> Event:
    """Quantize a raw ``(N, 8)`` point array into unique voxels.

    Coordinates are divided by ``voxel_size``, floored, and shifted so the
    minimum is the origin. Points landing in the same voxel have their features
    reduced by ``mean`` (default) or ``sum``.
    """
    if points.ndim != 2 or points.shape[1] != POINT_NCOLS:
        raise ValueError(f"expected (N,{POINT_NCOLS}) points, got {points.shape}")

    xyz = points[:, list(COORD_COLS)].astype(np.float64)
    q = np.floor(xyz / float(voxel_size)).astype(np.int64)
    q -= q.min(axis=0, keepdims=True)  # origin at 0

    feats = points[:, list(feature_cols)].astype(np.float64)

    key, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    agg = np.zeros((key.shape[0], feats.shape[1]), dtype=np.float64)
    np.add.at(agg, inv, feats)
    if reduce == "mean":
        counts = np.bincount(inv, minlength=key.shape[0]).astype(np.float64)
        agg /= np.maximum(counts[:, None], 1.0)
    elif reduce != "sum":
        raise ValueError(f"unknown reduce={reduce!r}")

    return Event(
        coords=key.astype(np.int32),
        feats=agg.astype(np.float32),
        n_points_raw=int(points.shape[0]),
    )


def _cache_path(split: str, voxel_size: float, cache_dir: Path) -> Path:
    tag = f"vox{voxel_size:g}".replace(".", "p")
    return cache_dir / f"pilarnet_m_mini_{split}_{tag}.npz"


def build_cache(
    split: str = "train",
    voxel_size: float = 1.0,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    limit: Optional[int] = None,
) -> Path:
    """Load a split via ``datasets`` and write a voxelized ragged-array cache."""
    _ensure_hf_home()
    from datasets import load_dataset

    hf_split = {"val": "validation"}.get(split, split)
    ds = load_dataset(REPO_ID, split=hf_split)
    n = len(ds) if limit is None else min(limit, len(ds))

    coords_list, feats_list, offsets = [], [], [0]
    for i in range(n):
        pts = _reshape_point(ds[i]["point"])
        ev = voxelize(pts, voxel_size=voxel_size)
        coords_list.append(ev.coords)
        feats_list.append(ev.feats)
        offsets.append(offsets[-1] + ev.n_voxels)

    coords = np.concatenate(coords_list, axis=0).astype(np.int32)
    feats = np.concatenate(feats_list, axis=0).astype(np.float32)
    offsets = np.asarray(offsets, dtype=np.int64)

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(split, voxel_size, cache_dir)
    np.savez(
        path,
        coords=coords,
        feats=feats,
        offsets=offsets,
        voxel_size=np.float64(voxel_size),
        grid_size=np.int64(GRID_SIZE),
    )
    return path


def load_events(
    split: str = "train",
    voxel_size: float = 1.0,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    rebuild: bool = False,
) -> List[Event]:
    """Return the voxelized events for a split, building the cache if needed."""
    path = _cache_path(split, voxel_size, cache_dir)
    if rebuild or not path.exists():
        path = build_cache(split, voxel_size, cache_dir)

    with np.load(path) as z:
        coords, feats, offsets = z["coords"], z["feats"], z["offsets"]

    events = []
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        events.append(
            Event(
                coords=coords[s:e].copy(),
                feats=feats[s:e].copy(),
                n_points_raw=e - s,
            )
        )
    return events


def make_batch(
    events: Sequence[Event],
    spatial_shape: Optional[Tuple[int, int, int]] = None,
) -> Batch:
    """Concatenate events into one batch with a leading batch index.

    Coordinates are ``[batch_idx, x, y, z]``. Each library adapter maps this to
    its own tensor convention. ``spatial_shape`` defaults to the tight bound
    over the batch (each axis = max coord + 1).
    """
    coords_parts, feats_parts = [], []
    for b, ev in enumerate(events):
        n = ev.n_voxels
        bcol = np.full((n, 1), b, dtype=np.int32)
        coords_parts.append(np.concatenate([bcol, ev.coords], axis=1))
        feats_parts.append(ev.feats)

    coords = np.concatenate(coords_parts, axis=0).astype(np.int32)
    feats = np.concatenate(feats_parts, axis=0).astype(np.float32)

    if spatial_shape is None:
        mx = coords[:, 1:4].max(axis=0) + 1
        spatial_shape = (int(mx[0]), int(mx[1]), int(mx[2]))

    return Batch(
        coords=coords,
        feats=feats,
        batch_size=len(events),
        spatial_shape=spatial_shape,
    )


def iter_batches(
    events: Sequence[Event],
    batch_size: int,
    spatial_shape: Optional[Tuple[int, int, int]] = None,
    drop_last: bool = True,
) -> List[Batch]:
    """Group events into fixed-size batches."""
    batches = []
    n = len(events)
    for start in range(0, n, batch_size):
        chunk = events[start : start + batch_size]
        if drop_last and len(chunk) < batch_size:
            break
        batches.append(make_batch(chunk, spatial_shape=spatial_shape))
    return batches
