"""Mosaic I/O: parse stringified mosaics, one-hot encode, and pad to a fixed size.

A knot mosaic is an n x n grid of tile types 0..10 (11 types). The CSV / SQLite
data stores a mosaic as a stringified Python list-of-lists, e.g.
    "[[0, 2, 1], [0, 6, 6], [0, 3, 4]]"

This module reuses the canonical parse/one-hot helpers from the repo's existing
``cnn_train.py`` where sensible (so the harness stays in lockstep with the
baseline pipeline) and adds variable-n padding for mixed-dimension batches.
"""

from __future__ import annotations

import ast
import os
import sys

import numpy as np

# Make the repo root importable so we can reuse mosaics.py / cnn_train.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

NUM_TILE_TYPES = 11  # tile ids 0..10


def parse_mosaic(mosaic):
    """Convert a stringified / list / ndarray mosaic into an (n, n) int array.

    Accepts the CSV string form, a Python list-of-lists, or a numpy array.
    Validates the matrix is square with entries in 0..10.
    """
    if isinstance(mosaic, str):
        mosaic = ast.literal_eval(mosaic)
    if hasattr(mosaic, "matrixRepresentation"):  # a mosaics.Mosaic instance
        mosaic = mosaic.matrixRepresentation
    if hasattr(mosaic, "tolist"):
        mosaic = mosaic.tolist()

    arr = np.asarray(mosaic, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"mosaic must be a 2D matrix, got shape {arr.shape}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"mosaic must be square, got shape {arr.shape}")
    if np.any((arr < 0) | (arr >= NUM_TILE_TYPES)):
        raise ValueError("mosaic entries must be integers from 0 to 10")
    return arr


def mosaic_to_str(arr) -> str:
    """Serialize an (n, n) int array to the canonical CSV-style string.

    Uses the exact same ``[[a, b], [c, d]]`` spacing as the dataset CSVs so the
    serialization round-trips and lexicographic comparison is well-defined.
    """
    arr = np.asarray(arr, dtype=np.int64)
    rows = ["[" + ", ".join(str(int(x)) for x in row) + "]" for row in arr]
    return "[" + ", ".join(rows) + "]"


def mosaic_to_onehot(mosaic, pad_to: int | None = None):
    """Convert a mosaic to an (n, n, 11) float32 one-hot array.

    If ``pad_to`` is given, the grid is zero-padded (with empty tiles, id 0,
    which one-hot to channel 0) up to ``pad_to x pad_to`` so mixed-dimension
    batches stack. Padding is centered is NOT done -- padding is appended on the
    bottom/right so the top-left content position is stable across dimensions.
    """
    arr = parse_mosaic(mosaic)
    if pad_to is not None:
        arr = pad_mosaic(arr, pad_to)
    return np.eye(NUM_TILE_TYPES, dtype=np.float32)[arr]


def pad_mosaic(arr, pad_to: int):
    """Zero-pad (empty tile = 0) an (n, n) int mosaic to (pad_to, pad_to)."""
    arr = np.asarray(arr, dtype=np.int64)
    n = arr.shape[0]
    if n > pad_to:
        raise ValueError(f"mosaic of size {n} exceeds pad_to={pad_to}")
    if n == pad_to:
        return arr
    out = np.zeros((pad_to, pad_to), dtype=np.int64)
    out[:n, :n] = arr
    return out


def onehot_to_mosaic(onehot):
    """Inverse: (n, n, 11) one-hot/logit array -> (n, n) int mosaic (argmax)."""
    arr = np.asarray(onehot)
    if arr.ndim < 3 or arr.shape[-1] != NUM_TILE_TYPES:
        raise ValueError(f"onehot must have final dimension 11, got shape {arr.shape}")
    return np.argmax(arr, axis=-1).astype(np.int64)


def stack_onehot(mosaics, pad_to: int | None = None):
    """One-hot a list of mosaics into a single (N, p, p, 11) batch.

    If ``pad_to`` is None it is inferred as the max grid dimension present, so
    a mixed-dimension list still stacks into one tensor.
    """
    parsed = [parse_mosaic(m) for m in mosaics]
    if pad_to is None:
        pad_to = max(a.shape[0] for a in parsed) if parsed else 0
    return np.stack([mosaic_to_onehot(a, pad_to=pad_to) for a in parsed])
