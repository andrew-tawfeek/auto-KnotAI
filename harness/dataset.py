"""Dataset loading + leakage-safe, canonical-key train/val/test splits.

The central rigor guarantee here: **all D4-symmetry-equivalent mosaics land in
the same split**. We compute each mosaic's canonical key (harness.canonical),
collapse duplicates/equivalents into one record per orbit, then split on the
canonical keys -- not on raw rows. This is what prevents the "memorize in train,
recognize in test" leakage that silently inflates accuracy.

Sources:
  * CSV (datasets/<task>/dim_*.csv) -- works now, used by the tests.
  * SQLite (mosaics.db) -- the 3.2M-row release DB. Not present locally, so the
    loader is written to the documented schema and is import-safe; it only
    touches sqlite3 when actually called with a real DB path.

SQLite schema (mosaics.db), per the project brief:
    mosaic TEXT PRIMARY KEY,   -- canonical str form
    dimension INTEGER,
    is_suitably_connected INTEGER,
    num_crossings INTEGER,
    has_crossing INTEGER,
    num_components INTEGER,
    is_unknot INTEGER,
    pd_code TEXT
"""

from __future__ import annotations

import glob
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .canonical import canonical_key
from .mosaic_io import parse_mosaic, stack_onehot

# Map a high-level task name to the SQLite label column and CSV label column.
TASK_LABELS = {
    "unknot": {"sqlite_col": "is_unknot", "csv_col": "Is Unknot"},
    "connected": {"sqlite_col": "is_suitably_connected", "csv_col": None},
    "has_crossing": {"sqlite_col": "has_crossing", "csv_col": None},
    # Multi-class crossing-number task. The raw integer label is num_crossings;
    # the loader buckets the sparse tail into a capped top class (see
    # CROSSING_CAP / bucket_crossings). CSV form stores the raw integer.
    "crossing": {"sqlite_col": "num_crossings", "csv_col": "Num Crossings"},
}

# Crossing-number is an integer >= 0 with a long, sparse tail. We frame it as
# multi-class classification with classes 0,1,2,3,4,5 and a capped "6+" bucket
# (class 6 == all num_crossings >= 6). The DB histogram (suitably-connected)
# justifies this: classes 0..5 each hold ~0.4M-1.5M mosaics and the 6+ bucket
# ~1.2M, so all 7 classes are well populated -- no absurdly tiny class.
CROSSING_CAP = 6
NUM_CROSSING_CLASSES = CROSSING_CAP + 1  # 0..CAP inclusive


def bucket_crossings(num_crossings, cap=CROSSING_CAP):
    """Map a raw num_crossings integer to its capped class id (>=cap -> cap)."""
    v = int(num_crossings)
    return cap if v >= cap else v


@dataclass
class SplitResult:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: dict = field(default_factory=dict)

    def as_tuple(self):
        return (
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
        )


# --------------------------------------------------------------------------- #
# Loading raw (mosaic_str, label) records
# --------------------------------------------------------------------------- #
def load_csv_records(csv_path, label_col=None):
    """Load (mosaic_str, label) pairs from a data_gen-style CSV."""
    df = pd.read_csv(csv_path)
    if "Mosaic" not in df.columns:
        raise ValueError(f"{csv_path}: CSV must contain a 'Mosaic' column")
    if label_col is None:
        others = [c for c in df.columns if c != "Mosaic"]
        if len(others) != 1:
            raise ValueError(
                f"{csv_path}: label_col required (columns: {list(df.columns)})"
            )
        label_col = others[0]
    mosaics = df["Mosaic"].astype(str).tolist()
    labels = df[label_col].to_numpy(dtype=np.int64)
    return list(zip(mosaics, labels.tolist()))


def load_csv_glob(pattern, label_col=None):
    """Load and concatenate records from every CSV matching a glob pattern."""
    records = []
    for path in sorted(glob.glob(pattern)):
        records.extend(load_csv_records(path, label_col=label_col))
    if not records:
        raise FileNotFoundError(f"no CSVs matched pattern: {pattern}")
    return records


def load_sqlite_records(db_path, task="unknot", dimension=None,
                        limit=None, suitably_connected_only=True):
    """Load (mosaic_str, label) pairs from the mosaics.db release DB.

    Written to the documented schema; only runs when given a real DB path. The
    label column is selected from TASK_LABELS[task].
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"{db_path} not found. The 3.2M-row mosaics.db lives on a remote "
            f"box; fetch it (see SQLITE_CHEATSHEET.md) before calling this."
        )
    col = TASK_LABELS[task]["sqlite_col"]
    where = []
    params = []
    if dimension is not None:
        where.append("dimension = ?")
        params.append(int(dimension))
    if suitably_connected_only:
        where.append("is_suitably_connected = 1")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    limit_sql = f" LIMIT {int(limit)}" if limit else ""
    sql = f"SELECT mosaic, {col} FROM mosaics{where_sql}{limit_sql}"
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [(m, int(lbl)) for (m, lbl) in rows if lbl is not None]


# --------------------------------------------------------------------------- #
# Canonical dedup
# --------------------------------------------------------------------------- #
def collapse_by_canonical(records):
    """Collapse records to one per D4 orbit, keyed by canonical form.

    records: list of (mosaic_str, label).
    Returns (canonical_records, stats) where canonical_records is a list of
    (canonical_key_str, label) with one entry per orbit, and stats reports how
    many raw rows collapsed.

    If an orbit appears with conflicting labels (should not happen for a true
    invariant, but we check), we keep the majority label and record the count.
    """
    by_key = {}
    conflicts = 0
    for mosaic_str, label in records:
        key = canonical_key(mosaic_str)
        if key not in by_key:
            by_key[key] = Counter()
        by_key[key][int(label)] += 1

    canonical_records = []
    for key, label_counts in by_key.items():
        if len(label_counts) > 1:
            conflicts += 1
        label = label_counts.most_common(1)[0][0]
        canonical_records.append((key, label))

    stats = {
        "raw_rows": len(records),
        "unique_canonical": len(by_key),
        "collapsed": len(records) - len(by_key),
        "label_conflicts": conflicts,
    }
    return canonical_records, stats


# --------------------------------------------------------------------------- #
# Leakage-safe split on canonical keys
# --------------------------------------------------------------------------- #
def split_canonical(canonical_records, val_fraction=0.15, test_fraction=0.15,
                    seed=42):
    """Stratified split over UNIQUE canonical records (one per orbit).

    Because we split the canonical records themselves -- and every raw mosaic
    maps to exactly one canonical record -- no orbit can straddle two splits.
    Stratification is per label, with a fixed seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    by_label = {}
    for key, label in canonical_records:
        by_label.setdefault(int(label), []).append(key)

    train_keys, val_keys, test_keys = [], [], []
    for label, keys in sorted(by_label.items()):
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        # Guarantee at least one held-out example per class when feasible.
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        n_test = min(n_test, n)
        n_val = min(n_val, n - n_test)
        test_keys += [(k, label) for k in keys[:n_test]]
        val_keys += [(k, label) for k in keys[n_test:n_test + n_val]]
        train_keys += [(k, label) for k in keys[n_test + n_val:]]

    rng.shuffle(train_keys)
    rng.shuffle(val_keys)
    rng.shuffle(test_keys)
    return train_keys, val_keys, test_keys


def _assert_no_leakage(train_keys, val_keys, test_keys):
    """Hard guarantee: canonical key sets are pairwise disjoint."""
    s_tr = {k for k, _ in train_keys}
    s_va = {k for k, _ in val_keys}
    s_te = {k for k, _ in test_keys}
    assert s_tr.isdisjoint(s_va), "leak: train/val share a canonical key"
    assert s_tr.isdisjoint(s_te), "leak: train/test share a canonical key"
    assert s_va.isdisjoint(s_te), "leak: val/test share a canonical key"


def make_leakage_safe_split(records, val_fraction=0.15, test_fraction=0.15,
                            seed=42, pad_to=None, verbose=True):
    """Full pipeline: dedup by canonical key, split, one-hot encode.

    records: list of (mosaic_str, label).
    Returns a SplitResult with one-hot X tensors (N, p, p, 11) and int y arrays,
    plus a meta dict (collapse stats, sizes, class balance, the canonical-key
    lists for auditing).
    """
    canonical_records, collapse_stats = collapse_by_canonical(records)
    train_keys, val_keys, test_keys = split_canonical(
        canonical_records, val_fraction=val_fraction,
        test_fraction=test_fraction, seed=seed,
    )
    _assert_no_leakage(train_keys, val_keys, test_keys)

    # Infer a common pad size across all splits so tensors stack.
    if pad_to is None:
        all_keys = [k for k, _ in canonical_records]
        pad_to = max(parse_mosaic(k).shape[0] for k in all_keys) if all_keys else 0

    def build(keyset):
        if not keyset:
            empty = np.zeros((0, pad_to, pad_to, 11), dtype=np.float32)
            return empty, np.zeros((0,), dtype=np.int64)
        mosaics = [k for k, _ in keyset]
        labels = np.array([lbl for _, lbl in keyset], dtype=np.int64)
        X = stack_onehot(mosaics, pad_to=pad_to)
        return X, labels

    X_train, y_train = build(train_keys)
    X_val, y_val = build(val_keys)
    X_test, y_test = build(test_keys)

    def balance(y):
        v, c = np.unique(y, return_counts=True)
        return {int(a): int(b) for a, b in zip(v, c)}

    meta = {
        "collapse": collapse_stats,
        "pad_to": int(pad_to),
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "sizes": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "class_balance": {
            "train": balance(y_train),
            "val": balance(y_val),
            "test": balance(y_test),
        },
        "keys": {
            "train": [k for k, _ in train_keys],
            "val": [k for k, _ in val_keys],
            "test": [k for k, _ in test_keys],
        },
    }
    if verbose:
        c = collapse_stats
        print(
            f"[dataset] {c['raw_rows']} raw rows -> {c['unique_canonical']} "
            f"unique canonical mosaics ({c['collapsed']} collapsed as "
            f"D4-equivalent; {c['label_conflicts']} label conflicts)."
        )
        print(
            f"[dataset] split (leakage-safe on canonical key): "
            f"train={meta['sizes']['train']}, val={meta['sizes']['val']}, "
            f"test={meta['sizes']['test']}, pad_to={pad_to}"
        )
    return SplitResult(X_train, y_train, X_val, y_val, X_test, y_test, meta)


def load_crossing_split(datasets_dir="datasets", dims=None, cap=CROSSING_CAP,
                        **kwargs):
    """Build a leakage-safe crossing-number split from the CSVs.

    The CSVs store the raw num_crossings integer (column "Num Crossings"); we
    bucket each label to its capped class (>=cap -> cap) BEFORE the canonical
    dedup, so the orbit-collapse and stratified split operate on the final
    multi-class labels. Crossing number is a D4 invariant, so collapsing the
    full 8-element orbit into one record is exactly correct here.

    dims: iterable of dimensions; None -> every dim_*.csv under crossing/.
    """
    label_col = TASK_LABELS["crossing"]["csv_col"]
    if dims is None:
        pattern = os.path.join(datasets_dir, "crossing", "dim_*.csv")
        records = load_csv_glob(pattern, label_col=label_col)
    else:
        records = []
        for d in dims:
            path = os.path.join(datasets_dir, "crossing", f"dim_{d}.csv")
            records.extend(load_csv_records(path, label_col=label_col))
    records = [(m, bucket_crossings(lbl, cap=cap)) for (m, lbl) in records]
    return make_leakage_safe_split(records, **kwargs)


def load_unknot_split(datasets_dir="datasets", dims=None, **kwargs):
    """Convenience: build a leakage-safe unknot-detection split from the CSVs.

    dims: iterable of dimensions (e.g. [3, 4, 5]); None -> every dim_*.csv.
    """
    label_col = TASK_LABELS["unknot"]["csv_col"]
    if dims is None:
        pattern = os.path.join(datasets_dir, "unknot", "dim_*.csv")
        records = load_csv_glob(pattern, label_col=label_col)
    else:
        records = []
        for d in dims:
            path = os.path.join(datasets_dir, "unknot", f"dim_{d}.csv")
            records.extend(load_csv_records(path, label_col=label_col))
    return make_leakage_safe_split(records, **kwargs)
