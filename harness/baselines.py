"""Honest non-CNN baselines, so any CNN accuracy is measured as real lift.

For unknot detection on knot mosaics we provide:

  * majority_class      -- always predict the most common training label.
  * crossing_zero_rule  -- a domain heuristic: a mosaic with 0 crossings is an
                           unknot (a diagram with no crossings cannot be knotted).
                           Predicts is_unknot = (num_crossings == 0). This is the
                           classic "is it obviously trivial" rule.
  * logreg_handcounts   -- logistic regression on cheap hand-counted features
                           (per-tile-type counts + crossing count + dimension).

Each returns accuracy AND balanced accuracy (mean of per-class recall), which is
the honest number on imbalanced data. The CNN must beat these to be interesting.
"""

from __future__ import annotations

import numpy as np

from .mosaic_io import onehot_to_mosaic

CROSSING_TILES = (9, 10)


def _balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for c in np.unique(y_true):
        mask = y_true == c
        if mask.any():
            recalls.append(float(np.mean(y_pred[mask] == c)))
    return float(np.mean(recalls)) if recalls else 0.0


def _accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def _confusion(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _report(name, y_true, y_pred):
    return {
        "name": name,
        "accuracy": _accuracy(y_true, y_pred),
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred),
        "confusion": _confusion(y_true, y_pred),
    }


def _num_crossings_from_onehot(X):
    """Count crossing tiles (9, 10) in each one-hot mosaic. X: (N,n,n,11)."""
    grids = np.argmax(X, axis=-1)  # (N, n, n)
    return np.sum((grids == 9) | (grids == 10), axis=(1, 2))


def majority_class(y_train, y_test):
    vals, counts = np.unique(y_train, return_counts=True)
    pred_label = int(vals[np.argmax(counts)])
    y_pred = np.full(len(y_test), pred_label, dtype=np.int64)
    rep = _report("majority_class", y_test, y_pred)
    rep["predicted_label"] = pred_label
    return rep


def crossing_zero_rule(X_test, y_test):
    """Predict is_unknot = (num_crossings == 0)."""
    nc = _num_crossings_from_onehot(X_test)
    y_pred = (nc == 0).astype(np.int64)
    return _report("crossing_zero_rule", y_test, y_pred)


def _handcount_features(X):
    """Cheap per-mosaic features: tile-type histogram (11) + crossings + dim."""
    grids = np.argmax(X, axis=-1)  # (N, n, n)
    N = grids.shape[0]
    n = grids.shape[1]
    hist = np.zeros((N, 11), dtype=np.float32)
    for t in range(11):
        hist[:, t] = np.sum(grids == t, axis=(1, 2))
    crossings = (hist[:, 9] + hist[:, 10]).reshape(-1, 1)
    dim = np.full((N, 1), float(n), dtype=np.float32)
    return np.concatenate([hist, crossings, dim], axis=1)


def logreg_handcounts(X_train, y_train, X_test, y_test, seed=42):
    """Logistic regression on hand-counted features (needs scikit-learn)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"name": "logreg_handcounts", "skipped": "scikit-learn not installed"}

    Ftr = _handcount_features(X_train)
    Fte = _handcount_features(X_test)
    scaler = StandardScaler().fit(Ftr)
    Ftr, Fte = scaler.transform(Ftr), scaler.transform(Fte)
    if len(np.unique(y_train)) < 2:
        return {"name": "logreg_handcounts", "skipped": "only one class in train"}
    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=seed)
    clf.fit(Ftr, y_train)
    y_pred = clf.predict(Fte)
    return _report("logreg_handcounts", y_test, y_pred)


def run_all_baselines(X_train, y_train, X_test, y_test, seed=42):
    """Run every baseline and return a list of report dicts."""
    return [
        majority_class(y_train, y_test),
        crossing_zero_rule(X_test, y_test),
        logreg_handcounts(X_train, y_train, X_test, y_test, seed=seed),
    ]
