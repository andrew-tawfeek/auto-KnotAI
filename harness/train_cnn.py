"""Small CNN for binary unknot detection on the leakage-safe split.

Adapted from the repo's cnn_train.py, but trained on the canonical-key split
from harness.dataset (so no D4-equivalent mosaic leaks across train/test) and
reported against the honest baselines in harness.baselines.

Designed to be CPU-runnable on the small CSV data for fast testing, and
parameterized (conv_filters, dense_units, epochs, batch_size) so the same code
scales to the L4 GPU + full mosaics.db later. Metrics are written to results/
as JSON.

Run:
    python3 -m harness.train_cnn --dims 3 4 5 --epochs 8
    python3 -m harness.train_cnn            # all dims, default epochs
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

from .baselines import run_all_baselines
from .dataset import load_unknot_split

_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def build_model(input_shape, conv_filters=(32, 64, 64, 32), dense_units=0,
                seed=42):
    """Build the small CNN. Returns a compiled tf.keras model.

    Kept architecturally close to cnn_train.py: stacked 3x3 same-padding convs,
    global average pooling, sigmoid head. GlobalAveragePooling makes the model
    dimension-agnostic, which is what lets one model span padded mosaics of
    mixed n (and later the full DB's many dimensions).
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.layers.Input(shape=input_shape)]
    for f in conv_filters:
        layers.append(
            tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")
        )
    layers.append(tf.keras.layers.GlobalAveragePooling2D())
    if dense_units:
        layers.append(tf.keras.layers.Dense(dense_units, activation="relu"))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    bal = (recall + spec) / 2
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "precision": float(prec),
        "recall": float(recall),
        "specificity": float(spec),
        "f1": float(f1),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def train(dims=None, epochs=8, batch_size=32, seed=42,
          conv_filters=(32, 64, 64, 32), dense_units=0, patience=5,
          datasets_dir="datasets", results_dir=_RESULTS_DIR, verbose=1):
    """Train + evaluate the CNN on the leakage-safe unknot split.

    Returns the result dict (also written to results/<timestamp>_unknot_cnn.json).
    Requires TensorFlow; raises a clear error if it's missing.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "harness.train_cnn requires TensorFlow. Install tensorflow-cpu "
            "(or tensorflow) to run; the rest of the harness runs without it."
        ) from exc

    split = load_unknot_split(datasets_dir=datasets_dir, dims=dims, seed=seed)
    Xtr, ytr, Xva, yva, Xte, yte = split.as_tuple()

    # Class weights so an imbalanced unknot/knot ratio does not just train the
    # model into the majority class (honesty matches the balanced baselines).
    classes, counts = np.unique(ytr, return_counts=True)
    total = counts.sum()
    class_weight = {int(c): float(total / (len(classes) * n))
                    for c, n in zip(classes, counts)}

    model = build_model(Xtr.shape[1:], conv_filters=conv_filters,
                        dense_units=dense_units, seed=seed)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva) if len(yva) else None,
        epochs=epochs, batch_size=batch_size, callbacks=callbacks,
        class_weight=class_weight, verbose=verbose,
    )

    probs = model.predict(Xte, verbose=0).reshape(-1)
    y_pred = (probs >= 0.5).astype(np.int64)
    cnn_metrics = _metrics(yte, y_pred)
    baselines = run_all_baselines(Xtr, ytr, Xte, yte, seed=seed)

    best_baseline_bal = max(
        (b["balanced_accuracy"] for b in baselines
         if "balanced_accuracy" in b),
        default=0.0,
    )
    result = {
        "task": "unknot_detection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dims": list(dims) if dims else "all",
        "split_meta": {k: v for k, v in split.meta.items() if k != "keys"},
        "hyperparams": {
            "epochs": epochs,
            "trained_epochs": len(history.history["loss"]),
            "batch_size": batch_size,
            "seed": seed,
            "conv_filters": list(conv_filters),
            "dense_units": dense_units,
            "patience": patience,
            "class_weight": class_weight,
        },
        "cnn": cnn_metrics,
        "baselines": baselines,
        "cnn_balanced_accuracy_lift_over_best_baseline": float(
            cnn_metrics["balanced_accuracy"] - best_baseline_bal
        ),
    }

    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(results_dir, f"{stamp}_unknot_cnn.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["results_path"] = out_path

    if verbose:
        print("\n=== unknot-detection (leakage-safe canonical split) ===")
        print(f"CNN: acc={cnn_metrics['accuracy']:.3f} "
              f"balanced_acc={cnn_metrics['balanced_accuracy']:.3f} "
              f"f1={cnn_metrics['f1']:.3f}")
        for b in baselines:
            if "balanced_accuracy" in b:
                print(f"  baseline {b['name']:18s} "
                      f"acc={b['accuracy']:.3f} "
                      f"balanced_acc={b['balanced_accuracy']:.3f}")
            else:
                print(f"  baseline {b['name']:18s} skipped: "
                      f"{b.get('skipped')}")
        print(f"CNN balanced-acc lift over best baseline: "
              f"{result['cnn_balanced_accuracy_lift_over_best_baseline']:+.3f}")
        print(f"saved: {out_path}")
    return result


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", type=int, nargs="*", default=None,
                   help="mosaic dimensions to include (default: all CSVs)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conv-filters", type=int, nargs="*",
                   default=[32, 64, 64, 32])
    p.add_argument("--dense-units", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--datasets-dir", type=str, default="datasets")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        dims=args.dims, epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, conv_filters=tuple(args.conv_filters),
        dense_units=args.dense_units, patience=args.patience,
        datasets_dir=args.datasets_dir,
    )
