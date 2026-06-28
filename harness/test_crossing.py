"""Unit tests for the crossing-number task path (no TensorFlow needed).

Run directly:  python3 harness/test_crossing.py

Proves:
  (a) bucket_crossings caps the sparse tail correctly;
  (b) the leakage-safe split carries the bucketed multi-class labels and stays
      orbit-disjoint across train/val/test;
  (c) the crossing-count baseline is an oracle (reads num_crossings off the
      grid) while majority-class sits at the 1/num_classes floor -- the honesty
      property the result JSON relies on.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.dataset import (  # noqa: E402
    CROSSING_CAP,
    bucket_crossings,
    make_leakage_safe_split,
)
from harness.baselines import run_all_crossing_baselines  # noqa: E402


def _synthetic_records(n_per_class=120, seed=1):
    """Build diverse 5x5 mosaics whose true crossing count = #tiles in {9,10}."""
    rng = random.Random(seed)
    recs = []
    for nc in range(0, CROSSING_CAP + 3):  # include a sparse tail -> 6+ bucket
        for _ in range(n_per_class):
            cells = [(i, j) for i in range(5) for j in range(5)]
            rng.shuffle(cells)
            g = [[0] * 5 for _ in range(5)]
            placed = 0
            for (i, j) in cells:
                if placed < nc:
                    g[i][j] = rng.choice([9, 10])
                    placed += 1
                else:
                    g[i][j] = rng.choice([1, 2, 3, 4, 5, 6, 7, 8])
            recs.append((str(g), bucket_crossings(nc)))
    return recs


class TestCrossingTask(unittest.TestCase):
    def test_bucketing(self):
        self.assertEqual([bucket_crossings(i) for i in range(9)],
                         [0, 1, 2, 3, 4, 5, 6, 6, 6])

    def test_split_labels_and_no_leakage(self):
        recs = _synthetic_records()
        split = make_leakage_safe_split(recs, seed=42, verbose=False)
        labels = set(split.y_train.tolist()) | set(split.y_test.tolist())
        self.assertTrue(labels.issubset(set(range(CROSSING_CAP + 1))))
        self.assertIn(CROSSING_CAP, labels)  # the 6+ bucket is present
        tr = set(split.meta["keys"]["train"])
        te = set(split.meta["keys"]["test"])
        va = set(split.meta["keys"]["val"])
        self.assertTrue(tr.isdisjoint(te) and tr.isdisjoint(va)
                        and te.isdisjoint(va))

    def test_oracle_vs_majority(self):
        recs = _synthetic_records()
        s = make_leakage_safe_split(recs, seed=42, verbose=False)
        classes = list(range(CROSSING_CAP + 1))
        bl = {b["name"]: b for b in run_all_crossing_baselines(
            s.X_train, s.y_train, s.X_test, s.y_test, classes, CROSSING_CAP)}
        # crossing-count rule reads the label off the grid -> ~perfect.
        self.assertGreater(bl["crossing_count_rule"]["balanced_accuracy"], 0.99)
        # majority-class is at / below the 1/num_classes chance floor.
        self.assertLessEqual(bl["majority_class"]["balanced_accuracy"],
                             1.0 / len(classes) + 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
