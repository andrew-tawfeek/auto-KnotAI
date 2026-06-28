"""Unit tests proving the D4 canonical form is a correct orbit representative.

Run directly:  python3 harness/test_canonical.py
(Also discoverable by unittest / pytest.)

Proves:
  (a) all 8 D4 symmetries of a mosaic map to the SAME canonical key;
  (b) two genuinely different mosaics map to DIFFERENT canonical keys;
  (c) the relabel is an involution-consistent group action (round-trips);
  (d) canonical_form is idempotent and symmetry-invariant on real data.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.canonical import (  # noqa: E402
    apply_symmetry,
    canonical_form,
    canonical_key,
    orbit,
    _D4_OPS,
)
from harness.mosaic_io import mosaic_to_str, parse_mosaic  # noqa: E402

# A few real suitably-connected mosaics taken from datasets/unknot/dim_*.csv.
SAMPLES = [
    "[[0, 2, 1], [0, 6, 6], [0, 3, 4]]",
    "[[0, 2, 1], [2, 10, 4], [3, 4, 0]]",
    "[[2, 1, 0], [3, 5, 1], [0, 3, 4]]",
    "[[0, 0, 0, 0, 0], [0, 0, 2, 1, 0], [0, 2, 10, 9, 1], "
    "[2, 4, 3, 10, 4], [3, 5, 5, 4, 0]]",
    "[[2, 1, 2, 1, 0], [6, 3, 4, 6, 0], [6, 2, 1, 3, 1], "
    "[6, 6, 6, 0, 6], [3, 4, 3, 5, 4]]",
]


class TestD4Canonical(unittest.TestCase):
    def test_all_eight_symmetries_share_one_key(self):
        """(a) Every D4 image of a mosaic has the same canonical key."""
        for s in SAMPLES:
            key = canonical_key(s)
            for op in range(len(_D4_OPS)):
                img = apply_symmetry(parse_mosaic(s), op)
                self.assertEqual(
                    canonical_key(img),
                    key,
                    msg=f"symmetry {op} of {s} produced a different canonical key",
                )

    def test_different_mosaics_different_keys(self):
        """(b) Genuinely different mosaics get different canonical keys."""
        keys = [canonical_key(s) for s in SAMPLES]
        self.assertEqual(
            len(set(keys)), len(keys),
            msg=f"distinct sample mosaics collided to keys: {keys}",
        )

    def test_canonical_is_idempotent(self):
        """(d) Canonicalizing an already-canonical mosaic is a no-op."""
        for s in SAMPLES:
            c = canonical_form(s)
            self.assertEqual(mosaic_to_str(c), canonical_key(s))
            self.assertEqual(mosaic_to_str(canonical_form(c)), canonical_key(s))

    def test_canonical_invariant_over_orbit(self):
        """canonical_form returns the exact same ARRAY for every orbit member."""
        for s in SAMPLES:
            cstr = canonical_key(s)
            for img in orbit(s):
                self.assertEqual(canonical_key(img), cstr)

    def test_group_action_round_trips(self):
        """Applying a symmetry then its inverse recovers the original mosaic.

        rot90 and rot270 are inverses; every reflection / 180 is its own
        inverse; identity is its own inverse. This checks the *coupled*
        geometry+relabel action really is a group action (not just geometry).
        """
        names = [n for (n, _f, _r) in _D4_OPS]
        idx = {n: i for i, n in enumerate(names)}
        inverse = {
            "identity": "identity",
            "rot90": "rot270",
            "rot270": "rot90",
            "rot180": "rot180",
            "flip_v": "flip_v",
            "flip_h": "flip_h",
            "transpose": "transpose",
            "anti_transpose": "anti_transpose",
        }
        for s in SAMPLES:
            arr = parse_mosaic(s)
            for name in names:
                fwd = apply_symmetry(arr, idx[name])
                back = apply_symmetry(fwd, idx[inverse[name]])
                self.assertTrue(
                    np.array_equal(back, arr),
                    msg=f"{name} then {inverse[name]} did not round-trip on {s}",
                )

    def test_orbit_size_divides_eight(self):
        """Each distinct-image count in an orbit must divide |D4| = 8."""
        for s in SAMPLES:
            distinct = {mosaic_to_str(a) for a in orbit(s)}
            self.assertEqual(8 % len(distinct), 0,
                             msg=f"orbit size {len(distinct)} does not divide 8")

    def test_relabel_preserves_validity_count(self):
        """A symmetry maps crossings to crossings (count is invariant)."""
        for s in SAMPLES:
            arr = parse_mosaic(s)
            base = int(np.sum((arr == 9) | (arr == 10)))
            for op in range(len(_D4_OPS)):
                img = apply_symmetry(arr, op)
                self.assertEqual(
                    int(np.sum((img == 9) | (img == 10))), base,
                    msg=f"symmetry {op} changed the crossing count of {s}",
                )


    def test_symmetry_preserves_suitable_connectivity(self):
        """The strongest geometric check: a D4 image of a suitably-connected
        mosaic is itself suitably-connected (and vice versa). This only holds
        if the tile relabel correctly tracks strand directions through the
        geometry -- so it validates the relabel end-to-end against mosaics.py.
        """
        from mosaics import Mosaic  # local import; needs the repo on sys.path

        for s in SAMPLES:
            arr = parse_mosaic(s)
            base = Mosaic(arr.tolist()).isSuitablyConnected()
            for op in range(len(_D4_OPS)):
                img = apply_symmetry(arr, op)
                self.assertEqual(
                    Mosaic(img.tolist()).isSuitablyConnected(), base,
                    msg=f"symmetry {op} changed suitable-connectivity of {s}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
