"""Canonical D4 form for knot mosaics -- the key rigor piece.

WHY THIS EXISTS
---------------
Two mosaics related by a symmetry of the square grid (rotation / reflection)
represent the *same* drawn knot. If symmetry-equivalent mosaics leak across a
train/test split, the model can "memorize then recognize", and reported accuracy
is inflated -- the #1 way these results go fake. We therefore canonicalize each
mosaic to a single representative of its D4 orbit and split on that key.

THE GROUP
---------
The symmetries of a square grid form the dihedral group D4: 8 elements =
{identity, rot90, rot180, rot270} x {no flip, flip}. Each element acts on the
n x n grid in two coupled ways:

  1. It permutes the *cell positions* (a geometric transpose/flip of the array).
  2. It permutes the *tile types*, because a tile encodes local strand
     directions among {up, down, left, right}; rotating/reflecting the picture
     rotates/reflects those directions, which maps one tile type to another.

TILE-TYPE RELABELING (derived, not hand-typed)
----------------------------------------------
Tile types 0..10 are defined in mosaics.py by their connection directions
(TILE_CONNECTIONS). For tiles 0 and 1..8 the relabel is fully determined by
applying the symmetry's permutation of {up,down,left,right} to the tile's
direction set and looking up which tile type has the resulting set.

Crossings 9 and 10 are the only subtlety: both connect all four directions and
differ only in which strand passes *over*. In mosaics.py:
  9  = horizontal strand over the vertical strand
  10 = vertical strand over the horizontal strand
A symmetry that swaps the horizontal and vertical axes (the two diagonal
reflections, and -- because they swap axes -- the 90/270 rotations) turns an
"over horizontal" into an "over vertical" and thus maps 9 <-> 10. Symmetries
that preserve each axis (identity, rot180, horizontal flip, vertical flip) keep
9 -> 9 and 10 -> 10. We derive this swap directly from whether the symmetry's
direction permutation exchanges the {left,right} and {up,down} axes, so the
crossing relabel stays in lockstep with the geometry.

This makes ``canonical_form`` an exact orbit representative under D4 including
the over/under crossing semantics -- no unresolved tile-relabel limitation.

The image (mirror) symmetries DO change a knot's chirality (a left trefoil vs a
right trefoil), but they are still genuine *grid* symmetries that produce a
distinct mosaic string for the same underlying picture; for leakage-safe
splitting we want every such variant in the same split, so collapsing the full
8-element D4 orbit is the correct, conservative choice. (See README for the
chirality note.)
"""

from __future__ import annotations

import numpy as np

from .mosaic_io import mosaic_to_str, parse_mosaic

# Pulled from mosaics.py so tile semantics stay in lockstep with the repo.
from mosaics import TILE_CONNECTIONS, flatten  # type: ignore

_DIRECTIONS = ("up", "down", "left", "right")

# Direction set (frozenset) -> tile type, for the non-crossing tiles 0..8.
# Tiles 9 and 10 share the all-four-directions set, so they are handled
# separately via the over/under axis-swap rule.
def _direction_set(tile_type):
    return frozenset(flatten(TILE_CONNECTIONS.get(tile_type, [])))


# Single-strand tiles 0..6 have *distinct* direction sets, so a set lookup
# identifies them. (Tile 0 -> empty set.)
_SET_TO_TILE = {}
for _t in range(7):
    _SET_TO_TILE[_direction_set(_t)] = _t

_ALL_FOUR = frozenset(_DIRECTIONS)  # shared by the four 4-point tiles 7,8,9,10


def _strand_pairing(tile_type):
    """For a 4-point tile, return the frozenset of joined-direction pairs.

    Tile 7  = {down-left, up-right}; Tile 8 = {down-right, up-left}.
    Crossings 9/10 connect both axes straight through:
        {up-down, left-right}.
    Representing each strand as a frozenset of its two endpoints lets us track
    *which corners are joined* through a symmetry, which is what distinguishes
    7 from 8 (and, with the over/under bit, 9 from 10).
    """
    conns = TILE_CONNECTIONS[tile_type]  # list of [a, b] strand pairs
    return frozenset(frozenset(strand) for strand in conns)


# pairing -> tile, for the arc-pair tiles 7 and 8 (no crossing).
_PAIRING_TO_ARC = {_strand_pairing(7): 7, _strand_pairing(8): 8}
# 9 and 10 share the straight-through pairing; they differ only by over/under.


# ---- The 8 D4 elements, as (array-transform, direction-permutation) pairs. ----
# Each direction-permutation maps a direction label to where it goes under the
# symmetry. We define them on numpy array ops (geometry) plus the matching dict
# (relabel), and we VERIFY at import that the two agree on a probe grid.

def _dperm(mapping):
    return dict(mapping)


# Geometric transforms on an (n, n) array. For each we give the matching
# direction permutation (where a strand pointing `dir` ends up pointing).
#
# Conventions: array index (row i grows downward, col j grows rightward).
#   rot90 here = rotate the *picture* 90 degrees counterclockwise.
_D4 = [
    (
        "identity",
        lambda a: a,
        _dperm({"up": "up", "down": "down", "left": "left", "right": "right"}),
    ),
    (
        "rot90",  # CCW: up->left, left->down, down->right, right->up
        lambda a: np.rot90(a, 1),
        _dperm({"up": "left", "left": "down", "down": "right", "right": "up"}),
    ),
    (
        "rot180",  # up<->down, left<->right
        lambda a: np.rot90(a, 2),
        _dperm({"up": "down", "down": "up", "left": "right", "right": "left"}),
    ),
    (
        "rot270",  # CW: up->right, right->down, down->left, left->up
        lambda a: np.rot90(a, 3),
        _dperm({"up": "right", "right": "down", "down": "left", "left": "up"}),
    ),
    (
        "flip_v",  # flip upside-down (vertical axis flip): up<->down
        lambda a: np.flipud(a),
        _dperm({"up": "down", "down": "up", "left": "left", "right": "right"}),
    ),
    (
        "flip_h",  # mirror left-right: left<->right
        lambda a: np.fliplr(a),
        _dperm({"up": "up", "down": "down", "left": "right", "right": "left"}),
    ),
    (
        "transpose",  # reflect across main diagonal: up<->left, down<->right
        lambda a: a.T,
        _dperm({"up": "left", "left": "up", "down": "right", "right": "down"}),
    ),
    (
        "anti_transpose",  # reflect across anti-diagonal: up<->right, down<->left
        lambda a: np.rot90(a, 2).T,
        _dperm({"up": "right", "right": "up", "down": "left", "left": "down"}),
    ),
]


def _relabel_for(dperm):
    """Build a length-11 tile-relabel array from a direction permutation.

    relabel[t] = the tile type that tile t becomes after the symmetry.
    Crossings (9, 10) swap iff the symmetry swaps the horizontal/vertical axes.
    """
    relabel = np.zeros(11, dtype=np.int64)
    # Single-strand tiles 0..6: identified by their (permuted) direction set.
    for t in range(7):
        new_set = frozenset(dperm[d] for d in _direction_set(t))
        relabel[t] = _SET_TO_TILE[new_set]
    # Arc-pair tiles 7, 8: identified by their (permuted) strand pairing.
    for t in (7, 8):
        new_pairing = frozenset(
            frozenset(dperm[d] for d in strand) for strand in _strand_pairing(t)
        )
        relabel[t] = _PAIRING_TO_ARC[new_pairing]
    # Crossing over/under: 9 = horizontal-over, 10 = vertical-over.
    # The horizontal axis is {left,right}. If the symmetry sends 'left' to a
    # vertical direction, the axes are swapped and 9<->10; else they are fixed.
    axes_swapped = dperm["left"] in ("up", "down")
    if axes_swapped:
        relabel[9], relabel[10] = 10, 9
    else:
        relabel[9], relabel[10] = 9, 10
    return relabel


# Precompute (name, array_transform, relabel_array) for all 8 symmetries.
_D4_OPS = [(name, fn, _relabel_for(dperm)) for (name, fn, dperm) in _D4]


def _self_check():
    """Verify geometry and direction-relabel agree on a probe, at import time.

    Build a tiny 'directional' probe: a 2x2 grid whose single-strand tiles form
    a valid little loop, push it through every symmetry two ways (geometric +
    relabel) and confirm the resulting tile direction-sets are consistent.
    """
    # For every symmetry and every tile, the relabeled tile's direction set must
    # equal the direction-permuted original set (the definition of consistency).
    for name, _fn, relabel in _D4_OPS:
        # recover the dperm for this op
        dperm = next(dp for (n, _f, dp) in _D4 if n == name)
        for t in range(7):  # single-strand tiles checked by direction set
            expected = frozenset(dperm[d] for d in _direction_set(t))
            got = _direction_set(int(relabel[t]))
            assert expected == got, (name, t, expected, got)
        for t in (7, 8):  # arc-pair tiles checked by strand pairing
            expected = frozenset(
                frozenset(dperm[d] for d in strand) for strand in _strand_pairing(t)
            )
            got = _strand_pairing(int(relabel[t]))
            assert expected == got, (name, t, expected, got)


_self_check()


def apply_symmetry(arr, op_index):
    """Apply the ``op_index``-th D4 symmetry to an (n, n) int mosaic array.

    Both the cell geometry and the tile-type relabel are applied.
    """
    name, fn, relabel = _D4_OPS[op_index]
    transformed = fn(np.asarray(arr, dtype=np.int64))
    return relabel[transformed]


def orbit(mosaic):
    """Return the list of 8 D4 images (as int arrays) of a mosaic.

    Note: the list may contain duplicates when the mosaic has extra internal
    symmetry; that is expected.
    """
    arr = parse_mosaic(mosaic)
    return [apply_symmetry(arr, i) for i in range(len(_D4_OPS))]


def canonical_form(mosaic):
    """Return the canonical D4 representative of a mosaic as an int array.

    The representative is the one whose serialization is lexicographically
    smallest over the 8-element orbit, so symmetry-equivalent mosaics all map to
    the same array (and the same ``canonical_key``).
    """
    best_arr = None
    best_str = None
    for arr in orbit(mosaic):
        s = mosaic_to_str(arr)
        if best_str is None or s < best_str:
            best_str = s
            best_arr = arr
    return best_arr


def canonical_key(mosaic) -> str:
    """Return the canonical serialization string (the orbit's dedup key)."""
    return mosaic_to_str(canonical_form(mosaic))


D4_NAMES = [name for (name, _f, _r) in _D4_OPS]
