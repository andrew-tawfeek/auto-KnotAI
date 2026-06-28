# `harness/` — leakage-safe training harness for the knot-invariant CNN

A research-quality harness for training CNNs on knot mosaics where **the
correctness of the train/test split matters more than model performance**. The
headline feature is a rigorous, symmetry-aware (D4) canonical dedup that
prevents the single most common way these results go fake: symmetry-equivalent
mosaics leaking across the train/test boundary.

## Why the canonical-dedup split matters (read this first)

A knot mosaic is an `n×n` grid of tile types `0..10`. Two mosaics related by a
symmetry of the square grid — a rotation or a reflection — are the **same drawn
knot**, just redrawn. There are 8 such symmetries: the **dihedral group D4**
(4 rotations × 2 reflections).

If a naive split puts mosaic *M* in the training set and a rotated copy of *M*
in the test set, the model can **memorize *M* during training and "recognize"
its rotation at test time**. The reported test accuracy is then inflated — it
measures memorization, not generalization. On knot-mosaic data, where the
generator emits many symmetry-related grids, this leakage is large and silent.

The fix, implemented here:

1. Compute a **canonical D4 representative** of every mosaic (`canonical.py`).
2. **Collapse** all mosaics sharing a canonical key into one record.
3. **Split on the canonical keys**, so an entire D4 orbit lands in exactly one
   split — never straddling train/val/test. A hard assertion
   (`_assert_no_leakage`) verifies the three key sets are pairwise disjoint.

On the committed sample CSVs this collapses hundreds of equivalent mosaics
(e.g. **386** D4-equivalents on `dim_3..7`), so without it those rows would have
leaked.

## The D4 canonical form (`canonical.py`) — the key rigor piece

`canonical_form(mosaic)` returns the lexicographically-smallest serialization
over the mosaic's 8-element D4 orbit; `canonical_key(mosaic)` returns that
string (the dedup key). Each of the 8 symmetries acts in **two coupled ways**:

- **Geometry** — it transposes/flips the `n×n` cell grid (`np.rot90`, `np.flipud`,
  `np.fliplr`, `.T`).
- **Tile relabel** — a tile encodes local strand *directions*
  (`up/down/left/right`), so rotating/reflecting the picture permutes those
  directions, which permutes the 11 tile types.

**The tile relabel is derived, not hand-typed.** For single-strand tiles
(`0..6`) the new tile is found by applying the symmetry's direction permutation
to the tile's connection-direction *set*. For the two arc-pair tiles (`7`, `8`)
we track the *strand pairing* (which pairs of corners are joined), which is what
distinguishes them. For the two crossings (`9`, `10`) — which differ only in
which strand passes *over* — the relabel swaps `9 ↔ 10` exactly when the
symmetry swaps the horizontal and vertical axes (the diagonal reflections and
the 90°/270° rotations), and leaves them fixed otherwise. This is derived from
the over/under semantics in `mosaics.py`.

The relabel is validated end-to-end against `mosaics.py`: `test_canonical.py`
proves that every D4 image of a **suitably-connected** mosaic is itself
suitably-connected, and that crossing counts are preserved — i.e. the relabel
genuinely respects strand connectivity, not just grid geometry. **There is no
unresolved tile-relabel limitation.**

### Chirality note (honest caveat)

The reflection symmetries take a knot to its **mirror image** (e.g. left vs
right trefoil), which is a genuinely different knot type for chiral knots. We
still collapse the full 8-element D4 orbit, because:

- For **leakage-safety** we *want* every grid-symmetric redraw — including
  mirrors — in the same split; that is the conservative, correct choice.
- For invariants that are mirror-insensitive (unknot detection, crossing
  number, suitable-connectivity, component count) the mirror is the same label,
  so collapsing is exactly right.

If you later study a **chirality-sensitive** target (e.g. signature, or knot
type with handedness), restrict the orbit to the 4 *rotations* only (the cyclic
subgroup `C4`) — the relabel arrays for the rotation ops are already computed in
`canonical.py` and can be sliced out. This is a one-line change to the op list.

## Modules

| file | purpose |
|------|---------|
| `mosaic_io.py` | parse stringified mosaic → `(n,n)` int array; one-hot → `(n,n,11)`; zero-pad variable `n`; serialize. Reuses `mosaics.py`. |
| `canonical.py` | **D4 canonical form** for symmetry-aware dedup (the key rigor piece). |
| `dataset.py`   | load CSV now / SQLite-ready by the documented `mosaics.db` schema; collapse by canonical key; **leakage-safe** stratified split; return one-hot tensors. |
| `baselines.py` | honest non-CNN baselines: majority-class, `num_crossings==0` rule, logistic regression on hand counts. |
| `train_cnn.py` | small CNN (adapted from `cnn_train.py`) for unknot detection on the leakage-safe split; metrics vs baselines → `results/*.json`. |
| `test_canonical.py` | unit tests proving the D4 canonical form is a correct orbit representative. |

## Usage

```bash
# Run the canonical-form unit tests (no GPU needed)
python3 harness/test_canonical.py

# Build a leakage-safe split + baselines (numpy/pandas/sklearn only)
python3 -c "from harness.dataset import load_unknot_split; \
from harness.baselines import run_all_baselines; \
s=load_unknot_split(dims=[3,4,5]); \
print(run_all_baselines(s.X_train,s.y_train,s.X_test,s.y_test))"

# Train the CNN (needs TensorFlow); small + CPU-runnable on the sample CSVs
python3 -m harness.train_cnn --dims 3 4 5 6 7 --epochs 12
```

### Scaling to the full DB / GPU later

`dataset.load_sqlite_records(db_path, task="unknot", dimension=…)` is written to
the `mosaics.db` schema (`mosaic, dimension, is_suitably_connected,
num_crossings, has_crossing, num_components, is_unknot, pd_code`) and is
import-safe — it only touches `sqlite3` when called with a real DB path. The CNN
is parameterized (`conv_filters`, `dense_units`, `epochs`, `batch_size`) and
uses `GlobalAveragePooling`, so the same model spans padded mosaics of mixed `n`
and moves to the L4 + full DB with only flag changes.

## What "honest baselines" buys you

The CNN's reported number is only meaningful as **lift** over a non-CNN
baseline. We report majority-class (the floor), a crossing-count rule (the
obvious domain heuristic), and a logistic regression on hand-counted tile
features. A CNN that doesn't beat logistic-regression-on-counts isn't learning
anything a histogram can't.
