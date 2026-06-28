# Knot-Invariant CNN — Autonomous Research Project Plan

A CNN-based study of how well neural nets can recognize **knot invariants** from
**knot-mosaic** grids — built on Andrew's `KnotAI` repo, run as an autonomous project.

## 1. What already exists in KnotAI (analysis of all 4 branches)
- **`mosaics.py`** — core library: `Matrix`/`Tile`/`Mosaic`; mosaic = n×n grid, entries 0–10
  (11 tile types). Methods: `isSuitablyConnected`, `findCrossings`, `numCrossings`, `walk`
  (PD-code traversal), `shift`, `zoom`. Random generation + knot checks.
- **`cnn_train.py`** (concurrency/tensorflow branches, 460 lines) — a WORKING pipeline:
  CSV → one-hot (n×n×11) → stratified split → TF CNN → unknot-detection (binary), with
  accuracy/balanced-acc/confusion + training plots. **This is our baseline.**
- **Datasets** — CSVs `unknot/dim_3..14`, `trefoil/dim_4`, `connected/dim_3` (label e.g.
  `Is Unknot`); dim_10–14 have thousands of rows.
- **Big data release** — GH release `data-20260525-214315` on `andrew-tawfeek/KnotAI`:
  a **3.2M-mosaic SQLite DB** (3.4GB raw / 401MB zst), per-dimension, labeled by
  suitably-connected / has-crossings / component-count / unknot-vs-knotted. Built by the
  "high-compute publish daemon" (the `high-compute` GCE box, currently terminated).
- **`concurrency`** branch = most complete: + `multithreading-data-gen.py`, `.deploy/`
  publish+watch daemons, data-release manifest, SQLite. (`pytorch` branch is an abandoned stub.)
- Stack: TensorFlow 2.16.2, numpy/pandas/matplotlib.

## 2. Scientific goal & questions
Train CNNs to recognize knot invariants from the raw mosaic grid, and measure **which
invariants are learnable** and **how that scales** with mosaic dimension and data size.
- Which invariants does a CNN recover from the grid, and which resist it?
- Accuracy vs mosaic dimension (3→14+) and vs training-set size (scaling curves).
- Cross-dimension generalization (train small grids → test larger?).
- Architecture: baseline CNN vs deeper/ResNet vs **D4-equivariant** CNN (mosaics have
  dihedral symmetry — equivariance should help and is a clean research angle).

## 3. Invariant targets (escalating difficulty)
1. **Unknot detection** (binary) — baseline; data ready.
2. **Connectivity / component count / has-crossings** — already labeled in the DB.
3. **Crossing number** (ordinal/regression) — via `mosaics.py:numCrossings`.
4. **Knot-type classification** (unknot/trefoil/figure-8/…) — labels via PD-code oracle.
5. **Polynomial invariants** (Jones, determinant, signature) — PD-code → SnapPy/Sage oracle.
   The frontier: can a CNN learn a "deep" algebraic invariant from a picture of the knot?

## 4. Rigor (carry the lessons from the evolutionary-RSI project)
- **Symmetry-aware dedup to prevent train/test leakage.** Mosaics related by the dihedral
  group / planar isotopy are the "same" knot; naive splits leak. Canonicalize each mosaic
  (orbit representative) and split on canonical IDs. **This is the #1 correctness gotcha.**
- **Honest baselines** (majority class, crossing-count heuristic, kNN on raw grid) so a
  reported CNN accuracy is real lift, not an artifact — the real-vs-illusory discipline.
- Balanced classes, per-dimension reporting, significance on accuracy, calibration.

## 5. Engineering structure (autonomous)
- `data/` — loaders for the CSVs + the SQLite release; balanced sampling; canonical-dedup splits.
- `oracles/` — invariant labelers (mosaics.py-based + PD-code → external lib).
- `models/` — baseline CNN, deeper CNN, D4-equivariant CNN.
- `train/`, `eval/` — experiment runner, metrics, scaling curves, leaderboard.
- `autonomous/` — research-agent loop + experiment queue + dashboard + Telegram + git
  auto-commit of results (reuse the `autonomous-research` toolkit).
- **Compute** — CNNs here are small (≤14×14×11); a modest GPU or even CPU suffices for the
  baselines; the high-compute box handles data-gen. (No box provisioned without explicit OK.)

## 6. Immediate next steps
1. Confirm the repo approach (see below) + name.
2. Stand up the project repo: plan + reorganized code from `concurrency`.
3. Pull the 3.2M-mosaic DB release; reproduce the unknot-detection CNN as the baseline,
   add the canonical-dedup leakage-safe split.
4. Expand to the next invariants + wire up the autonomous loop + dashboard.

## 7. Repo / "fork" note
`gh` is authenticated as `andrew-tawfeek`, so GitHub won't let us fork KnotAI into the same
account. Options: (a) **new repo** `andrew-tawfeek/knot-invariant-cnn` seeded from the
`concurrency` branch (recommended — a clean home for the autonomous project, like
`evolutionary-rsi`); (b) a dedicated long-lived branch inside `KnotAI`; (c) fork into a GitHub
org if you have one. Recommend (a).
