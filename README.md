# auto-KnotAI

Autonomous research project: **training CNNs to recognize knot invariants from knot mosaics**,
and studying *which* invariants are learnable and *how* that scales with mosaic dimension and
data size. Seeded from the `concurrency` branch of [`KnotAI`](https://github.com/andrew-tawfeek/KnotAI)
and structured on the [`autonomous-research`](https://github.com/andrew-tawfeek/autonomous-research)
toolkit (agent loop, monitoring, dashboard, crash-safe result committing).

See **[PROJECT_PLAN.md](PROJECT_PLAN.md)** for the full scientific scope, invariant ladder,
rigor protocol (symmetry-aware dedup, honest baselines), and engineering structure.

## Inherited from KnotAI
- `mosaics.py` — core mosaic library (Mosaic/Tile/Matrix; crossings, connectivity, PD-code walk).
- `cnn_train.py` — working unknot-detection CNN (the baseline).
- `data_gen.py`, `multithreading-data-gen.py` — mosaic dataset generation.
- `datasets/` — labeled CSVs (unknot dims 3–14, trefoil, connected).
- `.deploy/` — data publish/watch daemons. A 3.2M-mosaic SQLite DB is published as a
  KnotAI GitHub release (`data-20260525-214315`).

## Status
Kicked off 2026-06-28. Data generation resuming on the high-compute box; training harness +
autonomous loop + dashboard under construction.
