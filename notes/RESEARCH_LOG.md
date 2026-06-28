# Research notes (auto-maintained)

## 2026-06-28 — Project kickoff
- auto-KnotAI created, seeded from KnotAI@concurrency + autonomous-research foundation.
- **Data generation resumed** on the high-compute box (c2-standard-16, us-central1-c): the
  existing 3.2M-mosaic SQLite DB is growing via `tabulate_db.py generate 3 14 --workers 14`
  (PK-dedup + per-dimension exhaustion gating; dims 3–4 already exhausted, dim 5+ in progress).
  Kept alive by `keeper.sh`.
- Next: training harness (leakage-safe canonical-dedup splits) + the L4 GPU box, then the
  autonomous research loop + the monitoring dashboard.

## 2026-06-28 — First end-to-end training result + DB at ~8.8M
- **DB growth.** Now 8,794,622 mosaics across dims 3–20 (was ~3.2M at kickoff). Shape of the
  corpus: dims 3–6 are effectively exhaustive (dim 6 alone = 5.37M, with 3.83M
  suitably-connected / 1.30M unknot / 54k knotted); dims 7–10 are sampled at 400k each
  (knotted fraction climbs with dimension: 7.4k→12.9k→14.3k→23.2k as dim goes 7→10, as
  expected — higher mosaic dimension admits more genuinely-knotted diagrams). Dims 11–14 sit
  at 60k with a deliberately balanced 5000 unknot / 5000 knotted label split (good for
  training). Dims 15–20 (60k each) have suitably-connected counts (~36k–46k) but **0 unknot /
  0 knotted** — the connectivity filter has run there but the unknot-vs-knotted classification
  has not yet. Worth watching: those high dims are currently unlabeled and can't feed a
  classifier until that step lands.
- **First real CNN result** (results/20260628T024401Z_unknot_cnn.json), unknot_detection on
  dims 3–7. Leakage-safe split is doing its job: 3535 raw rows → 3149 unique canonical (386
  collapsed, **0 label conflicts**), 70/15/15 → test n=472. CNN: balanced acc **0.9026**,
  F1 0.903, precision 0.940, recall 0.869, specificity 0.937. Honest baselines on the same
  test set: majority-class 0.50, crossing-count==0 rule 0.60, **logreg-on-handcounts 0.806**.
  CNN lift over the best (logreg) baseline = **+0.096 balanced acc**.
- **Implication.** This replaces the earlier sample-data placeholder (90.3% vs 81%) with a
  result on the real DB, and the comparison now has teeth: the strongest non-deep baseline is
  hand-counted-feature logreg at ~0.806, and the CNN beats it by ~9.6 points. That gap is the
  evidence that the mosaic's 2-D spatial structure carries unknotting signal beyond what hand
  counts capture — the whole premise of the approach.
- **Next / watch.** (1) Push the same harness up to dims 8–14, which now have balanced labels,
  to see whether the lift holds or grows as diagrams get harder. (2) Get the unknot/knotted
  labeling run for dims 15–20 so the high-dimensional regime becomes trainable. (3) Current
  result is small (3149 canonical examples); more dims = more data and a sterner test of
  generalization.
