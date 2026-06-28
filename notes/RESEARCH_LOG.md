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

## 2026-06-28 — Dashboard: expanded mosaic / dataset / training statistics ("?" request)
- Andrew via the "?" channel: *"Add some more general statistics on the mosaics... more
  data and statistics on the dataset and the training."* Built it end-to-end.
- **collector.py.** (1) Fixed a real surfacing bug: result JSON nests CNN metrics under
  `cnn` and baselines as a list, but `scan_results()` only looked for flat keys, so the one
  real run was showing as `null` on the page. Now parses the nested `cnn` block + baselines
  list, and surfaces full per-run detail (precision/recall/specificity/F1, confusion matrix,
  every honest baseline, and the leakage-safe split stats: raw→canonical dedup, collapsed,
  label conflicts, train/val/test sizes, class balance, pad/seed). (2) Added `db.summary`:
  derived corpus stats (suitably-connected total + connectivity rate, labeled total + label
  coverage, knotted total + knotted rate, labeled vs unlabeled dimension lists, dim range).
- **index.html.** Database tab now shows 8 summary cards (added suitably-connected %, labeled,
  knotted %, unknot, dim range, unlabeled-dims warning), a new "Composition by dimension" line
  chart (connectivity rate + knotted-fraction-among-labeled — the climb is visible), and
  within-dimension conn%/knot% columns in the breakdown table. Results tab now renders a full
  per-run detail card: CNN metric chips, a confusion matrix, a baseline table with the CNN's
  lift over *each* honest baseline, and a dataset-construction panel (dedup, conflicts, splits,
  class balance). Verified: collector restarted & regenerated status.json, authed curl 200 /
  anon 401, page JS passes `node --check`.

## 2026-06-28 — Second CNN run (dims 5–12) + DB near 10M: the lift widens with dimension
- **DB growth.** 9,971,096 mosaics across dims 3–20, up +1,176,474 (~+1.18M) since the last
  note's 8.79M — and now essentially at the 10M mark. Composition: 6,861,150
  suitably-connected (68.8% connectivity rate), 2,255,244 labeled (32.9% label coverage of the
  SC set), 150,422 knotted (6.67% of labeled). The shape is unchanged from before — dims 3–6
  exhaustive-ish, 7–10 sampled at 400k, 11–14 at a balanced 60k (5k/5k), and **dims 15–20 still
  carry 0 unknot / 0 knotted** (connectivity filter run, unknot-vs-knotted classification not),
  so the high-dim regime remains untrainable. Same standing watch item as the last two notes.
- **Second real training run** (results/20260628T032325Z_unknot_cnn.json), unknot_detection,
  this time on **dims 5–12** and at full scale: 183,026 raw rows → 179,525 unique canonical
  (3,501 collapsed, **0 label conflicts**), 70/15/15 split → train 125,667 / test **26,929**.
  That test set is ~57× the first run's n=472, so the metric is now on solid ground. CNN:
  balanced acc **0.8750**, F1 0.870, precision 0.912, recall 0.832, specificity 0.918
  (confusion tp 11329 / tn 12221 / fp 1089 / fn 2290). Early-stopped at 8/15 epochs (patience 5).
- **The lift grew, and *why* is the interesting part.** Honest baselines on this set:
  majority 0.50, crossing-count==0 rule 0.520, **logreg-on-handcounts 0.7144**. CNN lift over
  the best baseline = **+0.1606** — up from +0.096 on the first (dims 3–7) run. Note that the
  handcounts-logreg baseline itself *fell* from 0.806 (dims 3–7) to 0.714 (dims 5–12). So the
  widening gap isn't the CNN getting luckier on more data; it's that **hand-counted features
  degrade faster than the CNN as dimension rises**. Higher-dimensional diagrams admit more
  complex spatial arrangements that crossing/strand tallies can't separate, while the CNN's 2-D
  structural read keeps holding (0.903→0.875, a mild drop) — exactly the regime where the
  mosaic-CNN premise should pay off, now with evidence at scale.
- **Next / watch.** (1) The dims 15–20 unknot/knotted labeling run remains the gating item for
  pushing past dim 12. (2) Recall (0.832) trailing specificity (0.918) says the model still
  misses a chunk of the positive class — worth seeing whether more epochs (it stopped at 8) or
  the harder high dims move that. (3) With both runs now surfaced on the dashboard, the lift
  trend (+0.096 → +0.161) is the headline to keep tracking as dims extend.

## 2026-06-28 — DB past 11.4M: labeling pours into dims 5–6, knotted-rate dilutes
- **DB growth.** 11,463,542 mosaics across dims 3–20, **+1,492,446 (~+1.49M)** since the last
  note's 9.97M. Composition: 7,924,737 suitably-connected (connectivity rate up a touch to
  **69.1%**), 2,630,023 labeled (label coverage of the SC set up to **33.2%**, +374,779 labels),
  165,281 knotted. Effectively all of this growth is in **dim 6** (now 8.04M) and **dim 5**
  (1.16M) — the exhaustive-ish low dims — while dims 7–10 hold at 400k, 11–14 at the balanced
  60k (5k/5k), and **dims 15–20 still sit at 0 unknot / 0 knotted**. Same standing gate as the
  last three notes: the high-dim regime stays untrainable until that classification run lands.
- **The composition tell.** The 374,779 new labels split ~359,920 unknot / ~14,859 knotted —
  only **3.96% knotted**, well under the corpus-wide 6.28%. So the knotted *rate* ticked down
  (6.67% → **6.28%**) not because knots got rarer but because the fresh labels are all low-dim
  (dim 5–6) where unknots dominate. Worth remembering when reading the headline knotted-fraction:
  it's currently being pulled toward the easy end of the dimension range, the opposite direction
  from where the signal-rich hard diagrams live.
- **No new training.** Still the two real runs (dims 3–7, lift +0.096; dims 5–12, lift +0.161) —
  nothing new landed in the Results panel this cycle. The DB is feeding the trainable low-dim
  pool, but the next *result* worth waiting on is still either a higher-dim retrain or the 15–20
  labeling that unlocks it. Lift trend to keep tracking: +0.096 → +0.161.

## 2026-06-28 — DB at 13.78M: dim 6 now 75% of the corpus — a balance flag
- **DB growth.** 13,781,085 mosaics across dims 3–20, **+2,317,543 (~+2.32M)** since the last
  note's 11.46M — the largest single-cycle jump yet (prior jumps +1.18M, +1.49M). Composition:
  9,575,424 suitably-connected (connectivity 69.5%), 3,210,795 labeled (label coverage 33.5%),
  188,307 knotted (5.86% of labeled). As before, **dims 15–20 still sit at 0 unknot / 0 knotted** —
  the same standing gate that keeps the high-dim regime untrainable.
- **The concentration is becoming the story.** Effectively *all* of this cycle's +2.32M landed in
  **dim 6**, which is now **10.36M mosaics = 75.1% of the entire corpus** (dim 5 unchanged at
  1.16M; dims 7–14 flat at their 400k / 60k caps). The DB is no longer just "low-dim heavy" — it
  is dim-6-dominated to the point that any model trained on a naive all-dims pool is implicitly a
  dim-6 model. **Recommendation (for the training agent / Andrew): the next unknot retrain should
  sample per-dimension with a cap, not draw proportionally**, or the +0.16 lift trend will quietly
  become a dim-6 measurement rather than a cross-dimensional one.
- **Dilution continues, as expected.** The 580,772 new labels split ~557,746 unknot / ~23,026
  knotted — **3.96% knotted**, again well under the corpus rate, so the knotted fraction ticked
  6.28% → **5.86%**. Same mechanism as the last two notes (fresh labels are all easy low-dim dim-6
  unknots), now a third confirming data point. The signal-rich knotted diagrams remain in the
  higher, under-sampled dims.
- **No new training.** Results panel unchanged — still the two real runs (dims 3–7, lift +0.096;
  dims 5–12, lift +0.161). Nothing new landed this cycle; the next *result* worth waiting on is
  still a balanced higher-dim retrain or the dims 15–20 labeling that unlocks past dim 12. Lift
  trend to keep tracking: +0.096 → +0.161.
