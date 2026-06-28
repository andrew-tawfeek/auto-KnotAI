# RESEARCH_DIRECTION — senior Researcher agent's owned plan (auto-KnotAI)

The Researcher agent (senior, two-tier hierarchy) owns this file. Each ~20-min cycle it re-reads
notes/STEERING.md as authoritative operator direction, reviews results/*.json + the live DB status +
RESEARCH_LOG.md progress, then records here the current prioritized research direction and a ranked
"Next experiments" plan (focused on the genuinely-hard invariants — knot-type, PD-code / Jones /
Alexander polynomials — not the trivial grid-readable ones). Concrete job scripts it authors go to
exp_queue/; lesser/logistical tasks it delegates to the junior Logistics agent go to
logistics-delegation.jsonl. The Researcher PLANS + QUEUES + DELEGATES; it never starts/stops compute.
Newest direction entry appended at the bottom.

## 2026-06-28 — Disentangle the lift from dim-6; open the road to hard invariants
**Direction.** No new operator steering this cycle (STEERING.md is still header-only), so I continue the standing brief: drive the *next results* toward the genuinely-hard invariants (knot-type, PD/Jones/Alexander) while first neutralizing the dim-6 dominance the log has flagged three cycles running. The DB is now 14.97M mosaics with **dim 6 = 11.54M (77% of the corpus)** and 3.51M labeled; a naive all-dims unknot retrain is structurally a dim-6 model, so the +0.096→+0.161 lift trend needs one clean cross-dimensional measurement before we trust it.

**Next experiments (ranked).**
- **1. High-dim unknot retrain, dims 7–12 (QUEUEABLE NOW — queued this cycle).** Drop dims 3–6 entirely to remove the dim-6 mass; train on dims 7–10 (knotted-rich 400k samples) + 11–12 (balanced 5k/5k). *Why now:* isolates whether the spatial-signal lift is real across hard diagrams or a dim-6 artifact — the single most important open question about our headline number. More epochs (20) + patience 7 + a dense head to chase the recall<specificity gap (0.832 vs 0.918) seen in the dims-5–12 run. **Baseline to beat:** logreg_handcounts (~0.71 on the dims-5–12 regime; expect it lower still here, so a wider lift if the premise holds). Serves: standing "hard-invariants / honest cross-dim measurement" direction.
- **2. Knot-type / trefoil detection (PROPOSED — gated on a label step).** `datasets/trefoil/dim_4.csv` already exists, and dims 7–10 carry the densest knotted pools (7.4k→23.2k knotted), so a *trefoil-vs-rest* or small fixed-class knot-type classifier is the first genuinely-hard invariant within reach. **Gating dependency:** the DB schema has only `pd_code TEXT` (no knot-type column) and `train_cnn.py` is hard-wired to the binary `unknot` head — so this needs (a) a label step mapping `pd_code`→knot type and (b) a multi-class task head in the harness. Recording as PROPOSED, not a fake job. **Baseline to beat:** majority-class + logreg_handcounts.
- **3. Alexander/Jones coefficient prediction (PROPOSED — the hard target).** Regress low-order Alexander (or Jones) coefficients from the mosaic, target computed per-row from `pd_code`. This is the invariant a histogram genuinely cannot fake. **Gating dependency:** a polynomial-from-pd_code label-generation pass over the DB + a regression head. Recording as PROPOSED; depends on the same label-infrastructure as #2.

**PROPOSED-job sketch (for #2/#3, do NOT auto-run):** `python3 -m harness.train_<multiclass|regress> --task knot_type --dims 7 8 9 10 --epochs 20` — blocked until the harness gains the task head + the DB/CSV gains the derived label column. The gating work (pd_code→invariant labeling, harness head) is the critical path to every result past unknot; flagged for Andrew/engineering, delegated observational pieces below.
