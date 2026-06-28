#!/usr/bin/env bash
# INTENT: High-dimensional unknot-detection retrain on dims 7-12 ONLY (dims 3-6
#   deliberately excluded to remove the dim-6 mass that now is 77% of the corpus).
#   Goal: a clean CROSS-DIMENSIONAL measurement that tells us whether the
#   +0.096 -> +0.161 CNN-over-baseline lift is real hard-diagram signal or a
#   dim-6 artifact. dims 7-10 = knotted-rich 400k samples; 11-12 = balanced 5k/5k.
#   More epochs (20) + patience 7 + a dense head to chase the recall<specificity
#   gap (0.832 vs 0.918) from the dims-5-12 run. Writes results/<ts>_unknot_cnn.json.
# SERVES STEERING: standing brief (no new operator steering this cycle) — honest
#   cross-dimensional measurement before trusting the headline lift; first step of
#   the pivot toward the genuinely-hard invariants.
# SAFETY: invokes only the existing leakage-safe harness; no box control, no SSH,
#   no DB writes. Relayed to the L4 by the ship process; NOT executed by Researcher.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m harness.train_cnn \
  --dims 7 8 9 10 11 12 \
  --epochs 20 \
  --patience 7 \
  --batch-size 64 \
  --dense-units 64 \
  --seed 42
