#!/usr/bin/env bash
# INTENT: High-dim unknot retrain (dims 7-12), WIDE-CAPACITY variant of the
#   standing dims-7-12 cross-dimensional measurement. conv-filters 64 64 +
#   dense-units 256 to test whether added capacity closes the recall<specificity
#   gap (0.832 vs 0.918) on hard diagrams without leaning on the dim-6 mass.
#   Distinct from the baseline dims-7-12 (dense 64, default conv) already queued.
# SERVES: RESEARCH_DIRECTION.md #1 (honest cross-dim hard-diagram measurement);
#   standing scaling track (vary conv-filters/dense-units). No new direction.
# SAFETY: leakage-safe harness only; no box control, no DB writes, no SSH.
set -euo pipefail
cd "$(dirname "$0")/.."
while tmux has-session -t train 2>/dev/null || tmux has-session -t train_cross 2>/dev/null; do sleep 20; done
python3 -m harness.train_cnn \
  --dims 7 8 9 10 11 12 \
  --epochs 20 \
  --patience 7 \
  --batch-size 64 \
  --conv-filters 64 64 \
  --dense-units 256 \
  --seed 42
