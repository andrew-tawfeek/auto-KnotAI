#!/usr/bin/env bash
# INTENT: Hardest-regime unknot retrain on dims 8-12 ONLY (drops dim-7 too, the
#   lightest of the high-dim pools) with a 3-stage conv stack 32 64 128 + dense
#   256, epochs 22. Stress-tests the spatial-signal premise on the densest-knotted
#   high dims with more representational depth. Distinct dims-set from the others.
# SERVES: RESEARCH_DIRECTION.md #1 (cross-dim hard-diagram measurement / isolate
#   dim-6 artifact); standing scaling track (vary dims/conv-filters). No new dir.
# SAFETY: leakage-safe harness only; no box control, no DB writes, no SSH.
set -euo pipefail
cd "$(dirname "$0")/.."
while tmux has-session -t train 2>/dev/null || tmux has-session -t train_cross 2>/dev/null; do sleep 20; done
python3 -m harness.train_cnn \
  --dims 8 9 10 11 12 \
  --epochs 22 \
  --patience 7 \
  --batch-size 64 \
  --conv-filters 32 64 128 \
  --dense-units 256 \
  --seed 42
