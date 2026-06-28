#!/usr/bin/env bash
# INTENT: High-dim unknot retrain (dims 7-12), DEEPER/LONGER variant. Two-stage
#   conv 32 64 + dense 128, epochs 24 / patience 8 to probe whether longer
#   training on hard diagrams (not dim-6) keeps improving the lift or plateaus.
#   Complements the wide-head and baseline dims-7-12 jobs (hyperparam sweep).
# SERVES: RESEARCH_DIRECTION.md #1 (cross-dim hard-diagram measurement); standing
#   scaling track (vary epochs/conv-filters/dense-units). No new direction.
# SAFETY: leakage-safe harness only; no box control, no DB writes, no SSH.
set -euo pipefail
cd "$(dirname "$0")/.."
while tmux has-session -t train 2>/dev/null || tmux has-session -t train_cross 2>/dev/null; do sleep 20; done
python3 -m harness.train_cnn \
  --dims 7 8 9 10 11 12 \
  --epochs 24 \
  --patience 8 \
  --batch-size 64 \
  --conv-filters 32 64 \
  --dense-units 128 \
  --seed 42
