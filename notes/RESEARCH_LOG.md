# Research notes (auto-maintained)

## 2026-06-28 — Project kickoff
- auto-KnotAI created, seeded from KnotAI@concurrency + autonomous-research foundation.
- **Data generation resumed** on the high-compute box (c2-standard-16, us-central1-c): the
  existing 3.2M-mosaic SQLite DB is growing via `tabulate_db.py generate 3 14 --workers 14`
  (PK-dedup + per-dimension exhaustion gating; dims 3–4 already exhausted, dim 5+ in progress).
  Kept alive by `keeper.sh`.
- Next: training harness (leakage-safe canonical-dedup splits) + the L4 GPU box, then the
  autonomous research loop + the monitoring dashboard.
