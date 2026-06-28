"""Leakage-safe training harness for the knot-invariant CNN project.

Modules:
  mosaic_io  -- parse / one-hot / pad mosaics
  canonical  -- D4 (dihedral) canonical form for symmetry-aware dedup
  dataset    -- canonical-key leakage-safe train/val/test splits (CSV + SQLite-ready)
  baselines  -- honest non-CNN baselines
  train_cnn  -- small CNN for unknot detection on the leakage-safe split
"""
