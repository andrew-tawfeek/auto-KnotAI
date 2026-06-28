"""Tabulate knot-mosaic properties into a single SQLite database.

This module is a thin, conflict-free layer on top of the existing sampling
engine: it reuses ``data_gen``'s literature-grounded exhaustion model and
``mosaics``'s ``random_mosaic`` / property methods, and adds (a) a property
computer that records *every* attribute of a mosaic in one row, (b) a parallel
generation pipeline that writes those rows into one SQLite table, and (c) a
prepopulator that imports the legacy per-task CSVs in ``datasets/``.

Every mosaic is keyed by its canonical matrix string and inserted with
``INSERT OR IGNORE``, so existing rows are never overwritten and re-runs simply
skip mosaics that are already tabulated.

Generation walks one dimension at a time. A dimension finishes when the
coupon-collector exhaustion detector fires (small dimensions genuinely exhaust
the finite mosaic space) or a safety cap on attempts/rows is hit (large
dimensions never truly exhaust).

CLI::

    python tabulate_db.py prepopulate                  # import datasets/*/dim_*.csv
    python tabulate_db.py generate 3 20                # tabulate dims 3..20
    python tabulate_db.py generate 3 20 --workers 32 --unknot-max-dim 12
    python tabulate_db.py stats                        # per-dimension summary

Typically driven end-to-end by ``run_tabulation.sh <lo> <hi>``.
"""

import argparse
import ast
import os
import random
import sqlite3
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import data_gen
from mosaics import Mosaic, random_mosaic, pdCode


DATASET_DIR = Path("datasets")
DEFAULT_DB = Path("mosaics.db")

# Column order used everywhere a row tuple is built or inserted.
_COLUMNS = (
    "mosaic",                  # canonical str(list(matrixRepresentation)) -- PRIMARY KEY
    "dimension",
    "is_suitably_connected",   # 0/1
    "num_crossings",
    "has_crossing",            # 0/1 (num_crossings > 0)
    "num_components",          # NULL unless suitably connected
    "is_unknot",               # NULL unless suitably connected single-component knot
    "pd_code",                 # str(pdCode(m)); NULL if no crossings / failure
)

_INSERT_SQL = (
    f"INSERT OR IGNORE INTO mosaics ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(_COLUMNS))})"
)


# ---------------------------------------------------------------------------
# Property computation (runs inside worker processes -- the expensive
# unknot_check is therefore parallel)
# ---------------------------------------------------------------------------
def _row_from_mosaic(m, trusted_unknot=None, compute_unknot=True):
    """Derive the full property row tuple (in ``_COLUMNS`` order) from a Mosaic.

    ``trusted_unknot`` (bool or None): if the unknot status is already known
    (a CSV label, or because the generator targeted it), pass it to skip the
    expensive spherogram call. Only honored for genuine knots (suitably
    connected, single component).

    ``compute_unknot``: when False, never invoke ``unknot_check``; ``is_unknot``
    stays NULL for knots. Lets the caller throttle the expensive call at large
    dimensions.
    """
    matrix = list(m.matrixRepresentation)  # list of int-lists (canonical ints)
    mosaic_str = str(matrix)
    dimension = len(matrix)

    num_crossings = m.numCrossings()
    has_crossing = 1 if num_crossings > 0 else 0

    try:
        sc = bool(m.isSuitablyConnected())
    except Exception:
        sc = False
    is_sc = 1 if sc else 0

    num_components = None
    is_unknot = None
    if sc:
        try:
            num_components = int(m.numComponents())
        except Exception:
            num_components = None

        if num_components == 1:
            if trusted_unknot is not None:
                is_unknot = 1 if trusted_unknot else 0
            elif compute_unknot:
                try:
                    is_unknot = 1 if m.unknot_check() else 0
                except Exception:
                    is_unknot = None

    pd = None
    if num_crossings > 0:
        try:
            pd = str(pdCode(m))
        except Exception:
            pd = None

    return (mosaic_str, dimension, is_sc, num_crossings, has_crossing,
            num_components, is_unknot, pd)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def connect(db_path):
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _init_db(conn)
    return conn


def _init_db(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mosaics (
            mosaic                TEXT PRIMARY KEY,
            dimension             INTEGER NOT NULL,
            is_suitably_connected INTEGER NOT NULL,
            num_crossings         INTEGER NOT NULL,
            has_crossing          INTEGER NOT NULL,
            num_components        INTEGER,
            is_unknot             INTEGER,
            pd_code               TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dimension ON mosaics(dimension)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_unknot ON mosaics(dimension, is_unknot)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_components ON mosaics(dimension, num_components)")
    conn.commit()


def _existing_count(conn, dimension):
    return conn.execute(
        "SELECT COUNT(*) FROM mosaics WHERE dimension=?", (dimension,)
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Prepopulation from the legacy per-task CSVs
# ---------------------------------------------------------------------------
# Only the unknot label lets us skip an expensive recomputation; everything
# else is recomputed from the matrix (cheap) for consistency.
_TRUSTED_UNKNOT_COLUMNS = {"Is Unknot"}


def _prepopulate_worker(args):
    """Worker: (mosaic_str, trusted_unknot) -> full row tuple, or None."""
    mosaic_str, trusted_unknot = args
    try:
        m = Mosaic(ast.literal_eval(mosaic_str))
        return _row_from_mosaic(m, trusted_unknot=trusted_unknot, compute_unknot=True)
    except Exception:
        return None


def prepopulate_from_csv(conn, csv_dir=DATASET_DIR, workers=None):
    """Import every ``datasets/*/dim_*.csv`` into the DB (INSERT OR IGNORE)."""
    import pandas as pd
    import multiprocessing as mp

    csv_dir = Path(csv_dir)
    csv_paths = sorted(csv_dir.glob("*/dim_*.csv"))
    if not csv_paths:
        print(f"No CSVs found under {csv_dir}/. Nothing to prepopulate.")
        return 0

    # Build deduped work list across all CSVs: mosaic_str -> trusted_unknot.
    jobs = {}
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"  ! skipping {path}: {exc}")
            continue
        if df.empty or "Mosaic" not in df.columns:
            continue
        trusted_col = next((c for c in df.columns if c in _TRUSTED_UNKNOT_COLUMNS), None)
        before = len(jobs)
        for _, row in df.iterrows():
            mosaic_str = row["Mosaic"]
            trusted = bool(row[trusted_col]) if trusted_col is not None else None
            # A trusted label upgrades an earlier untrusted (None) entry.
            if mosaic_str not in jobs or (jobs[mosaic_str] is None and trusted is not None):
                jobs[mosaic_str] = trusted
        print(f"  {path}: {len(df)} rows (+{len(jobs) - before} new unique), "
              f"unknot label={'yes' if trusted_col else 'no'}")

    work = list(jobs.items())
    print(f"\nComputing properties for {len(work):,} unique mosaics from {len(csv_paths)} CSVs...")

    workers = workers or os.cpu_count() or 1
    inserted = 0
    processed = 0
    start = time.monotonic()
    ctx = mp.get_context("spawn")
    cur = conn.cursor()
    batch = []

    def flush():
        nonlocal inserted
        if not batch:
            return
        before = conn.total_changes
        cur.executemany(_INSERT_SQL, batch)
        inserted += conn.total_changes - before
        conn.commit()
        batch.clear()

    with ctx.Pool(processes=workers) as pool:
        for row in pool.imap_unordered(_prepopulate_worker, work, chunksize=64):
            processed += 1
            if row is not None:
                batch.append(row)
            if len(batch) >= 500:
                flush()
            if processed % 2000 == 0:
                elapsed = time.monotonic() - start
                print(f"\r  processed {processed:,}/{len(work):,} "
                      f"inserted={inserted:,} elapsed={elapsed:,.1f}s", end="", flush=True)
        flush()

    elapsed = time.monotonic() - start
    print(f"\nPrepopulation complete: {inserted:,} new rows "
          f"({processed - inserted:,} already present) in {elapsed:,.1f}s.")
    return inserted


# ---------------------------------------------------------------------------
# Parallel generation
# ---------------------------------------------------------------------------
# Weighted mixed sampling strategies; each returns (Mosaic, trusted_unknot).
# We deliberately avoid unknot-targeted generation: random_mosaic(unknot=...)
# calls the expensive spherogram check up to 5000x *internally* per draw. Plain
# suitably-connected / component-targeted sampling discovers both unknots and
# non-unknots naturally, and each knot gets exactly one unknot_check.
_STRATEGY_WEIGHTS = (
    ("free", 2),    # unrestricted: populates non-suitably-connected rows
    ("sc", 3),      # suitably connected: real knots/links of every kind
    ("comp1", 1),   # single-component knots (biases toward unknot labeling)
    ("comp2", 1),   # two-component links
)
_FAIL_LIMIT = 2     # consecutive failures before a strategy is disabled at a dim

# Above this dimension, drop the component-targeted strategies. random_mosaic
# with an exact num_components almost never satisfies the constraint on a large
# random mosaic, so it burns its full internal 5000-retry budget per draw
# (each retry runs a numComponents strand-walk) -- collapsing throughput to a
# crawl. Worse, when targeting occasionally *succeeds* the per-strategy
# "disable after N failures" counter resets, so the strategy never gets retired
# and the stall persists. The "sc" strategy still yields suitably-connected
# mosaics with naturally varied component counts, so num_components stays
# populated; we just forgo *targeted* 1-/2-component oversampling at large dims.
_COMPONENT_TARGET_MAX_DIM = 12
_COMPONENT_STRATEGIES = ("comp1", "comp2")

# Per-worker state (module-level so it survives across batch calls under spawn).
_WORKER = {}


def _make_strategies(dimension):
    builders = {
        "free": lambda: (random_mosaic(dimension, suitably_connected=False), None),
        "sc": lambda: (random_mosaic(dimension, suitably_connected=True), None),
        "comp1": lambda: (random_mosaic(dimension, num_components=1), None),
        "comp2": lambda: (random_mosaic(dimension, num_components=2), None),
    }
    pool = []
    for name, weight in _STRATEGY_WEIGHTS:
        if name in _COMPONENT_STRATEGIES and dimension > _COMPONENT_TARGET_MAX_DIM:
            continue
        pool.extend([name] * weight)
    return builders, pool


def _worker_init(dimension, compute_unknot):
    random.seed()
    np.random.seed(int.from_bytes(os.urandom(4), "little"))
    builders, pool = _make_strategies(dimension)
    _WORKER.update(
        builders=builders,
        pool=pool,
        dead={name: False for name, _ in _STRATEGY_WEIGHTS},
        streak={name: 0 for name, _ in _STRATEGY_WEIGHTS},
        i=0,
        compute_unknot=compute_unknot,
    )


def _worker_next_mosaic():
    """Round-robin one live strategy and return (Mosaic, trusted) or None."""
    builders, pool = _WORKER["builders"], _WORKER["pool"]
    dead, streak = _WORKER["dead"], _WORKER["streak"]
    name = pool[_WORKER["i"] % len(pool)]
    _WORKER["i"] += 1
    if dead[name]:
        # "free" is unconstrained and never dies, so it is always a fallback.
        if all(dead[n] for n, _ in _STRATEGY_WEIGHTS if n != "free"):
            name = "free"
        else:
            return None
    try:
        result = builders[name]()
        streak[name] = 0
        return result
    except (ValueError, AssertionError):
        streak[name] += 1
        if streak[name] >= _FAIL_LIMIT and name != "free":
            dead[name] = True
        return None


def _worker_produce_batch(batch_size):
    """Produce up to ``batch_size`` row tuples (Nones for skipped draws)."""
    rows = []
    for _ in range(batch_size):
        drawn = _worker_next_mosaic()
        if drawn is None:
            rows.append(None)
            continue
        m, trusted = drawn
        try:
            rows.append(_row_from_mosaic(m, trusted_unknot=trusted,
                                         compute_unknot=_WORKER["compute_unknot"]))
        except Exception:
            rows.append(None)
    return rows


def _parallel_rows(dimension, workers, batch_size, window, compute_unknot):
    """Endless stream of row tuples (or None) from a worker pool.

    Mirrors ``data_gen._parallel_candidates``: a bounded window of in-flight
    batches keeps workers busy and provides backpressure; closing the generator
    cancels pending work and tears the pool down without waiting on speculative
    candidates still in progress.
    """
    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(dimension, compute_unknot),
    )
    inflight = deque()
    try:
        for _ in range(window):
            inflight.append(executor.submit(_worker_produce_batch, batch_size))
        while True:
            future = inflight.popleft()
            batch = future.result()
            inflight.append(executor.submit(_worker_produce_batch, batch_size))
            yield from batch
    finally:
        for future in inflight:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)


def _generate_dimension(conn, dimension, workers, batch_size, max_attempts,
                        target_rows, compute_unknot, progress=True):
    existing = _existing_count(conn, dimension)
    cur = conn.cursor()
    attempts = 0
    new_count = 0       # all new rows (incl. non-suitably-connected)
    new_sc = 0          # new *suitably connected* rows -- gates exhaustion
    sc_seen = 0         # SC rows seen (new OR duplicate) -- coupon-collector rate
    last_new_sc = 0     # attempt index of the last new SC mosaic
    start = time.monotonic()
    next_progress = start
    initial_stall = data_gen._exhaustion_stall(dimension)
    stall = initial_stall
    stop_reason = None

    print(
        f"\n=== dim {dimension}: {existing:,} rows already tabulated; "
        f"{workers} workers (batch={batch_size}); "
        f"unknot_check={'on' if compute_unknot else 'off'}; "
        f"D_n_upper={data_gen._mosaic_count_upper(dimension):,} ==="
    )

    window = max(workers * 2, workers + 1)
    rows = _parallel_rows(dimension, workers, batch_size, window, compute_unknot)
    row_iter = iter(rows)

    try:
        while True:
            if max_attempts and attempts >= max_attempts:
                stop_reason = "max_attempts"
                break
            if target_rows and (existing + new_count) >= target_rows:
                stop_reason = "target_rows"
                break

            # Exhaustion is gated on *suitably connected* novelty: those form
            # the finite, D_n-bounded population we can actually exhaust. The
            # non-suitably-connected space is ~11^(n*n) and never runs dry, so
            # gating on it would mean "never exhausted". The coupon-collector
            # denominator is the rate at which we *draw from* the SC space (new
            # OR duplicate) -- it stays roughly constant, whereas the new-SC
            # rate collapses to 0 post-exhaustion and would inflate the window
            # without bound.
            sc_rate = (sc_seen / attempts) if attempts else 1.0
            stall = (data_gen._exhaustion_stall(dimension, sc_rate)
                     if attempts > 1000 else initial_stall)
            if attempts > 0 and (attempts - last_new_sc) >= stall:
                stop_reason = "exhausted"
                break

            attempts += 1
            row = next(row_iter)
            if row is not None:
                is_sc = bool(row[2])  # is_suitably_connected
                if is_sc:
                    sc_seen += 1
                before = conn.total_changes
                cur.execute(_INSERT_SQL, row)
                if conn.total_changes > before:
                    new_count += 1
                    if is_sc:
                        new_sc += 1
                        last_new_sc = attempts

            if attempts % 500 == 0:
                conn.commit()

            if progress:
                now = time.monotonic()
                if now >= next_progress:
                    total = existing + new_count
                    rate = attempts / (now - start) if now > start else 0
                    since_new = attempts - last_new_sc
                    print(
                        f"\r  dim {dimension}: rows={total:,} (+{new_count:,}, "
                        f"+{new_sc:,} sc) attempts={attempts:,} sc_rate={sc_rate:6.2%} "
                        f"stall={since_new:,}/{stall:,} "
                        f"{rate:,.0f}/s elapsed={now - start:,.1f}s   ",
                        end="", flush=True,
                    )
                    next_progress = now + 0.5
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\n  interrupted; flushing and stopping workers...")
    finally:
        row_iter.close()  # tears down the worker pool
        conn.commit()

    total = existing + new_count
    print(
        f"\n  dim {dimension} done [{stop_reason}]: {total:,} rows total "
        f"(+{new_count:,} new this run, {new_sc:,} suitably connected, "
        f"{attempts:,} attempts)."
    )
    return new_count, stop_reason


def generate(db_path, dim_lo, dim_hi, workers=None, batch_size=4,
             max_attempts=None, target_rows=None, unknot_max_dim=None,
             progress=True):
    """Tabulate dimensions [dim_lo, dim_hi], one at a time, into the DB."""
    workers = workers or os.cpu_count() or 1
    conn = connect(db_path)
    grand_total_new = 0
    try:
        for dimension in range(dim_lo, dim_hi + 1):
            compute_unknot = (unknot_max_dim is None) or (dimension <= unknot_max_dim)
            new_count, _ = _generate_dimension(
                conn, dimension, workers, batch_size, max_attempts,
                target_rows, compute_unknot, progress=progress,
            )
            grand_total_new += new_count
    finally:
        conn.commit()
        conn.close()
    print(f"\nAll dimensions [{dim_lo}, {dim_hi}] complete. {grand_total_new:,} new rows added.")
    return grand_total_new


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_stats(db_path):
    conn = connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM mosaics").fetchone()[0]
    print(f"Database: {db_path}  ({total:,} mosaics)\n")
    if total == 0:
        conn.close()
        return
    header = (f"{'dim':>4} {'rows':>12} {'sc':>10} {'w/cross':>10} "
              f"{'1-comp':>10} {'multi':>10} {'unknot':>10} {'knotted':>10} {'unk?':>8}")
    print(header)
    print("-" * len(header))
    for r in conn.execute(
        """
        SELECT dimension,
               COUNT(*),
               SUM(is_suitably_connected),
               SUM(has_crossing),
               SUM(CASE WHEN num_components = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN num_components > 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN is_unknot = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN is_unknot = 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN num_components = 1 AND is_unknot IS NULL THEN 1 ELSE 0 END)
        FROM mosaics GROUP BY dimension ORDER BY dimension
        """
    ).fetchall():
        dim, n, sc, cross, c1, cmulti, unk, knotted, unlabeled = r
        print(f"{dim:>4} {n:>12,} {sc or 0:>10,} {cross or 0:>10,} {c1 or 0:>10,} "
              f"{cmulti or 0:>10,} {unk or 0:>10,} {knotted or 0:>10,} {unlabeled or 0:>8,}")
    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help=f"SQLite database path (default: {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("prepopulate", help="Import all datasets/*/dim_*.csv into the DB")
    p_pre.add_argument("--csv-dir", type=Path, default=DATASET_DIR)
    p_pre.add_argument("--workers", type=int, default=None)

    p_gen = sub.add_parser("generate", help="Tabulate a dimension range into the DB")
    p_gen.add_argument("dim_lo", type=int)
    p_gen.add_argument("dim_hi", type=int)
    p_gen.add_argument("--workers", type=int, default=None,
                       help="worker processes (default: all CPUs)")
    p_gen.add_argument("--batch-size", type=int, default=4,
                       help="mosaics computed per worker task (default: 4)")
    p_gen.add_argument("--max-attempts", type=int, default=None,
                       help="safety cap on sampling attempts per dimension")
    p_gen.add_argument("--target-rows", type=int, default=None,
                       help="stop a dimension once it holds this many total rows")
    p_gen.add_argument("--unknot-max-dim", type=int, default=None,
                       help="skip the expensive unknot_check above this dimension")
    p_gen.add_argument("--no-progress", action="store_true")

    sub.add_parser("stats", help="Print a per-dimension summary of the DB")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.command == "prepopulate":
        conn = connect(args.db)
        try:
            prepopulate_from_csv(conn, csv_dir=args.csv_dir, workers=args.workers)
        finally:
            conn.close()
    elif args.command == "generate":
        if args.dim_lo < 1 or args.dim_hi < args.dim_lo:
            print("Require 1 <= dim_lo <= dim_hi", file=sys.stderr)
            sys.exit(2)
        generate(
            args.db, args.dim_lo, args.dim_hi,
            workers=args.workers,
            batch_size=args.batch_size,
            max_attempts=args.max_attempts,
            target_rows=args.target_rows,
            unknot_max_dim=args.unknot_max_dim,
            progress=not args.no_progress,
        )
    elif args.command == "stats":
        print_stats(args.db)


if __name__ == "__main__":
    main()
