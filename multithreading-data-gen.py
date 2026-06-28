"""Multi-dataset generation driver.

Generates several balanced datasets at once, combining two layers of
concurrency without oversubscribing the machine:

  * outer (multiprocessing.Process): independent (type, dimension) jobs run
    concurrently, in waves;
  * inner (data_gen's worker pool): each job parallelizes its own candidate
    generation across worker processes.

A total CPU budget (``--workers``) is split between the two layers: we run
``--jobs-parallel`` jobs at a time and give each job ``budget // jobs-parallel``
inner workers. For the cheap dataset types -- whose bottleneck is the
single-process bookkeeping rather than candidate generation -- inner workers are
pinned to 1 and the budget is spent entirely on running more jobs concurrently.

Examples:
    # original default behavior, now core-aware:
    python multithreading-data-gen.py
    # explicit:
    python multithreading-data-gen.py --type unknot --dims 10-14 --samples 10000
    python multithreading-data-gen.py --type component --dims 5,8,10 --samples 3000 --workers 10
"""

import argparse
import multiprocessing
import os

import data_gen


def _parse_dims(spec):
    """Parse a dimension spec like ``"10-14"`` or ``"5,8,10"`` into a list."""
    dims = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            dims.extend(range(int(lo), int(hi) + 1))
        else:
            dims.append(int(part))
    return dims


def _run_job(data_type, num_samples, dimension, inner_workers):
    data_gen.generate(data_type, num_samples, dimension, workers=inner_workers)


def _plan(data_type, n_jobs, total_workers, jobs_parallel):
    """Decide outer concurrency and per-job inner worker count from the budget."""
    outer = jobs_parallel or min(n_jobs, total_workers)
    outer = max(1, min(outer, n_jobs))
    # Heavy types benefit from inner candidate parallelism; cheap types don't, so
    # spend the whole budget on running more jobs at once instead.
    if data_type in data_gen._HEAVY_TYPES:
        inner = max(1, total_workers // outer)
    else:
        inner = 1
    return outer, inner


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple balanced mosaic datasets concurrently.",
    )
    parser.add_argument("--type", default="unknot", choices=sorted(data_gen._GENERATORS))
    parser.add_argument("--dims", default="10-14", help="e.g. '10-14' or '5,8,10'")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="total CPU budget across all jobs (default: all CPU cores).",
    )
    parser.add_argument(
        "--jobs-parallel",
        type=int,
        default=None,
        help="how many dataset jobs to run at once (default: as many as the budget allows).",
    )
    args = parser.parse_args()

    dims = _parse_dims(args.dims)
    total = args.workers or os.cpu_count() or 1
    outer, inner = _plan(args.type, len(dims), total, args.jobs_parallel)

    print(
        f"Generating '{args.type}' x {args.samples} for dims {dims}: "
        f"{outer} concurrent job(s) x {inner} inner worker(s) "
        f"(budget={total} cores)."
    )

    # Run jobs in waves of `outer` concurrent processes.
    pending = list(dims)
    while pending:
        wave, pending = pending[:outer], pending[outer:]
        procs = [
            multiprocessing.Process(
                target=_run_job,
                args=(args.type, args.samples, d, inner),
            )
            for d in wave
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
