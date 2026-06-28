import ast
import math
import os
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mosaics import *
import numpy as np
import pandas as pd


DATASET_DIR = Path("datasets")


# Per-dataset metadata. ``batch_size`` is how many candidates a worker process
# computes per task, chosen to amortize inter-process overhead against the cost
# of a single candidate: cheap candidates (cheap predicates) batch more, while
# spherogram-backed unknot checks are expensive enough that single candidates
# already dwarf the IPC cost (and small batches limit speculative waste near the
# end of generation).
_DATASET_SPECS = {
    "unknot": {"label_col": "Is Unknot", "batch_size": 1},
    "connected": {"label_col": "Is Suitably Connected", "batch_size": 32},
    "has_crossing": {"label_col": "Has Crossing", "batch_size": 32},
    "component": {"label_col": "Is Multi Component", "batch_size": 8},
    "trefoil": {"label_col": "Is Unknot", "batch_size": 1},
}

# Types whose per-candidate cost (spherogram for unknot/trefoil; strand tracing
# for component) is high enough that multiprocessing pays off. For the cheap
# types the main-process bookkeeping is the bottleneck, so extra workers only
# add IPC overhead -- they default to serial. (Override either way via workers.)
_HEAVY_TYPES = {"unknot", "trefoil", "component"}


def _resolve_workers(data_type, workers):
    """Map the public ``workers`` argument to a concrete worker count.

    ``None`` -> smart per-type default (all cores for heavy types, 1 otherwise);
    ``0``    -> force all CPU cores regardless of type;
    other    -> use exactly that many.
    """
    cpu = os.cpu_count() or 1
    if workers is None:
        return cpu if data_type in _HEAVY_TYPES else 1
    if workers == 0:
        return cpu
    return max(1, workers)


# Exact D_n (# of suitably connected knot n-mosaics) via the state-matrix
# recursion of Oh-Hong-Lee-Lee, "Quantum knots and the number of knot mosaics",
# Quantum Information Processing 14 (2015).
_KNOWN_D_N = {
    1: 1,
    2: 2,
    3: 22,
    4: 2_594,
    5: 4_183_954,
    6: 101_393_411_126,
    7: 38_572_794_946_976_688,
    8: 234_855_052_870_954_480_828_416,
}
# Oh (2017), Topology Appl.: 4 <= delta <= (5+sqrt(13))/2; use the upper bound
# as a safe extrapolation for n beyond the table.
_GROWTH_UPPER = (5 + math.sqrt(13)) / 2  # approx 4.303


def _mosaic_count_upper(n):
    if n in _KNOWN_D_N:
        return _KNOWN_D_N[n]
    return int(_GROWTH_UPPER ** (n * n))


def _exhaustion_stall(n, accept_rate=1.0, confidence=0.999):
    """Consecutive attempts with no new unique mosaic before a needed class
    is declared exhausted. Coupon-collector tail: with class size K, c*K
    draws miss any remaining mosaic with probability <= exp(-c). We use
    K <= D_n as a literature-grounded upper bound; accept_rate folds in the
    fraction of draws that yield a usable candidate at all.
    """
    c = math.log(1.0 / (1.0 - confidence))  # ~6.91 at 99.9%
    p = max(accept_rate, 1e-6)
    return max(int(c * _mosaic_count_upper(n) / p), 1000)


def dataset_path(data_type, dimension):
    return DATASET_DIR / data_type / f"dim_{dimension}.csv"


def _legacy_dataset_path(data_type):
    paths = {
        "unknot": "unknot_data.csv",
        "connected": "suitably_connected_data.csv",
        "has_crossing": "has_crossing_data.csv",
        "component": "component_data.csv",
        "trefoil": "trefoil_vs_unknot_data.csv",
    }
    return Path(paths[data_type])


def _mosaic_dimension(mosaic_text):
    return len(ast.literal_eval(mosaic_text))


def _load_existing_rows(csv_path, label_col, dimension):
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    if "Mosaic" not in df.columns or label_col not in df.columns:
        raise ValueError(f"{csv_path} must contain 'Mosaic' and '{label_col}' columns")

    df = df.drop_duplicates(subset=["Mosaic"])
    df = df[df["Mosaic"].map(_mosaic_dimension) == dimension]
    return df[["Mosaic", label_col]].values.tolist()


def _print_progress(counts, num_samples, attempts, max_attempts, start_time, done=False):
    accepted = sum(counts)
    fraction = min(accepted / num_samples, 1) if num_samples else 0
    bar_width = 30
    filled = int(bar_width * fraction)
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = time.monotonic() - start_time
    acceptance_rate = accepted / attempts if attempts else 0
    suffix = "\n" if done else "\r"
    print(
        f"[{bar}] {accepted}/{num_samples} ({fraction:6.2%}) "
        f"class0={counts[0]} class1={counts[1]} "
        f"attempts={attempts}/{max_attempts} "
        f"accept={acceptance_rate:5.2%} elapsed={elapsed:,.1f}s",
        end=suffix,
        flush=True,
    )


def _alternating_target_sampler(make_with_target, failure_limit=3, verbose=True):
    """Wraps a `make_with_target(t: bool) -> mosaic` callable so it alternates
    targets uniformly, but disables any target that raises `failure_limit`
    times in a row (some constraints are infeasible at small dimension --
    e.g. non-unknot 1-component knots don't exist at dim=3, so `unknot=False`
    would otherwise burn random_mosaic's full internal retry budget on
    every other call).

    ``verbose`` controls the "disabling target" notice; worker processes pass
    verbose=False so the main process keeps sole ownership of the progress bar.
    """
    streak = {True: 0, False: 0}
    dead = {True: False, False: False}

    def sample():
        live = [t for t in (True, False) if not dead[t]]
        if not live:
            raise ValueError("All sampling targets disabled (constraints likely infeasible at this dimension)")
        target = random.choice(live)
        try:
            mosaic = make_with_target(target)
            streak[target] = 0
            return mosaic
        except (AssertionError, ValueError):
            streak[target] += 1
            if streak[target] >= failure_limit and not dead[target]:
                dead[target] = True
                if verbose:
                    print(
                        f"\n[sampler] disabling target={target!r} after "
                        f"{failure_limit} consecutive failures (constraint likely infeasible at this dimension).",
                        flush=True,
                    )
            raise

    return sample


def _build_sampler_and_predicate(data_type, dimension, verbose=True):
    """Constructs the (sample_fn, predicate) pair for a dataset type.

    Centralized so that both the in-process serial path and freshly spawned
    worker processes can rebuild identical samplers from picklable arguments
    (the data_type string and dimension), keeping the per-type sampling logic
    in exactly one place.
    """
    if data_type == "unknot":
        sampler = _alternating_target_sampler(
            lambda t: random_mosaic(dimension, num_components=1, unknot=t),
            verbose=verbose,
        )
        predicate = lambda m: m.unknot_check()
    elif data_type == "connected":
        sampler = lambda: random_mosaic(dimension, suitably_connected=random.choice([True, False]))
        predicate = lambda m: m.isSuitablyConnected()
    elif data_type == "has_crossing":
        sampler = lambda: random_mosaic(dimension)
        predicate = lambda m: m.numCrossings() > 0
    elif data_type == "component":
        sampler = _alternating_target_sampler(
            # t=True targets multi-component (class 1), t=False single-component (class 0).
            lambda t: random_mosaic(dimension, num_components=(2 if t else 1)),
            verbose=verbose,
        )
        predicate = lambda m: m.numComponents() > 1
    elif data_type == "trefoil":
        sampler = _alternating_target_sampler(
            lambda t: random_mosaic(dimension, num_crossings=3, num_components=1, unknot=t),
            verbose=verbose,
        )
        predicate = lambda m: m.unknot_check()
    else:
        raise ValueError(f"unknown data_type {data_type!r}")
    return sampler, predicate


def _classify(sample_fn, predicate):
    """Draws one candidate and classifies it.

    Returns ("ok", mosaic_str, label) on success or ("skip", None, reason) when
    the draw raises an expected infeasibility error. The mosaic is reduced to
    its string form here so the (cheap, picklable) result is all that crosses a
    process boundary -- never a Mosaic/ndarray.
    """
    try:
        mosaic = sample_fn()
        label = 1 if predicate(mosaic) else 0
        return ("ok", str(list(mosaic.matrixRepresentation)), label)
    except (AssertionError, ValueError) as exc:
        return ("skip", None, str(exc))


# --- Worker-process plumbing (module-level so it is picklable under 'spawn') ---

_WORKER_STATE = {}


def _worker_init(data_type, dimension):
    """ProcessPoolExecutor initializer: build this worker's sampler once.

    Each worker reseeds from OS entropy so the per-process candidate streams are
    independent (and so a fork()-based start method cannot duplicate the parent's
    RNG state).
    """
    random.seed()
    np.random.seed(int.from_bytes(os.urandom(4), "little"))
    sample_fn, predicate = _build_sampler_and_predicate(data_type, dimension, verbose=False)
    _WORKER_STATE["sample_fn"] = sample_fn
    _WORKER_STATE["predicate"] = predicate


def _worker_produce_batch(batch_size):
    sample_fn = _WORKER_STATE["sample_fn"]
    predicate = _WORKER_STATE["predicate"]
    return [_classify(sample_fn, predicate) for _ in range(batch_size)]


def _serial_candidates(data_type, dimension):
    """Endless single-process stream of classified candidates."""
    sample_fn, predicate = _build_sampler_and_predicate(data_type, dimension)
    while True:
        yield _classify(sample_fn, predicate)


def _parallel_candidates(data_type, dimension, workers, batch_size, window):
    """Endless stream of classified candidates produced by a worker pool.

    A bounded window of in-flight batch tasks keeps every worker busy while the
    main process does its bookkeeping, and blocking on the oldest task provides
    natural backpressure so we never queue unbounded speculative work. On close
    (the consumer stops early), pending tasks are cancelled and the pool is torn
    down without waiting on speculative candidates still in progress.
    """
    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(data_type, dimension),
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


def _save_rows(rows, csv_path, label_col):
    df = pd.DataFrame(rows, columns=['Mosaic', label_col])
    df = df.drop_duplicates(subset=["Mosaic"])
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}.")
    return len(df)


def _generate_balanced(
    num_samples,
    data_type,
    dimension,
    label_col,
    csv_path,
    legacy_csv_path=None,
    progress=True,
    workers=1,
):
    """Rejection-sample a balanced 50/50 dataset.

    Label encoding: 1 if the dataset's predicate holds, 0 otherwise.

    Candidate generation (the expensive, CPU-bound part) is delegated to a
    candidate source: a single-process generator when ``workers <= 1``, or a
    worker pool when ``workers > 1``. The acceptance/dedup/exhaustion bookkeeping
    below is unchanged and stays in this one process, so the resulting balanced
    dataset is produced by the same algorithm regardless of worker count.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_samples % 2 != 0:
        raise ValueError("num_samples must be even for a balanced 50/50 dataset")

    max_attempts = num_samples * (10 + dimension * dimension)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_existing_rows(csv_path, label_col, dimension)
    migrated_from_legacy = False
    if not rows and legacy_csv_path is not None and Path(legacy_csv_path).exists():
        rows = _load_existing_rows(Path(legacy_csv_path), label_col, dimension)
        if rows:
            migrated_from_legacy = True
            print(f"Found existing legacy data in {legacy_csv_path}; migrating to {csv_path}.")

    rows = [[row[0], int(row[1])] for row in rows]
    seen = {row[0] for row in rows}
    half = num_samples // 2
    counts = [
        sum(1 for row in rows if row[1] == 0),
        sum(1 for row in rows if row[1] == 1),
    ]
    if counts[0] >= half and counts[1] >= half:
        if migrated_from_legacy:
            _save_rows(rows, csv_path, label_col)
            print(f"Migrated data written to {csv_path}.")
        print(
            f"{csv_path} already has enough balanced samples for this request: "
            f"class0={counts[0]}, class1={counts[1]}, requested={num_samples}."
        )
        print("No new mosaics generated.")
        return

    print(
        f"Updating {csv_path}: existing class0={counts[0]}, class1={counts[1]}, "
        f"target per class={half}. (max_attempts={max_attempts})"
    )
    attempts = 0
    skipped = 0
    start_time = time.monotonic()
    next_progress_time = start_time
    attempts_at_last_new = [attempts, attempts]
    initial_stall = _exhaustion_stall(dimension)
    stall = initial_stall
    exhausted = [False, False]
    stop_reason = None

    def maybe_print_progress(extra=None):
        nonlocal next_progress_time
        if not progress:
            return
        now = time.monotonic()
        if now < next_progress_time:
            return
        _print_progress(counts, num_samples, attempts, max_attempts, start_time)
        if extra:
            print(extra, flush=True)
        next_progress_time = now + 0.5

    if progress:
        _print_progress(counts, num_samples, attempts, max_attempts, start_time)

    if workers and workers > 1:
        batch_size = _DATASET_SPECS[data_type]["batch_size"]
        window = max(workers * 2, workers + 1)
        candidates = _parallel_candidates(data_type, dimension, workers, batch_size, window)
        print(f"Using {workers} worker processes (batch_size={batch_size}).")
    else:
        candidates = _serial_candidates(data_type, dimension)
    candidate_iter = iter(candidates)

    try:
        while counts[0] < half or counts[1] < half:
            if attempts >= max_attempts:
                stop_reason = "max_attempts"
                break

            # Exhaustion: each still-needed class is declared exhausted (sticky)
            # only after its own stall-counter exceeds the coupon-collector tail.
            # When we declare a class exhausted, we reset the other class's
            # counter so it gets a fresh, full window of attention before being
            # judged too. Only when both classes are full-or-exhausted do we stop.
            accept_rate = (counts[0] + counts[1]) / attempts if attempts else 1.0
            stall = _exhaustion_stall(dimension, accept_rate) if attempts > 1000 else initial_stall
            for i in (0, 1):
                if exhausted[i] or counts[i] >= half:
                    continue
                if (attempts - attempts_at_last_new[i]) >= stall:
                    exhausted[i] = True
                    other = 1 - i
                    if not exhausted[other] and counts[other] < half:
                        attempts_at_last_new[other] = attempts
                        print(
                            f"\nClass {i} declared exhausted at attempt {attempts:,} "
                            f"(counts={counts}); shifting effort to class {other}.",
                            flush=True,
                        )
                    break  # only flip one class per iteration so the other gets its fresh window
            if all(counts[i] >= half or exhausted[i] for i in (0, 1)):
                stop_reason = "exhausted" if any(exhausted) else "complete"
                break

            attempts += 1

            status, mosaic_str, payload = next(candidate_iter)
            if status == "skip":
                skipped += 1
                maybe_print_progress(f" skipped invalid candidates={skipped}; last reason: {payload}")
                continue

            idx = payload
            if counts[idx] >= half:
                maybe_print_progress()
                continue

            if mosaic_str in seen:
                continue
            seen.add(mosaic_str)
            rows.append([mosaic_str, idx])
            counts[idx] += 1
            attempts_at_last_new[idx] = attempts

            maybe_print_progress()
        else:
            stop_reason = "complete"
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\nInterrupted by user; flushing collected mosaics to disk.")
    finally:
        candidate_iter.close()  # tears down the worker pool if one was started
        if progress:
            _print_progress(counts, num_samples, attempts, max_attempts, start_time, done=True)

        summary = f"class0={counts[0]}, class1={counts[1]} ({len(rows)} total)"
        if stop_reason == "max_attempts":
            print(f"Reached max_attempts={max_attempts}; saving partial dataset with {summary}.")
        elif stop_reason == "exhausted":
            stalled = [i for i in (0, 1) if exhausted[i]]
            print(
                f"Exhaustion detected: class(es) {stalled} produced no new unique "
                f"mosaic in {stall:,}+ attempts (dim={dimension}, D_n_upper="
                f"{_mosaic_count_upper(dimension):,}). Saving partial dataset with {summary}."
            )

        _save_rows(rows, csv_path, label_col)

        if stop_reason == "max_attempts" and min(counts) == 0:
            raise ValueError(
                f"Could not generate both classes after {max_attempts} attempts; "
                f"counts={counts}"
            )


def generate(data_type, num_samples, dimension, workers=None, progress=True):
    """Generate a balanced dataset for ``data_type``.

    ``workers`` controls candidate-generation parallelism: ``None`` uses the
    smart per-type default, ``0`` forces all CPU cores, ``1`` is serial, and any
    other value uses exactly that many worker processes. See _resolve_workers.
    """
    workers = _resolve_workers(data_type, workers)
    spec = _DATASET_SPECS[data_type]
    _generate_balanced(
        num_samples=num_samples,
        data_type=data_type,
        dimension=dimension,
        label_col=spec["label_col"],
        csv_path=dataset_path(data_type, dimension),
        legacy_csv_path=_legacy_dataset_path(data_type),
        workers=workers,
        progress=progress,
    )


# Backwards-compatible named entry points (preserve the original call sites,
# e.g. multithreading-data-gen.py and the Makefile-driven CLI dispatch).
def generate_unknot_data(num_samples, dimension, workers=1):
    generate("unknot", num_samples, dimension, workers=workers)


def generate_suitably_connected_data(num_samples, dimension, workers=1):
    generate("connected", num_samples, dimension, workers=workers)


def generate_has_crossing_data(num_samples, dimension, workers=1):
    generate("has_crossing", num_samples, dimension, workers=workers)


def generate_component_data(num_samples, dimension, workers=1):
    generate("component", num_samples, dimension, workers=workers)


def generate_trefoil_vs_unknot_data(num_samples, dimension, workers=1):
    generate("trefoil", num_samples, dimension, workers=workers)


_GENERATORS = {
    "unknot": generate_unknot_data,
    "connected": generate_suitably_connected_data,
    "has_crossing": generate_has_crossing_data,
    "component": generate_component_data,
    "trefoil": generate_trefoil_vs_unknot_data,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_type", type=str, choices=sorted(_GENERATORS))
    parser.add_argument("num_samples", type=int)
    parser.add_argument("dimension", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="candidate-generation worker processes (default: all CPU cores; 1 = serial).",
    )
    args = parser.parse_args()

    generate(args.data_type, args.num_samples, args.dimension, workers=args.workers)




#   1. Dataset size sweep

#   For the same task and dimension:

#   python data_gen.py component 1000 10
#   python cnn_train.py component 50

#   python data_gen.py component 3000 10
#   python cnn_train.py component 50

#   python data_gen.py component 10000 10
#   python cnn_train.py component 50

#   Compare best validation loss epoch, test accuracy, and overfitting gap.

#   2. Dimension sweep

#   Same task, same sample count:

#   python data_gen.py component 3000 5
#   python cnn_train.py component 50

#   python data_gen.py component 3000 10
#   python cnn_train.py component 50

#   python data_gen.py component 3000 15
#   python cnn_train.py component 50

#   This tells you whether larger mosaics are harder or easier.

#   3. Task sweep

#   Train each label type separately:

#   python data_gen.py connected 3000 10
#   python cnn_train.py connected 50

#   python data_gen.py has_crossing 3000 10
#   python cnn_train.py has_crossing 50

#   python data_gen.py component 3000 10
#   python cnn_train.py component 50

#   Some tasks may be almost local, like “has crossing,” while others require global topology.

#   4. Generalization test

#   Train on one dimension, test mentally/externally against another. Eventually you should add explicit cross-dimension
#   evaluation, but for now compare cached runs:

#   train component 3000 5
#   train component 3000 10
#   train component 3000 15

#   If 5x5 performs well but 15x15 collapses, the CNN may not be capturing enough global structure.

#   5. Repeat same run with different seeds

#   Same data settings, different train/test splits:

#   train_ccnn("component_data.csv", epochs=50, seed=1)
#   train_ccnn("component_data.csv", epochs=50, seed=2)
#   train_ccnn("component_data.csv", epochs=50, seed=3)

#   If accuracy swings a lot, your dataset/model is unstable.

#   6. Look at false positives and false negatives

#   Use training_test_predictions.csv. The most useful examples are:

#   - high-confidence wrong positives: actual 0, predicted 1, p_label_1 > 0.9
#   - high-confidence wrong negatives: actual 1, predicted 0, p_label_1 < 0.1

#   Those tell you what patterns the CNN misunderstands.
