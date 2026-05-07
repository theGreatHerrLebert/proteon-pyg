"""Infrastructure demo: does the batch helper actually parallelize?

Times ``proteon_pyg_data_batch`` against a Python loop calling
``proteon_pyg_data`` per file, at several thread counts. Plots a speedup
curve so you can see where the rayon parallelism pays off and where it
saturates.

This is not a benchmark of proteon's own kernels (proteon ships its own
oracle benchmarks). It's a benchmark of the proteon-pyg integration:
"does running through the adapter destroy proteon's parallelism?".
Expected answer: no, the adapter is a thin tensor transcription layer
on top of proteon's batch primitives.

Run:
    /scratch/TMAlign/proteon/.venv/bin/python demos/infra_demo.py
    /scratch/TMAlign/proteon/.venv/bin/python demos/infra_demo.py --n 200

Outputs land in ``demos/output/infra/``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from proteon_pyg import proteon_pyg_data, proteon_pyg_data_batch


CORPUS_DIR = Path("/scratch/TMAlign/proteon/validation/pdbs_10k")
OUTPUT_DIR = Path(__file__).parent / "output" / "infra"


def _pick_paths(n: int) -> list[Path]:
    if not CORPUS_DIR.exists():
        print(
            f"[infra] corpus missing: {CORPUS_DIR}. Skipping demo cleanly.",
            file=sys.stderr,
        )
        sys.exit(0)
    files = sorted(CORPUS_DIR.glob("*.pdb"))
    if len(files) < n:
        print(
            f"[infra] only {len(files)} PDBs available, requested {n}; using all of them"
        )
        return files
    # Stride so we sample diverse structures rather than the alphabetical head.
    stride = len(files) // n
    return [files[i * stride] for i in range(n)]


def _filter_loadable(paths: list[Path]) -> tuple[list[Path], int]:
    """Drop PDBs that proteon-pyg cannot currently process.

    The 10K-PDB validation corpus contains structures where proteon's
    dssp() output length disagrees with structure.residues' AA count
    (a proteon-side discrepancy, not a proteon-pyg bug). We skip those
    here so the timing demo measures the throughput of the happy path
    rather than aborting on the first edge case.
    """
    ok: list[Path] = []
    skipped = 0
    for p in paths:
        try:
            _ = proteon_pyg_data(p, energy=False, dssp=True, hbond_count=False, dihedrals=False)
            ok.append(p)
        except (RuntimeError, ValueError, OSError, Exception):  # noqa: BLE001
            # Catch broadly: proteon raises OSError on SEQRES mismatches,
            # RuntimeError on dssp/AA-count mismatches, and various others.
            skipped += 1
    return ok, skipped


def _time_loop(paths: list[Path], reps: int) -> float:
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        for p in paths:
            _ = proteon_pyg_data(p, energy=False, hbond_count=True, dihedrals=True)
        samples.append(time.perf_counter() - t)
    return mean(samples)


def _time_batch(paths: list[Path], n_threads: int, reps: int) -> float:
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        _ = proteon_pyg_data_batch(
            paths,
            energy=False,
            hbond_count=True,
            dihedrals=True,
            n_threads=n_threads,
        )
        samples.append(time.perf_counter() - t)
    return mean(samples)


def _plot_speedup(
    n: int,
    loop_time: float,
    thread_counts: list[int],
    batch_times: list[float],
    out: Path,
) -> None:
    speedups = [loop_time / t for t in batch_times]
    throughput = [n / t for t in batch_times]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(thread_counts, speedups, "o-")
    axes[0].axhline(1.0, color="0.7", linestyle="--", label="loop baseline")
    axes[0].set_xlabel("n_threads")
    axes[0].set_ylabel(f"speedup vs single-call loop ({loop_time:.2f}s)")
    axes[0].set_title(f"Batch-vs-loop speedup (N = {n} PDBs)")
    axes[0].legend()

    axes[1].plot(thread_counts, throughput, "o-", color="C1")
    axes[1].set_xlabel("n_threads")
    axes[1].set_ylabel("throughput (PDBs / sec)")
    axes[1].set_title("Batch throughput")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="number of PDBs to time")
    parser.add_argument(
        "--reps", type=int, default=2, help="repeats per condition (averaged)"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_paths = _pick_paths(args.n)
    n_cores = os.cpu_count() or 16
    thread_counts = sorted({1, 4, 8, n_cores})

    print(f"[infra] sampled {len(raw_paths)} PDBs from corpus")
    print("[infra] filtering to loadable subset...")
    paths, skipped = _filter_loadable(raw_paths)
    n = len(paths)
    if skipped:
        print(
            f"[infra]   skipped {skipped}/{len(raw_paths)} that hit proteon DSSP / "
            "AA-count mismatches (proteon-side discrepancy, not a proteon-pyg bug)"
        )
    if n == 0:
        print("[infra] no loadable PDBs left, aborting", file=sys.stderr)
        return 1

    print(f"[infra] timing {n} PDBs, {args.reps} reps per condition")
    print(f"[infra] thread counts: {thread_counts} (machine has {n_cores} cores)")

    print("[infra] warmup...")
    _ = proteon_pyg_data_batch(
        paths[: min(8, n)],
        energy=False,
        hbond_count=True,
        dihedrals=True,
        n_threads=1,
    )

    print(f"[infra] timing single-call loop ({args.reps} reps)...")
    loop_time = _time_loop(paths, args.reps)
    print(f"[infra]   loop:           {loop_time:6.2f}s  ({n / loop_time:5.1f} PDBs/s)")

    batch_times: list[float] = []
    for nt in thread_counts:
        t = _time_batch(paths, nt, args.reps)
        batch_times.append(t)
        speedup = loop_time / t
        print(
            f"[infra]   batch n={nt:>2}: {t:6.2f}s  ({n / t:5.1f} PDBs/s, speedup x{speedup:.2f})"
        )

    _plot_speedup(n, loop_time, thread_counts, batch_times, OUTPUT_DIR / "speedup.png")
    print(f"[infra] plot in {OUTPUT_DIR}/speedup.png")

    # Summary ratios — useful for the commit message.
    best = min(batch_times)
    print(
        f"[infra] best speedup: x{loop_time / best:.2f} "
        f"at n_threads={thread_counts[batch_times.index(best)]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
