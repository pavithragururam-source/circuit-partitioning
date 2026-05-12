#!/usr/bin/env python3
"""
VLSI Circuit Partitioning Framework — CLI entry point.

Interface mirrors OpenROAD/TritonPart parameters so results are directly
comparable with the baseline. Run `python main.py --help` for usage.

Example
-------
python main.py \\
    --benchmark benchmarks/c432.bench \\
    --algorithm ABC HHO \\
    --num-parts 2 \\
    --balance-constraint 0.1 \\
    --seeds 0 1 2 \\
    --results-csv results/run.csv \\
    --figures-dir docs/figures
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import traceback
from pathlib import Path
from typing import List

import numpy as np

# Allow running from src/ or from project root
sys.path.insert(0, os.path.dirname(__file__))

from objective import PartitionConfig
from benchmark_io import load_benchmark, load_manifest, apply_global_net_threshold
from openroad_adapter import OpenROADAdapter, write_solution_file
from reporting import append_result, generate_all_figures, print_summary_table, summarise_results, load_results
from optimizers import get_optimizer, REGISTRY


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vlsi-partition",
        description="Metaheuristic VLSI circuit partitioning (OpenROAD-aligned)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Benchmark input
    p.add_argument("--benchmark", "-b", metavar="FILE",
                   help="Path to a .bench, .v, or .hgr benchmark file.")
    p.add_argument("--manifest", metavar="CSV",
                   help="Run all benchmarks listed in a manifest CSV "
                        "(overrides --benchmark).")
    p.add_argument("--family", metavar="NAME",
                   help="Override family tag in result rows (e.g. ISCAS85).")

    # Partitioning parameters (aligned to OpenROAD)
    p.add_argument("--num-parts", type=int, default=2, metavar="K",
                   help="Number of partitions (-num_parts).")
    p.add_argument("--balance-constraint", type=float, default=0.10,
                   metavar="B",
                   help="Allowed imbalance ratio (-balance_constraint).")
    p.add_argument("--timing-aware", action="store_true",
                   help="Enable timing-aware mode (-timing_aware_flag).")
    p.add_argument("--placement-file", metavar="FILE", default="",
                   help="Placement embedding file (-placement_file).")
    p.add_argument("--global-net-threshold", type=int, default=1000,
                   metavar="T",
                   help="Skip hyperedges larger than this (-global_net_threshold).")

    # Objective weights
    p.add_argument("--w-cut",      type=float, default=1.0)
    p.add_argument("--w-balance",  type=float, default=10.0)
    p.add_argument("--w-timing",   type=float, default=0.0)
    p.add_argument("--w-placement",type=float, default=0.0)

    # Optimiser selection
    p.add_argument("--algorithm", "-a", nargs="+",
                   choices=list(REGISTRY.keys()) + ["ALL"],
                   default=["HHO"],
                   metavar="ALG",
                   help=f"Algorithms to run. Choices: {list(REGISTRY)} or ALL.")
    p.add_argument("--pop-size",  type=int, default=30)
    p.add_argument("--max-iter",  type=int, default=200)
    p.add_argument("--seeds",     type=int, nargs="+", default=[0],
                   metavar="S",
                   help="Random seeds (one run per seed).")

    # Output
    p.add_argument("--results-csv", default="results/run.csv", metavar="CSV",
                   help="Append results to this CSV file.")
    p.add_argument("--solution-dir", default="results/solutions", metavar="DIR",
                   help="Directory to write per-run .part solution files.")
    p.add_argument("--figures-dir",  default="docs/figures", metavar="DIR",
                   help="Directory to save generated figures.")
    p.add_argument("--no-figures",   action="store_true",
                   help="Skip figure generation.")

    # OpenROAD baseline
    p.add_argument("--openroad-baseline", action="store_true",
                   help="Also run TritonPart baseline (requires OpenROAD in PATH).")
    p.add_argument("--openroad-bin", default="openroad", metavar="BIN",
                   help="Path to the OpenROAD binary.")

    # Verbosity
    p.add_argument("--verbose", "-v", action="store_true")

    return p


# ---------------------------------------------------------------------------
# Core run logic
# ---------------------------------------------------------------------------

def run_single(
    benchmark_path: str,
    family: str,
    algorithm: str,
    cfg: PartitionConfig,
    pop_size: int,
    max_iter: int,
    seed: int,
    results_csv: str,
    solution_dir: str,
    verbose: bool = False,
) -> dict:
    """Run one (benchmark, algorithm, seed) experiment and return the result dict."""
    from benchmark_io import load_benchmark, apply_global_net_threshold

    hg = load_benchmark(benchmark_path)
    hg = apply_global_net_threshold(hg, cfg.global_net_threshold)
    circuit_name = Path(benchmark_path).stem

    opt = get_optimizer(algorithm, pop_size=pop_size, max_iter=max_iter)

    t0 = time.perf_counter()
    partition, history = opt.optimize(hg, cfg, seed=seed)
    runtime = time.perf_counter() - t0

    from objective import evaluate as obj_eval
    metrics = obj_eval(hg, partition, cfg)

    row = {
        "benchmark":          circuit_name,
        "family":             family,
        "algorithm":          algorithm,
        "num_parts":          cfg.num_parts,
        "balance_constraint": cfg.balance_constraint,
        "seed":               seed,
        "cutsize":            metrics["cutsize"],
        "runtime_sec":        round(runtime, 4),
        "feasible":           metrics["feasible"],
        "notes":              "",
    }
    append_result(row, results_csv)

    # Save solution file
    os.makedirs(solution_dir, exist_ok=True)
    sol_path = os.path.join(
        solution_dir, f"{circuit_name}_{algorithm}_k{cfg.num_parts}_s{seed}.part"
    )
    write_solution_file(partition, sol_path)

    if verbose:
        print(
            f"  [{algorithm}] {circuit_name} seed={seed} "
            f"cut={metrics['cutsize']:.0f} imbalance={metrics['imbalance']:.3f} "
            f"feasible={metrics['feasible']} t={runtime:.2f}s"
        )

    return row, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve algorithm list
    algorithms = list(REGISTRY.keys()) if "ALL" in args.algorithm else args.algorithm

    # Build PartitionConfig
    cfg = PartitionConfig(
        num_parts=args.num_parts,
        balance_constraint=args.balance_constraint,
        timing_aware=args.timing_aware,
        placement_aware=bool(args.placement_file),
        global_net_threshold=args.global_net_threshold,
        w_cut=args.w_cut,
        w_balance=args.w_balance,
        w_timing=args.w_timing,
        w_placement=args.w_placement,
    )

    # Collect benchmark paths
    if args.manifest:
        manifest = load_manifest(args.manifest)
        bench_list = [
            (row["source_url"] if row.get("status") == "available" else None,
             row.get("family", ""),
             row["circuit"])
            for row in manifest
        ]
        # Filter to locally available files only
        bench_entries = []
        for url, family, circuit in bench_list:
            ext = ".bench"
            local_path = os.path.join("benchmarks", circuit + ext)
            if os.path.exists(local_path):
                bench_entries.append((local_path, family or args.family or "UNKNOWN"))
            else:
                if args.verbose:
                    print(f"  [skip] {circuit} — not downloaded yet")
    elif args.benchmark:
        bench_entries = [(args.benchmark, args.family or "UNKNOWN")]
    else:
        parser.error("Provide --benchmark FILE or --manifest CSV")

    # OpenROAD baseline
    or_adapter = OpenROADAdapter(openroad_bin=args.openroad_bin)

    all_histories = {}   # {(bench, algo): history}
    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)

    for bench_path, family in bench_entries:
        circuit = Path(bench_path).stem
        print(f"\n=== {circuit} ({family}) ===")

        for algo in algorithms:
            for seed in args.seeds:
                try:
                    row, history = run_single(
                        benchmark_path=bench_path,
                        family=family,
                        algorithm=algo,
                        cfg=cfg,
                        pop_size=args.pop_size,
                        max_iter=args.max_iter,
                        seed=seed,
                        results_csv=args.results_csv,
                        solution_dir=args.solution_dir,
                        verbose=args.verbose,
                    )
                    key = (circuit, algo)
                    if key not in all_histories:
                        all_histories[key] = history
                except Exception as exc:
                    print(f"  [ERROR] {algo} seed={seed}: {exc}")
                    if args.verbose:
                        traceback.print_exc()

        # OpenROAD baseline (HGR only)
        if args.openroad_baseline and bench_path.endswith(".hgr"):
            try:
                result = or_adapter.triton_part_hypergraph(bench_path, cfg)
                row = {
                    "benchmark": circuit, "family": family,
                    "algorithm": "TritonPart",
                    "num_parts": cfg.num_parts,
                    "balance_constraint": cfg.balance_constraint,
                    "seed": 0,
                    "cutsize": result.get("cutsize", ""),
                    "runtime_sec": "",
                    "feasible": result.get("feasible", ""),
                    "notes": "OpenROAD baseline",
                }
                append_result(row, args.results_csv)
                if args.verbose:
                    print(f"  [TritonPart] cut={result.get('cutsize')}")
            except RuntimeError as e:
                print(f"  [TritonPart] skipped: {e}")

    # Summary
    if os.path.exists(args.results_csv):
        rows = load_results(args.results_csv)
        summary = summarise_results(rows)
        if summary:
            print("\n--- Summary (top 10 by mean cut) ---")
            print_summary_table(summary, top_k=10)

    # Figures
    if not args.no_figures and os.path.exists(args.results_csv):
        try:
            generate_all_figures(args.results_csv, output_dir=args.figures_dir)
        except Exception as exc:
            print(f"  [figures] skipped: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
