"""
Reporting utilities: CSV result tables, convergence plots, cut comparison charts.

Figure rules (publication-ready):
  - White background
  - Non-overlapping labels (adjustText where needed)
  - Orthogonal arrows in pipeline diagrams
  - Equal spacing between boxes
  - SVG first, PNG on request
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Matplotlib is imported lazily so the module can be imported without a display.
_ALGORITHMS = ["ABC", "KH", "MBO", "EWA", "EHO", "MS", "SMA", "HHO"]
_ALGO_COLORS = {
    "ABC": "#1f77b4",
    "KH":  "#ff7f0e",
    "MBO": "#2ca02c",
    "EWA": "#d62728",
    "EHO": "#9467bd",
    "MS":  "#8c564b",
    "SMA": "#e377c2",
    "HHO": "#7f7f7f",
}


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "benchmark", "family", "algorithm", "num_parts",
    "balance_constraint", "seed", "cutsize", "runtime_sec", "feasible", "notes",
]


def append_result(row: Dict, csv_path: str) -> None:
    """Append one result row to the CSV, creating it with headers if absent."""
    path = Path(csv_path)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_results(csv_path: str) -> List[Dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def summarise_results(rows: List[Dict]) -> Dict:
    """
    Return a per-(benchmark, algorithm) summary:
      mean cutsize, std cutsize, min cutsize, feasibility rate.
    """
    from collections import defaultdict
    groups: Dict = defaultdict(list)
    for r in rows:
        key = (r["benchmark"], r["algorithm"])
        try:
            groups[key].append(float(r["cutsize"]))
        except (ValueError, KeyError):
            pass

    summary = {}
    for key, cuts in groups.items():
        a = np.array(cuts)
        summary[key] = {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "n_runs": len(a),
        }
    return summary


# ---------------------------------------------------------------------------
# Convergence plots
# ---------------------------------------------------------------------------

def plot_convergence(
    histories: Dict[str, List[float]],
    title: str = "Convergence",
    output_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot convergence curves for multiple algorithms.

    Parameters
    ----------
    histories  : {algorithm_name: [best_obj_per_iteration]}
    title      : plot title
    output_path: if given, save SVG (and PNG if path ends with .png)
    show       : call plt.show()
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")

    for algo, hist in histories.items():
        color = _ALGO_COLORS.get(algo, None)
        ax.plot(hist, label=algo, color=color, linewidth=1.6)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Objective (Cut)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if output_path:
        svg_path = str(output_path)
        if not svg_path.endswith(".svg"):
            svg_path = svg_path.rsplit(".", 1)[0] + ".svg"
        fig.savefig(svg_path, format="svg", dpi=150, bbox_inches="tight",
                    facecolor="white")
        if output_path.endswith(".png"):
            fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight",
                        facecolor="white")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cut comparison bar chart
# ---------------------------------------------------------------------------

def plot_cut_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "mean",
    title: str = "Cut Size Comparison",
    output_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Grouped bar chart: benchmarks on x-axis, algorithms as groups.

    Parameters
    ----------
    results : {benchmark: {algorithm: value}}
    metric  : label for the y-axis
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    benchmarks = sorted(results.keys())
    algorithms = sorted({a for v in results.values() for a in v.keys()})
    n_bench = len(benchmarks)
    n_algo = len(algorithms)

    x = np.arange(n_bench)
    bar_width = 0.8 / max(n_algo, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_bench * 1.2), 5), facecolor="white")
    ax.set_facecolor("white")

    for j, algo in enumerate(algorithms):
        vals = [results[b].get(algo, float("nan")) for b in benchmarks]
        offset = (j - n_algo / 2.0 + 0.5) * bar_width
        color = _ALGO_COLORS.get(algo, None)
        ax.bar(x + offset, vals, width=bar_width * 0.9,
               label=algo, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(f"Cut Size ({metric})", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right",
              ncol=max(1, n_algo // 4))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()

    if output_path:
        svg_path = str(output_path)
        if not svg_path.endswith(".svg"):
            svg_path = svg_path.rsplit(".", 1)[0] + ".svg"
        fig.savefig(svg_path, format="svg", dpi=150, bbox_inches="tight",
                    facecolor="white")
        if output_path.endswith(".png"):
            fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight",
                        facecolor="white")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-benchmark summary table (terminal / Markdown)
# ---------------------------------------------------------------------------

def print_summary_table(summary: Dict, top_k: int = 5) -> None:
    """Print a Markdown-style summary table to stdout."""
    header = f"{'Benchmark':<14} {'Algorithm':<8} {'Mean Cut':>10} {'Std':>8} {'Best':>8} {'Runs':>5}"
    print(header)
    print("-" * len(header))

    items = sorted(summary.items(), key=lambda x: x[1]["mean"])
    for (bench, algo), stats in items[:top_k]:
        print(
            f"{bench:<14} {algo:<8} "
            f"{stats['mean']:>10.1f} {stats['std']:>8.1f} "
            f"{stats['min']:>8.1f} {stats['n_runs']:>5}"
        )


# ---------------------------------------------------------------------------
# Generate all figures from a results CSV
# ---------------------------------------------------------------------------

def generate_all_figures(
    csv_path: str,
    output_dir: str = "docs/figures",
    benchmark_filter: Optional[List[str]] = None,
) -> None:
    """
    Load results CSV and produce:
      - cut_comparison.svg  per family (ISCAS85, ISCAS89, ISPD98)
      - convergence_<benchmark>.svg  for each benchmark in benchmark_filter
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = load_results(csv_path)

    families = {r["family"] for r in rows}
    for family in families:
        fam_rows = [r for r in rows if r["family"] == family]
        by_bench_algo: Dict[str, Dict[str, float]] = {}
        for r in fam_rows:
            b = r["benchmark"]
            a = r["algorithm"]
            try:
                cut = float(r["cutsize"])
            except (ValueError, KeyError):
                continue
            by_bench_algo.setdefault(b, {}).setdefault(a, []).append(cut)

        # Take mean per benchmark/algorithm
        mean_cut = {
            b: {a: float(np.mean(v)) for a, v in algos.items()}
            for b, algos in by_bench_algo.items()
        }

        out = os.path.join(output_dir, f"cut_comparison_{family.lower()}.svg")
        plot_cut_comparison(
            mean_cut,
            title=f"Cut Size Comparison — {family}",
            output_path=out,
        )
        print(f"Saved: {out}")
