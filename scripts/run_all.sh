#!/usr/bin/env bash
# run_all.sh — Batch partition all downloaded benchmarks with all 8 algorithms.
#
# Usage:
#   bash scripts/run_all.sh [--k 2] [--seeds "0 1 2"] [--pop 30] [--iter 200]
#                           [--balance 0.1] [--results results/run.csv]
#                           [--figures docs/figures] [--dry-run]
#
# Prerequisites:
#   pip install numpy matplotlib pyyaml
#   bash scripts/fetch_benchmarks.sh

set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
K=2
SEEDS="0 1 2"
POP=30
ITER=200
BALANCE=0.10
RESULTS_CSV="results/run.csv"
FIGURES_DIR="docs/figures"
BENCH_DIR="benchmarks/circuits"
DRY_RUN=false
VERBOSE=false
ALGORITHMS="ABC KH MBO EWA EHO MS SMA HHO"

# ---- Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --k)         K="$2";          shift 2 ;;
    --seeds)     SEEDS="$2";      shift 2 ;;
    --pop)       POP="$2";        shift 2 ;;
    --iter)      ITER="$2";       shift 2 ;;
    --balance)   BALANCE="$2";    shift 2 ;;
    --results)   RESULTS_CSV="$2";shift 2 ;;
    --figures)   FIGURES_DIR="$2";shift 2 ;;
    --bench-dir) BENCH_DIR="$2";  shift 2 ;;
    --algorithms)ALGORITHMS="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=true;    shift   ;;
    --verbose)   VERBOSE=true;    shift   ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

SEED_ARGS=$(echo "$SEEDS" | tr ' ' '\n' | xargs)
ALGO_ARGS=$(echo "$ALGORITHMS")

# ---- Sanity checks ----------------------------------------------------------
if [[ ! -d "$BENCH_DIR" ]]; then
  echo "Benchmark directory '$BENCH_DIR' not found."
  echo "Run:  bash scripts/fetch_benchmarks.sh"
  exit 1
fi

BENCH_FILES=("$BENCH_DIR"/*.bench "$BENCH_DIR"/*.v "$BENCH_DIR"/*.hgr)
BENCH_FILES=("${BENCH_FILES[@]//*\*/}")   # remove glob non-matches

if [[ ${#BENCH_FILES[@]} -eq 0 ]]; then
  echo "No benchmark files found in $BENCH_DIR"
  exit 1
fi

mkdir -p "$(dirname "$RESULTS_CSV")" "$FIGURES_DIR"

# ---- Build Python command ----------------------------------------------------
PYTHON="${PYTHON:-python3}"
VERBOSE_FLAG=""
$VERBOSE && VERBOSE_FLAG="--verbose"

echo "======================================================"
echo " VLSI Partitioning Batch Run"
echo "  Benchmarks : $BENCH_DIR  (${#BENCH_FILES[@]} files)"
echo "  Algorithms : $ALGORITHMS"
echo "  k          : $K"
echo "  Seeds      : $SEEDS"
echo "  pop/iter   : $POP / $ITER"
echo "  balance    : $BALANCE"
echo "  Results    : $RESULTS_CSV"
echo "======================================================"

for BENCH in "${BENCH_FILES[@]}"; do
  [[ -f "$BENCH" ]] || continue
  CIRCUIT=$(basename "$BENCH")
  echo ""
  echo "--- $CIRCUIT ---"

  CMD=(
    "$PYTHON" src/main.py
    --benchmark "$BENCH"
    --algorithm $ALGO_ARGS
    --num-parts "$K"
    --balance-constraint "$BALANCE"
    --seeds $SEED_ARGS
    --pop-size "$POP"
    --max-iter "$ITER"
    --results-csv "$RESULTS_CSV"
    --figures-dir "$FIGURES_DIR"
    $VERBOSE_FLAG
  )

  if $DRY_RUN; then
    echo "  [dry] ${CMD[*]}"
  else
    "${CMD[@]}"
  fi
done

echo ""
echo "======================================================"
echo " Batch run complete. Results in: $RESULTS_CSV"
echo "======================================================"
