#!/usr/bin/env bash
# fetch_benchmarks.sh — Download ISCAS'85 and ISCAS'89 benchmark files.
#
# Usage:
#   bash scripts/fetch_benchmarks.sh [--dir benchmarks/circuits] [--dry-run]
#
# Requires: curl or wget

set -euo pipefail

DIR="benchmarks/circuits"
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --dir=*)   DIR="${arg#*=}" ;;
    --dir)     shift; DIR="$1" ;;
    --dry-run) DRY_RUN=true ;;
  esac
done

mkdir -p "$DIR"

BASE_BENCH="https://sportlab.usc.edu/~msabrishami/files/bench"
BASE_VERILOG="https://sportlab.usc.edu/~msabrishami/files/verilog"

# ISCAS'85 combinational circuits
ISCAS85_BENCH=(
  c17 c432 c499 c880
  c1355 c1908 c2670 c3540
  c5315 c6288 c7552
)

# ISCAS'89 sequential circuits  (BENCH format)
ISCAS89_BENCH=(
  s27 s298 s344 s349 s382 s386 s400
  s444 s510 s641 s713 s820 s832 s953
  s1196 s1238 s1432 s1488 s1494
  s5378 s9234 s13207 s15850 s35932 s38417 s38584
)

# ISCAS'89 sequential circuits  (Verilog format)
ISCAS89_VERILOG=(
  s420
)

_download() {
  local url="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    echo "  [skip]  $dest (already present)"
    return
  fi
  if $DRY_RUN; then
    echo "  [dry]   $url -> $dest"
    return
  fi
  if command -v curl &>/dev/null; then
    curl -sSL --retry 3 --retry-delay 2 -o "$dest" "$url" && \
      echo "  [ok]    $dest" || \
      echo "  [fail]  $url"
  elif command -v wget &>/dev/null; then
    wget -q --tries=3 --waitretry=2 -O "$dest" "$url" && \
      echo "  [ok]    $dest" || \
      echo "  [fail]  $url"
  else
    echo "  [error] Neither curl nor wget found." >&2
    exit 1
  fi
}

echo "=== Downloading ISCAS'85 BENCH files ==="
for c in "${ISCAS85_BENCH[@]}"; do
  _download "${BASE_BENCH}/${c}.bench" "${DIR}/${c}.bench"
done

echo ""
echo "=== Downloading ISCAS'89 BENCH files ==="
for c in "${ISCAS89_BENCH[@]}"; do
  _download "${BASE_BENCH}/${c}.bench" "${DIR}/${c}.bench"
done

echo ""
echo "=== Downloading ISCAS'89 Verilog files ==="
for c in "${ISCAS89_VERILOG[@]}"; do
  _download "${BASE_VERILOG}/${c}.v" "${DIR}/${c}.v"
done

echo ""
echo "Done. Files saved to: $DIR"
