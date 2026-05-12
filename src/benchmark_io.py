"""
Benchmark parsers for ISCAS BENCH, Verilog, and ISPD98 HGR (hMETIS) formats.

All parsers return a Hypergraph instance using the common representation:
  - vertices  = gates / cells / primary IOs
  - hyperedges = nets  (driver + all sinks)
"""

from __future__ import annotations

import re
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from objective import Hypergraph


# ---------------------------------------------------------------------------
# BENCH format  (ISCAS'85 and ISCAS'89)
# ---------------------------------------------------------------------------

# Gate types recognised in standard ISCAS BENCH files
_BENCH_GATE_TYPES = {
    "AND", "NAND", "OR", "NOR", "NOT", "BUFF", "BUF",
    "XOR", "XNOR", "DFF",                        # DFF = D flip-flop (ISCAS'89)
    "INPUT", "OUTPUT",
}

_GATE_LINE_RE = re.compile(
    r"^(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)\s*$"
)


def parse_bench(path: str) -> Hypergraph:
    """
    Parse an ISCAS BENCH file and return a Hypergraph.

    Signal graph:
      - Each named signal (primary input or gate output) → one vertex.
      - Each signal's driven-fan-in forms one hyperedge:
          {driver_vertex}  ∪  {all vertices that take this signal as input}
    """
    path = Path(path)
    text = path.read_text(errors="replace")

    primary_inputs: List[str] = []
    primary_outputs: List[str] = []
    gate_defs: Dict[str, Tuple[str, List[str]]] = {}  # output → (type, inputs)

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # PRIMARY INPUT / OUTPUT declarations
        m_io = re.match(r"^(INPUT|OUTPUT)\s*\(\s*(\w+)\s*\)\s*$", line, re.IGNORECASE)
        if m_io:
            kind, sig = m_io.group(1).upper(), m_io.group(2)
            if kind == "INPUT":
                primary_inputs.append(sig)
            else:
                primary_outputs.append(sig)
            continue

        # Gate assignment:  out = TYPE(in1, in2, ...)
        m_gate = _GATE_LINE_RE.match(line)
        if m_gate:
            out_sig = m_gate.group(1)
            gate_type = m_gate.group(2).upper()
            in_sigs = [s.strip() for s in m_gate.group(3).split(",") if s.strip()]
            gate_defs[out_sig] = (gate_type, in_sigs)

    # -----------------------------------------------------------------------
    # Build vertex index  (primary inputs first, then gates in definition order)
    # -----------------------------------------------------------------------
    all_signals: List[str] = list(primary_inputs)
    for sig in gate_defs:
        if sig not in all_signals:
            all_signals.append(sig)

    sig_to_idx = {s: i for i, s in enumerate(all_signals)}
    n = len(all_signals)

    # -----------------------------------------------------------------------
    # Build hyperedges:
    #   For each signal X, create one hyperedge containing:
    #     • the vertex for X  (the driver)
    #     • every vertex whose gate uses X as an input  (the sinks)
    # -----------------------------------------------------------------------
    # Build a fanout map: signal → list of gate output signals that consume it
    fanout: Dict[str, List[str]] = {s: [] for s in all_signals}
    for out_sig, (_, inputs) in gate_defs.items():
        for in_sig in inputs:
            if in_sig in fanout:
                fanout[in_sig].append(out_sig)

    hyperedges: List[List[int]] = []
    for sig in all_signals:
        sinks = fanout.get(sig, [])
        if not sinks:
            continue  # dangling signal — skip
        edge = [sig_to_idx[sig]] + [sig_to_idx[s] for s in sinks if s in sig_to_idx]
        if len(edge) >= 2:
            hyperedges.append(edge)

    return Hypergraph(
        num_vertices=n,
        num_hyperedges=len(hyperedges),
        hyperedges=hyperedges,
        vertex_names=all_signals,
    )


# ---------------------------------------------------------------------------
# Verilog format  (structural gate-level — ISCAS'89 variant)
# ---------------------------------------------------------------------------

_VERILOG_MODULE_RE = re.compile(r"\bmodule\s+(\w+)", re.IGNORECASE)
_VERILOG_WIRE_RE = re.compile(r"\bwire\b([^;]+);", re.IGNORECASE)
_VERILOG_INPUT_RE = re.compile(r"\binput\b([^;]+);", re.IGNORECASE)
_VERILOG_OUTPUT_RE = re.compile(r"\boutput\b([^;]+);", re.IGNORECASE)
# e.g.  and  g1 ( y, a, b );   or   DFF  ff1 ( .Q(q), .D(d), .CK(clk) );
_VERILOG_INST_RE = re.compile(
    r"(\w+)\s+(\w+)\s*\(([^)]*)\)\s*;", re.IGNORECASE
)


def parse_verilog(path: str) -> Hypergraph:
    """
    Parse a structural (gate-level) Verilog file and return a Hypergraph.

    Handles both positional port lists and named port lists (.port(signal)).
    Primitive gates: and, or, nand, nor, not, buf, xor, xnor, dff variants.
    """
    path = Path(path)
    text = path.read_text(errors="replace")
    # Strip line comments
    text = re.sub(r"//[^\n]*", "", text)
    # Strip block comments
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)

    wires: List[str] = []
    inputs: List[str] = []
    outputs: List[str] = []
    instances: List[Tuple[str, str, List[str]]] = []  # (type, inst_name, ports)

    for m in _VERILOG_INPUT_RE.finditer(text):
        inputs += [s.strip() for s in m.group(1).split(",") if s.strip()]
    for m in _VERILOG_OUTPUT_RE.finditer(text):
        outputs += [s.strip() for s in m.group(1).split(",") if s.strip()]
    for m in _VERILOG_WIRE_RE.finditer(text):
        wires += [s.strip() for s in m.group(1).split(",") if s.strip()]

    primitive_types = {
        "and", "nand", "or", "nor", "not", "buf", "buff",
        "xor", "xnor", "dff",
    }

    for m in _VERILOG_INST_RE.finditer(text):
        gtype = m.group(1).lower()
        inst_name = m.group(2)
        port_str = m.group(3).strip()
        if gtype in ("module", "endmodule", "input", "output", "wire", "reg"):
            continue

        # Resolve named (.port(sig)) or positional port lists
        named = re.findall(r"\.\w+\s*\((\w+)\)", port_str)
        if named:
            ports = named
        else:
            ports = [s.strip() for s in port_str.split(",") if s.strip()]

        instances.append((gtype, inst_name, ports))

    # Collect all signals as vertices
    all_signals = list(dict.fromkeys(inputs + wires + outputs))
    sig_to_idx = {s: i for i, s in enumerate(all_signals)}
    n = len(all_signals)

    # Build hyperedges: each instance becomes a hyperedge over its port signals
    hyperedges: List[List[int]] = []
    for _, _, ports in instances:
        edge = [sig_to_idx[p] for p in ports if p in sig_to_idx]
        if len(edge) >= 2:
            hyperedges.append(edge)

    return Hypergraph(
        num_vertices=n,
        num_hyperedges=len(hyperedges),
        hyperedges=hyperedges,
        vertex_names=all_signals,
    )


# ---------------------------------------------------------------------------
# hMETIS / ISPD98 HGR format
# ---------------------------------------------------------------------------

def parse_hgr(path: str) -> Hypergraph:
    """
    Parse an hMETIS hypergraph (.hgr) file.

    Supported fmt flags (from first-line descriptor):
      0  — no weights
      1  — hyperedge weights only
      10 — vertex weights only
      11 — both weights
    """
    path = Path(path)
    lines = [l.rstrip() for l in path.read_text(errors="replace").splitlines()]

    # Skip comment lines (start with %)
    lines = [l for l in lines if not l.startswith("%")]

    if not lines:
        raise ValueError(f"Empty HGR file: {path}")

    header = lines[0].split()
    num_he = int(header[0])
    num_v = int(header[1])
    fmt = int(header[2]) if len(header) >= 3 else 0

    has_he_weights = fmt in (1, 11)
    has_v_weights = fmt in (10, 11)

    hyperedges: List[List[int]] = []
    he_weights: List[float] = []

    for i in range(1, num_he + 1):
        if i >= len(lines):
            break
        tokens = lines[i].split()
        if not tokens:
            hyperedges.append([])
            he_weights.append(1.0)
            continue
        if has_he_weights:
            he_weights.append(float(tokens[0]))
            verts = [int(t) - 1 for t in tokens[1:]]  # 1-indexed → 0-indexed
        else:
            he_weights.append(1.0)
            verts = [int(t) - 1 for t in tokens]
        hyperedges.append(verts)

    v_weights: List[float] = [1.0] * num_v
    if has_v_weights:
        for j in range(num_v):
            idx = num_he + 1 + j
            if idx < len(lines):
                v_weights[j] = float(lines[idx].split()[0])

    return Hypergraph(
        num_vertices=num_v,
        num_hyperedges=num_he,
        hyperedges=hyperedges,
        vertex_weights=v_weights,
        hyperedge_weights=he_weights,
    )


# ---------------------------------------------------------------------------
# Dispatch by file extension / format string
# ---------------------------------------------------------------------------

def load_benchmark(path: str, fmt: Optional[str] = None) -> Hypergraph:
    """
    Load a benchmark file into a Hypergraph.

    fmt can be 'bench', 'verilog', 'hgr', or None (auto-detect from extension).
    """
    ext = Path(path).suffix.lower()
    if fmt is None:
        if ext == ".bench":
            fmt = "bench"
        elif ext in (".v", ".verilog"):
            fmt = "verilog"
        elif ext in (".hgr", ".hg"):
            fmt = "hgr"
        else:
            raise ValueError(f"Cannot auto-detect format for extension '{ext}'")

    if fmt == "bench":
        return parse_bench(path)
    elif fmt == "verilog":
        return parse_verilog(path)
    elif fmt == "hgr":
        return parse_hgr(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")


# ---------------------------------------------------------------------------
# Benchmark manifest loader
# ---------------------------------------------------------------------------

def load_manifest(csv_path: str) -> List[Dict]:
    """Return list of dicts from benchmark_manifest.csv."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def apply_global_net_threshold(hg: Hypergraph, threshold: int) -> Hypergraph:
    """
    Remove hyperedges larger than threshold (mirrors -global_net_threshold).
    Returns a new Hypergraph.
    """
    filtered_he = []
    filtered_hw = []
    for i, he in enumerate(hg.hyperedges):
        if len(he) <= threshold:
            filtered_he.append(he)
            filtered_hw.append(hg.hyperedge_weights[i])

    return Hypergraph(
        num_vertices=hg.num_vertices,
        num_hyperedges=len(filtered_he),
        hyperedges=filtered_he,
        vertex_weights=list(hg.vertex_weights),
        hyperedge_weights=filtered_hw,
        vertex_names=list(hg.vertex_names),
        vertex_coords=hg.vertex_coords,
    )
