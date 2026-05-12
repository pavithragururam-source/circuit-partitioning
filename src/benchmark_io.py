"""
Benchmark parsers for ISCAS BENCH, Verilog, and ISPD98 HGR (hMETIS) formats.

Expert corrections vs original version
---------------------------------------
* parse_bench: Dangling primary-output vertices (no internal consumers) no
  longer silently dropped.  Each PO net receives a virtual external-port
  vertex so every cell participates in at least one hyperedge.
* parse_bench: Gate type stored in hg.gate_types for timing analysis.
* parse_bench: DFF (ISCAS'89 flip-flop) detected; hg.is_sequential set True;
  DFF Q→D feedback removed from combinational hyperedges to respect clock
  boundaries.
* parse_hgr: Fixed-vertex section supported (fmt bit 2 = has_fixed).
* apply_global_net_threshold: Returns hg with updated metadata intact.
* New: infer_vertex_areas — maps gate type → normalised area weight so balance
  is area-driven rather than cell-count-driven.
"""

from __future__ import annotations

import re
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from objective import Hypergraph


# ---------------------------------------------------------------------------
# Gate-type area table (normalised NAND-2 equivalent area)
# Source: typical standard-cell characterisation at 28 nm node
# ---------------------------------------------------------------------------
_GATE_AREA: Dict[str, float] = {
    "INPUT":  0.0,   # primary input pad — no logic area
    "OUTPUT": 0.0,   # primary output marker
    "BUFF":   1.0,
    "BUF":    1.0,
    "NOT":    0.67,
    "INV":    0.67,
    "NAND":   1.0,
    "AND":    1.33,
    "NOR":    1.0,
    "OR":     1.33,
    "XOR":    2.67,
    "XNOR":   2.67,
    "DFF":    6.00,   # D flip-flop — much larger than combinational gates
}

# Combinational gate delay in normalised units (FO4 = 1.0)
_GATE_DELAY: Dict[str, float] = {
    "INPUT":  0.0,
    "OUTPUT": 0.0,
    "BUFF":   0.5,
    "BUF":    0.5,
    "NOT":    0.5,
    "INV":    0.5,
    "NAND":   1.0,
    "AND":    1.5,
    "NOR":    1.0,
    "OR":     1.5,
    "XOR":    2.0,
    "XNOR":   2.0,
    "DFF":    0.0,   # DFF is a timing endpoint; delay modelled separately
}

_GATE_LINE_RE = re.compile(r"^(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)\s*$")


# ---------------------------------------------------------------------------
# BENCH format  (ISCAS'85 combinational and ISCAS'89 sequential)
# ---------------------------------------------------------------------------

def parse_bench(
    path: str,
    use_area_weights: bool = True,
) -> Hypergraph:
    """
    Parse an ISCAS BENCH file into a Hypergraph.

    Vertex model
    ~~~~~~~~~~~~
    - Each named signal (primary input or gate output) is one vertex.
    - Gate type stored in hg.gate_types[v].
    - Vertex weight = normalised gate area (if use_area_weights) else 1.0.

    Net / hyperedge model
    ~~~~~~~~~~~~~~~~~~~~~
    For each signal X the net is the hyperedge:
        {driver(X)} ∪ {all gates consuming X as input}

    Special handling
    ~~~~~~~~~~~~~~~~
    - PRIMARY OUTPUT signals with no internal consumers receive a virtual
      external-port vertex so they are not disconnected (dangling).
    - DFF Q→D feedback arcs are removed from combinational net lists to
      respect clock-cycle boundaries (combinational cone analysis).
    - hg.is_sequential = True when DFFs are present (ISCAS'89).
    """
    path = Path(path)
    text = path.read_text(errors="replace")

    primary_inputs:  List[str] = []
    primary_outputs: Set[str]  = set()
    gate_defs: Dict[str, Tuple[str, List[str]]] = {}   # out → (type, inputs)

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")):
            continue

        m_io = re.match(r"^(INPUT|OUTPUT)\s*\(\s*(\w+)\s*\)\s*$", line, re.IGNORECASE)
        if m_io:
            kind, sig = m_io.group(1).upper(), m_io.group(2)
            if kind == "INPUT":
                primary_inputs.append(sig)
            else:
                primary_outputs.add(sig)
            continue

        m_gate = _GATE_LINE_RE.match(line)
        if m_gate:
            out_sig   = m_gate.group(1)
            gate_type = m_gate.group(2).upper()
            in_sigs   = [s.strip() for s in m_gate.group(3).split(",") if s.strip()]
            gate_defs[out_sig] = (gate_type, in_sigs)

    # -----------------------------------------------------------------------
    # Build vertex list:  PIs first, then gates, then virtual external ports
    # -----------------------------------------------------------------------
    has_dff = any(gt == "DFF" for gt, _ in gate_defs.values())

    # DFF Q outputs: their D inputs are clock boundaries; remove D→Q feedback
    dff_q_outputs: Set[str] = {
        sig for sig, (gt, _) in gate_defs.items() if gt == "DFF"
    }

    all_signals: List[str] = list(primary_inputs)
    for sig in gate_defs:
        if sig not in all_signals:
            all_signals.append(sig)

    sig_to_idx = {s: i for i, s in enumerate(all_signals)}
    n_real = len(all_signals)

    # -----------------------------------------------------------------------
    # Build fanout map, skipping DFF D-input arcs (clock boundary)
    # -----------------------------------------------------------------------
    fanout: Dict[str, List[str]] = {s: [] for s in all_signals}
    for out_sig, (gate_type, inputs) in gate_defs.items():
        # DFF: only propagate the Q output through fanout; the D input is a
        # register boundary — do not create a combinational arc from D back to Q
        for in_sig in inputs:
            if gate_type == "DFF":
                continue   # DFF D-input is a sequential endpoint, not combinational
            if in_sig in fanout:
                fanout[in_sig].append(out_sig)

    # -----------------------------------------------------------------------
    # Build hyperedges; collect dangling POs for virtual-port treatment
    # -----------------------------------------------------------------------
    hyperedges: List[List[int]] = []
    virtual_port_signals: List[str] = []  # PO signals that need virtual ports

    for sig in all_signals:
        sinks = fanout.get(sig, [])

        if not sinks:
            # Signal has no internal consumers.
            # If it is a declared PRIMARY OUTPUT, create a virtual external port.
            if sig in primary_outputs:
                virtual_port_signals.append(sig)
            # else: truly dangling internal wire — skip (rare, malformed netlists)
            continue

        edge = [sig_to_idx[sig]] + [sig_to_idx[s] for s in sinks if s in sig_to_idx]
        if len(edge) >= 2:
            hyperedges.append(edge)

    # Add virtual external ports for dangling POs
    next_idx = n_real
    for po_sig in virtual_port_signals:
        # Virtual vertex represents the off-chip connection
        all_signals.append(f"__PORT_{po_sig}__")
        sig_to_idx[f"__PORT_{po_sig}__"] = next_idx
        hyperedges.append([sig_to_idx[po_sig], next_idx])
        next_idx += 1

    n_total = len(all_signals)

    # -----------------------------------------------------------------------
    # Vertex weights (gate area) and gate-type list
    # -----------------------------------------------------------------------
    vertex_weights: List[float] = []
    gate_types:     List[str]   = []

    for sig in all_signals:
        if sig.startswith("__PORT_"):
            vertex_weights.append(0.0)   # virtual port has zero area
            gate_types.append("PORT")
            continue
        if sig in gate_defs:
            gt = gate_defs[sig][0]
        elif sig in primary_inputs:
            gt = "INPUT"
        else:
            gt = "UNKNOWN"
        area = _GATE_AREA.get(gt, 1.0) if use_area_weights else 1.0
        vertex_weights.append(area)
        gate_types.append(gt)

    return Hypergraph(
        num_vertices=n_total,
        num_hyperedges=len(hyperedges),
        hyperedges=hyperedges,
        vertex_weights=vertex_weights,
        vertex_names=all_signals,
        gate_types=gate_types,
        is_sequential=has_dff,
    )


# ---------------------------------------------------------------------------
# Verilog format  (structural gate-level — ISCAS'89)
# ---------------------------------------------------------------------------

_VERILOG_WIRE_RE   = re.compile(r"\bwire\b([^;]+);",   re.IGNORECASE)
_VERILOG_INPUT_RE  = re.compile(r"\binput\b([^;]+);",  re.IGNORECASE)
_VERILOG_OUTPUT_RE = re.compile(r"\boutput\b([^;]+);", re.IGNORECASE)
_VERILOG_INST_RE   = re.compile(r"(\w+)\s+(\w+)\s*\(([^)]*)\)\s*;", re.IGNORECASE)


def parse_verilog(path: str) -> Hypergraph:
    """
    Parse a structural (gate-level) Verilog file into a Hypergraph.

    Handles both positional port lists and named port lists (.port(signal)).
    Each gate instance becomes one hyperedge connecting all its port signals.
    """
    path = Path(path)
    text = path.read_text(errors="replace")
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)

    inputs:  List[str] = []
    outputs: List[str] = []
    wires:   List[str] = []
    instances: List[Tuple[str, str, List[str]]] = []

    for m in _VERILOG_INPUT_RE.finditer(text):
        inputs  += [s.strip() for s in m.group(1).split(",") if s.strip()]
    for m in _VERILOG_OUTPUT_RE.finditer(text):
        outputs += [s.strip() for s in m.group(1).split(",") if s.strip()]
    for m in _VERILOG_WIRE_RE.finditer(text):
        wires   += [s.strip() for s in m.group(1).split(",") if s.strip()]

    skip_kw = {"module", "endmodule", "input", "output", "wire", "reg", "assign"}

    for m in _VERILOG_INST_RE.finditer(text):
        gtype = m.group(1).lower()
        if gtype in skip_kw:
            continue
        port_str = m.group(3).strip()
        named = re.findall(r"\.\w+\s*\((\w+)\)", port_str)
        ports = named if named else [s.strip() for s in port_str.split(",") if s.strip()]
        instances.append((m.group(1).upper(), m.group(2), ports))

    all_signals = list(dict.fromkeys(inputs + wires + outputs))
    sig_to_idx  = {s: i for i, s in enumerate(all_signals)}
    n = len(all_signals)

    hyperedges: List[List[int]] = []
    gate_types: List[str]       = ["WIRE"] * n

    for gtype, inst_name, ports in instances:
        edge = [sig_to_idx[p] for p in ports if p in sig_to_idx]
        if len(edge) >= 2:
            hyperedges.append(edge)
            # Tag the first port (gate output) with gate type
            if edge:
                gate_types[edge[0]] = gtype

    has_dff = any("DFF" in gt or "FF" in gt for gt, _, _ in instances)

    return Hypergraph(
        num_vertices=n,
        num_hyperedges=len(hyperedges),
        hyperedges=hyperedges,
        vertex_names=all_signals,
        gate_types=gate_types,
        is_sequential=has_dff,
    )


# ---------------------------------------------------------------------------
# hMETIS / ISPD98 HGR format
# ---------------------------------------------------------------------------

def parse_hgr(path: str) -> Hypergraph:
    """
    Parse an hMETIS hypergraph (.hgr) file.

    First-line format descriptor (3rd token if present):
      bit 0 set  → has hyperedge weights
      bit 1 set  → has vertex weights
      bit 2 set  → has vertex fixedness (ISPD98 extension)

    Fixed vertices (if present) are stored in hg.fixed_vertices.
    """
    path  = Path(path)
    lines = [l.rstrip() for l in path.read_text(errors="replace").splitlines()]
    lines = [l for l in lines if not l.startswith("%")]

    if not lines:
        raise ValueError(f"Empty HGR file: {path}")

    header  = lines[0].split()
    num_he  = int(header[0])
    num_v   = int(header[1])
    fmt     = int(header[2]) if len(header) >= 3 else 0

    has_he_weights = bool(fmt & 1)
    has_v_weights  = bool(fmt & 10) or fmt in (10, 11)   # hMETIS convention
    has_fixed      = bool(fmt & 100) or fmt in (100, 110, 101, 111)

    # Normalise: hMETIS uses fmt 1=edge-wt, 10=vert-wt, 11=both
    if fmt == 1:
        has_he_weights, has_v_weights = True, False
    elif fmt == 10:
        has_he_weights, has_v_weights = False, True
    elif fmt == 11:
        has_he_weights, has_v_weights = True, True
    else:
        has_he_weights = fmt in (1, 11)
        has_v_weights  = fmt in (10, 11)

    hyperedges: List[List[int]] = []
    he_weights: List[float]     = []

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
            verts = [int(t) - 1 for t in tokens[1:]]
        else:
            he_weights.append(1.0)
            verts = [int(t) - 1 for t in tokens]
        hyperedges.append(verts)

    v_weights: List[float]   = [1.0] * num_v
    fixed_vertices: List[int] = []

    offset = num_he + 1
    if has_v_weights:
        for j in range(num_v):
            idx = offset + j
            if idx < len(lines):
                v_weights[j] = float(lines[idx].split()[0])
        offset += num_v

    if has_fixed:
        for j in range(num_v):
            idx = offset + j
            if idx < len(lines) and lines[idx].strip() == "1":
                fixed_vertices.append(j)

    return Hypergraph(
        num_vertices=num_v,
        num_hyperedges=num_he,
        hyperedges=hyperedges,
        vertex_weights=v_weights,
        hyperedge_weights=he_weights,
        fixed_vertices=fixed_vertices if fixed_vertices else None,
    )


# ---------------------------------------------------------------------------
# Utility: infer cell areas from gate-type table
# ---------------------------------------------------------------------------

def infer_vertex_areas(hg: Hypergraph) -> List[float]:
    """
    Return a vertex-weight list using the NAND-equivalent area table.
    Use when the hypergraph was loaded without area weights (e.g., HGR).
    """
    if hg.gate_types is None:
        return [1.0] * hg.num_vertices
    return [_GATE_AREA.get(gt, 1.0) for gt in hg.gate_types]


# ---------------------------------------------------------------------------
# Dispatch  (auto-detect from file extension)
# ---------------------------------------------------------------------------

def load_benchmark(path: str, fmt: Optional[str] = None,
                   use_area_weights: bool = True) -> Hypergraph:
    """
    Load a benchmark file.  fmt ∈ {'bench','verilog','hgr'} or None (auto).
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
            raise ValueError(f"Cannot auto-detect format for '{ext}'")

    if fmt == "bench":
        return parse_bench(path, use_area_weights=use_area_weights)
    elif fmt == "verilog":
        return parse_verilog(path)
    elif fmt == "hgr":
        return parse_hgr(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------

def load_manifest(csv_path: str) -> List[Dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Global-net filter
# ---------------------------------------------------------------------------

def apply_global_net_threshold(hg: Hypergraph, threshold: int) -> Hypergraph:
    """
    Remove hyperedges with > threshold pins (high-fanout / global nets).
    All metadata (gate_types, fixed_vertices, net_criticality) is preserved.
    """
    keep_he   = []
    keep_hw   = []
    keep_crit = []

    for i, he in enumerate(hg.hyperedges):
        if len(he) <= threshold:
            keep_he.append(he)
            keep_hw.append(hg.hyperedge_weights[i])
            if hg.net_criticality is not None and i < len(hg.net_criticality):
                keep_crit.append(hg.net_criticality[i])

    return Hypergraph(
        num_vertices=hg.num_vertices,
        num_hyperedges=len(keep_he),
        hyperedges=keep_he,
        vertex_weights=list(hg.vertex_weights),
        hyperedge_weights=keep_hw,
        vertex_names=list(hg.vertex_names),
        vertex_coords=hg.vertex_coords,
        fixed_vertices=hg.fixed_vertices,
        gate_types=hg.gate_types,
        net_criticality=keep_crit if keep_crit else None,
        is_sequential=hg.is_sequential,
    )
