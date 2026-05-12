"""
Static Timing Analysis (STA) for ISCAS BENCH netlists.

Computes per-net criticality weights used by the timing-aware partitioning
objective:

    timing_weighted_cut = Σ_n  w(n) * (1 + criticality(n))  if net n is cut

Theory
------
For a combinational circuit (ISCAS'85) or within a clock domain (ISCAS'89):

  Arrival Time (AT)      — latest data arrival at a cell's output.
                           Forward traversal in topological order.
  Required Arrival (RAT) — latest time data must arrive to meet timing.
                           Backward traversal from primary outputs.
  Slack                  — RAT - AT.  Negative = timing violated.
  Criticality            — (max_slack - slack) / max_slack ∈ [0, 1].
                           Nets on the critical path have criticality ≈ 1.

Gate delays  (normalised FO4 units; conservative 28 nm-style values)
-------------------------------------------------------------------
  INV / BUF  : 0.5
  NAND / NOR : 1.0
  AND / OR   : 1.5
  XOR / XNOR : 2.0
  DFF        : 0.0   (register boundary; D-input is a timing endpoint)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from objective import Hypergraph
from benchmark_io import _GATE_DELAY


# ---------------------------------------------------------------------------
# Topological sort  (Kahn's algorithm on signal/gate DAG)
# ---------------------------------------------------------------------------

def _topological_sort(
    gate_defs: Dict[str, tuple],   # out_sig → (gate_type, [in_sigs])
    primary_inputs: List[str],
) -> List[str]:
    """
    Return signals in topological order (PI first, PO last).
    DFF Q→D arcs are cut (register boundary).
    """
    in_degree: Dict[str, int] = {}
    dependents: Dict[str, List[str]] = {}

    for sig, (gt, inputs) in gate_defs.items():
        in_degree.setdefault(sig, 0)
        if gt == "DFF":
            continue  # DFF breaks combinational path
        for s in inputs:
            in_degree[sig] = in_degree.get(sig, 0) + 1
            dependents.setdefault(s, []).append(sig)

    for pi in primary_inputs:
        in_degree.setdefault(pi, 0)

    queue = deque([s for s, d in in_degree.items() if d == 0])
    order: List[str] = []

    while queue:
        sig = queue.popleft()
        order.append(sig)
        for dep in dependents.get(sig, []):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    return order


# ---------------------------------------------------------------------------
# Forward AT propagation
# ---------------------------------------------------------------------------

def _compute_at(
    topo_order: List[str],
    gate_defs: Dict[str, tuple],
    primary_inputs: List[str],
) -> Dict[str, float]:
    """Compute arrival time at each signal's output."""
    at: Dict[str, float] = {}

    for sig in primary_inputs:
        at[sig] = 0.0

    for sig in topo_order:
        if sig in primary_inputs:
            continue
        if sig not in gate_defs:
            at.setdefault(sig, 0.0)
            continue
        gt, inputs = gate_defs[sig]
        if gt == "DFF":
            at[sig] = 0.0   # DFF Q output is a new timing start-point
            continue
        delay = _GATE_DELAY.get(gt, 1.0)
        max_input_at = max((at.get(s, 0.0) for s in inputs), default=0.0)
        at[sig] = max_input_at + delay

    return at


# ---------------------------------------------------------------------------
# Backward RAT propagation
# ---------------------------------------------------------------------------

def _compute_rat(
    topo_order: List[str],
    gate_defs: Dict[str, tuple],
    at: Dict[str, float],
    primary_outputs: List[str],
    clock_period: Optional[float] = None,
) -> Dict[str, float]:
    """Compute required arrival time at each signal's output."""
    if clock_period is None:
        # Use the critical path length as the clock period
        clock_period = max(at.values(), default=1.0)

    rat: Dict[str, float] = {}

    for po in primary_outputs:
        rat[po] = clock_period

    # Build reverse fanout map
    fanin_map: Dict[str, List[str]] = {}  # driver_sig → [consumer_sigs]
    for sig, (gt, inputs) in gate_defs.items():
        if gt == "DFF":
            continue
        for s in inputs:
            fanin_map.setdefault(s, []).append(sig)

    for sig in reversed(topo_order):
        if sig in primary_outputs:
            continue
        consumers = fanin_map.get(sig, [])
        if not consumers:
            rat.setdefault(sig, clock_period)
            continue
        gt_consumers = [gate_defs[c][0] for c in consumers if c in gate_defs]
        # RAT at sig = min over consumers of (RAT[consumer] - delay[consumer])
        rat[sig] = min(
            rat.get(c, clock_period) - _GATE_DELAY.get(gate_defs[c][0], 1.0)
            for c in consumers if c in gate_defs
        )

    return rat


# ---------------------------------------------------------------------------
# Per-net criticality
# ---------------------------------------------------------------------------

def compute_net_criticality(
    hg: Hypergraph,
    gate_defs: Optional[Dict[str, tuple]] = None,
    primary_inputs: Optional[List[str]] = None,
    primary_outputs: Optional[List[str]] = None,
    clock_period: Optional[float] = None,
) -> List[float]:
    """
    Compute per-net criticality ∈ [0, 1].

    If gate_defs / primary_inputs / primary_outputs are supplied (from
    parse_bench internals), a full AT/RAT analysis is performed.
    Otherwise a structural criticality heuristic is used:
      - Nets with more than one pin connected to a DFF are assigned
        criticality = 1.0 (register-to-register paths).
      - All other nets get a fanout-normalised heuristic criticality.

    The result is stored in hg.net_criticality.
    """
    if gate_defs is None or primary_inputs is None:
        return _heuristic_criticality(hg)

    po_list = primary_outputs or []

    # Build vertex-name → index map for net annotation
    name_to_idx = {n: i for i, n in enumerate(hg.vertex_names)}

    topo = _topological_sort(gate_defs, primary_inputs)
    at   = _compute_at(topo, gate_defs, primary_inputs)
    rat  = _compute_rat(topo, gate_defs, at, po_list, clock_period)

    # Slack per signal
    max_slack = max(
        (rat.get(s, 0.0) - at.get(s, 0.0) for s in at),
        default=1.0,
    )
    max_slack = max(max_slack, 1e-6)

    def sig_criticality(sig: str) -> float:
        a = at.get(sig, 0.0)
        r = rat.get(sig, 0.0)
        slack = r - a
        return float(np.clip(1.0 - slack / max_slack, 0.0, 1.0))

    # Per-net criticality = max criticality over pins in the net
    crit = []
    for he in hg.hyperedges:
        max_c = 0.0
        for v in he:
            name = hg.vertex_names[v] if v < len(hg.vertex_names) else ""
            c = sig_criticality(name)
            if c > max_c:
                max_c = c
        crit.append(max_c)

    return crit


def _heuristic_criticality(hg: Hypergraph) -> List[float]:
    """
    Fanout-based heuristic criticality when full STA is not available.
    Nets with high fanout are treated as less critical (broadcast signals).
    Nets with a single sink are more likely on a critical path.
    """
    if hg.net_criticality is not None:
        return hg.net_criticality

    max_pins = max((len(he) for he in hg.hyperedges), default=1)
    crit = []
    for he in hg.hyperedges:
        # Inverted fanout: single-sink net (pin count=2) → crit close to 1
        fanout = max(len(he) - 1, 1)
        c = 1.0 / (1.0 + np.log1p(fanout))
        # DFF output nets have criticality = 1 (register-to-register)
        if hg.gate_types is not None and len(he) > 0:
            if hg.gate_types[he[0]] == "DFF":
                c = 1.0
        crit.append(float(c))
    return crit


# ---------------------------------------------------------------------------
# Annotate hypergraph in place
# ---------------------------------------------------------------------------

def annotate_timing(
    hg: Hypergraph,
    gate_defs: Optional[Dict] = None,
    primary_inputs: Optional[List[str]] = None,
    primary_outputs: Optional[List[str]] = None,
    clock_period: Optional[float] = None,
) -> Hypergraph:
    """
    Run STA and store per-net criticality in hg.net_criticality.
    Returns the same hg object (modified in place).
    """
    hg.net_criticality = compute_net_criticality(
        hg, gate_defs, primary_inputs, primary_outputs, clock_period
    )
    return hg


# ---------------------------------------------------------------------------
# Convenience: run full STA from a .bench file path
# ---------------------------------------------------------------------------

def run_sta_from_bench(bench_path: str, hg: Hypergraph,
                       clock_period: Optional[float] = None) -> Hypergraph:
    """
    Re-parse the BENCH file to extract gate_defs and PI/PO lists,
    run STA, and annotate hg.net_criticality.
    """
    from pathlib import Path
    import re

    text = Path(bench_path).read_text(errors="replace")
    _gate_line_re = re.compile(r"^(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)\s*$")

    primary_inputs:  List[str] = []
    primary_outputs: List[str] = []
    gate_defs: Dict[str, tuple] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")):
            continue
        m_io = re.match(r"^(INPUT|OUTPUT)\s*\(\s*(\w+)\s*\)\s*$", line, re.I)
        if m_io:
            kind, sig = m_io.group(1).upper(), m_io.group(2)
            (primary_inputs if kind == "INPUT" else primary_outputs).append(sig)
            continue
        m = _gate_line_re.match(line)
        if m:
            out_sig  = m.group(1)
            gt       = m.group(2).upper()
            in_sigs  = [s.strip() for s in m.group(3).split(",") if s.strip()]
            gate_defs[out_sig] = (gt, in_sigs)

    return annotate_timing(hg, gate_defs, primary_inputs, primary_outputs, clock_period)
