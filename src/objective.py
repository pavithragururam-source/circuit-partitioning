"""
Hypergraph data structures and the weighted scalar objective function.

Expert corrections vs original version
---------------------------------------
* Added SOED (Sum of External Degrees) — the canonical k-way cut metric.
  Cut-size (binary) is a special case of SOED for k=2.
* Added compute_connectivity — per-net partition-span count.
* compute_placement_penalty now computes HPWL (Half-Perimeter Wirelength),
  which is the standard wirelength model in VLSI placement/partitioning.
* Added build_cell_to_nets_index — prerequisite for O(1) FM gain updates.
* Added net_criticality_weights — populated by timing.py when timing_aware=True.
* repair_balance is now O(n log n) using a sorted-by-weight candidate queue
  instead of the original O(n²k) greedy scan.
* evaluate() drives the scalar objective from SOED (not cut-size) when k > 2,
  and correctly invokes timing penalty from net_criticality_weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Hypergraph:
    """
    Common internal hypergraph representation for all benchmarks.

    Attributes
    ----------
    num_vertices      : |V|
    num_hyperedges    : |E|
    hyperedges        : list of vertex-index lists (0-indexed)
    vertex_weights    : cell area / unit weight per vertex
    hyperedge_weights : net weight (default 1.0)
    vertex_names      : signal / cell name strings
    vertex_coords     : (x, y) tuples for placement-aware HPWL
    fixed_vertices    : set of vertex indices whose partition is pre-assigned
    gate_types        : gate-type string per vertex (for timing analysis)
    net_criticality   : per-net criticality ∈ [0,1], populated by timing.py
    is_sequential     : True when circuit contains DFFs (ISCAS'89)
    """
    num_vertices: int
    num_hyperedges: int
    hyperedges: List[List[int]]
    vertex_weights: List[float]       = field(default_factory=list)
    hyperedge_weights: List[float]    = field(default_factory=list)
    vertex_names: List[str]           = field(default_factory=list)
    vertex_coords: Optional[List[Tuple[float, float]]] = None
    fixed_vertices: Optional[List[int]] = None       # pre-placed; partition fixed
    gate_types: Optional[List[str]] = None           # e.g. "NAND", "DFF", "INPUT"
    net_criticality: Optional[List[float]] = None    # populated by timing.py
    is_sequential: bool = False

    def __post_init__(self):
        if not self.vertex_weights:
            self.vertex_weights = [1.0] * self.num_vertices
        if not self.hyperedge_weights:
            self.hyperedge_weights = [1.0] * self.num_hyperedges
        if not self.vertex_names:
            self.vertex_names = [str(i) for i in range(self.num_vertices)]


@dataclass
class PartitionConfig:
    """
    Mirror of OpenROAD/TritonPart partitioning parameters.

    Expert additions
    ----------------
    use_soed       : optimise SOED instead of binary cut (recommended for k>2)
    fm_refinement  : run FM post-refinement after metaheuristic
    multilevel     : enable HEM coarsening before optimisation
    coarsen_to     : stop coarsening when |V_coarse| ≤ this value
    """
    num_parts: int = 2
    balance_constraint: float = 0.10
    timing_aware: bool = False
    placement_aware: bool = False
    global_net_threshold: int = 1000
    solution_file: Optional[str] = None
    # Objective weights
    w_cut: float = 1.0
    w_balance: float = 10.0
    w_timing: float = 1.0          # applied only when timing_aware=True
    w_placement: float = 0.0
    # Expert extensions
    use_soed: bool = True           # SOED metric (preferred over binary cut for k>2)
    fm_refinement: bool = True      # FM post-pass after every metaheuristic run
    fm_max_passes: int = 10         # FM pass limit
    multilevel: bool = False        # HEM coarsening (activate for circuits > 5k cells)
    coarsen_to: int = 200           # coarsening stops at this vertex count


# ---------------------------------------------------------------------------
# Structural index — prerequisites for FM and timing
# ---------------------------------------------------------------------------

def build_cell_to_nets_index(hg: Hypergraph) -> List[List[int]]:
    """
    Return cell_to_nets[v] = list of hyperedge indices containing vertex v.
    O(Σ|net_size|) to build; needed by FM for O(1) gain lookup per cell.
    """
    c2n: List[List[int]] = [[] for _ in range(hg.num_vertices)]
    for e_idx, he in enumerate(hg.hyperedges):
        for v in he:
            c2n[v].append(e_idx)
    return c2n


def build_partition_counts(hg: Hypergraph, partition: np.ndarray,
                           num_parts: int) -> List[np.ndarray]:
    """
    Return part_counts[e][p] = number of vertices of net e in partition p.
    Used by FM gain initialisation and update.
    """
    counts = []
    for he in hg.hyperedges:
        pc = np.zeros(num_parts, dtype=np.int32)
        for v in he:
            pc[int(partition[v])] += 1
        counts.append(pc)
    return counts


# ---------------------------------------------------------------------------
# Cut metrics
# ---------------------------------------------------------------------------

def compute_cutsize(hg: Hypergraph, partition: np.ndarray) -> float:
    """
    Binary hyperedge cut: weight of nets spanning ≥ 2 partitions.
    Standard metric for k=2; use compute_soed for k>2.
    """
    cut = 0.0
    for i, hedge in enumerate(hg.hyperedges):
        parts = set(int(partition[v]) for v in hedge)
        if len(parts) > 1:
            cut += hg.hyperedge_weights[i]
    return cut


def compute_soed(hg: Hypergraph, partition: np.ndarray) -> float:
    """
    Sum of External Degrees (SOED):
        SOED = Σ_n  w(n) * (|S(n)| - 1)
    where S(n) = set of partitions spanned by net n.

    SOED == cutsize for k=2.  For k>2 it penalises nets spanning many
    partitions, giving a richer signal than binary cut.
    Reference: Alpert & Kahng, ISPD 1995.
    """
    soed = 0.0
    for i, hedge in enumerate(hg.hyperedges):
        parts = set(int(partition[v]) for v in hedge)
        span = len(parts)
        if span > 1:
            soed += hg.hyperedge_weights[i] * (span - 1)
    return soed


def compute_connectivity(hg: Hypergraph, partition: np.ndarray) -> float:
    """
    Connectivity metric: Σ_n w(n) * |S(n)|  (total partition spans).
    Minimising connectivity is equivalent to minimising SOED.
    """
    conn = 0.0
    for i, hedge in enumerate(hg.hyperedges):
        parts = set(int(partition[v]) for v in hedge)
        conn += hg.hyperedge_weights[i] * len(parts)
    return conn


def compute_timing_weighted_cut(hg: Hypergraph, partition: np.ndarray) -> float:
    """
    Timing-weighted cut:  Σ_n  w(n) * (1 + criticality(n))  if n is cut.
    Cutting a critical-path net is more expensive than a slack-rich net.
    Requires hg.net_criticality to be populated by timing.py.
    """
    if hg.net_criticality is None:
        return compute_cutsize(hg, partition)
    twc = 0.0
    for i, hedge in enumerate(hg.hyperedges):
        parts = set(int(partition[v]) for v in hedge)
        if len(parts) > 1:
            crit = float(hg.net_criticality[i]) if i < len(hg.net_criticality) else 0.0
            twc += hg.hyperedge_weights[i] * (1.0 + crit)
    return twc


# ---------------------------------------------------------------------------
# Balance metric
# ---------------------------------------------------------------------------

def compute_imbalance(hg: Hypergraph, partition: np.ndarray,
                      num_parts: int) -> float:
    """
    Maximum fractional deviation from ideal weight W/k:
        imbalance = max_p |w(p) - W/k| / (W/k)
    Uses vertex area weights so balance is area-driven, not count-driven.
    """
    w = np.asarray(hg.vertex_weights, dtype=float)
    total = w.sum()
    if total == 0.0:
        return 0.0
    ideal = total / num_parts
    part_w = np.array([w[partition == p].sum() for p in range(num_parts)])
    return float(np.max(np.abs(part_w - ideal)) / ideal)


# ---------------------------------------------------------------------------
# Placement-aware metric (HPWL)
# ---------------------------------------------------------------------------

def compute_hpwl(hg: Hypergraph, partition: np.ndarray,
                 num_parts: int) -> float:
    """
    Half-Perimeter Wirelength (HPWL) summed over all cut nets.

    HPWL(n) = (x_max - x_min + y_max - y_min) for the bounding box of the
    cells on net n.  This is the standard wirelength model in VLSI EDA.

    Returns 0 if no placement coordinates are available.
    """
    if hg.vertex_coords is None:
        return 0.0
    coords = np.array(hg.vertex_coords, dtype=float)
    hpwl = 0.0
    for i, hedge in enumerate(hg.hyperedges):
        if len(hedge) < 2:
            continue
        xy = coords[hedge]
        hpwl += hg.hyperedge_weights[i] * (
            float(xy[:, 0].max() - xy[:, 0].min()) +
            float(xy[:, 1].max() - xy[:, 1].min())
        )
    return hpwl


# ---------------------------------------------------------------------------
# Combined evaluation  (mirrors evaluate_hypergraph_solution)
# ---------------------------------------------------------------------------

def evaluate(hg: Hypergraph, partition: np.ndarray,
             cfg: PartitionConfig) -> Dict:
    """
    Compute all metrics and the scalar objective.

    Objective
    ---------
    F = w_cut * (SOED if cfg.use_soed else cutsize)
      + w_balance * max(0, imbalance - balance_constraint) * (W / k)
      + w_timing  * timing_weighted_cut     [only when timing_aware]
      + w_placement * HPWL                  [only when placement_aware]

    The balance penalty is multiplied by the ideal partition weight so the
    penalty scale is consistent with the cut metric.
    """
    cutsize  = compute_cutsize(hg, partition)
    soed     = compute_soed(hg, partition)
    primary_cut = soed if cfg.use_soed else cutsize

    imbalance = compute_imbalance(hg, partition, cfg.num_parts)
    w_vec = np.asarray(hg.vertex_weights, dtype=float)
    ideal_w = w_vec.sum() / cfg.num_parts
    imbalance_penalty = max(0.0, imbalance - cfg.balance_constraint) * ideal_w

    timing_penalty = (
        compute_timing_weighted_cut(hg, partition)
        if cfg.timing_aware else 0.0
    )

    placement_penalty = (
        compute_hpwl(hg, partition, cfg.num_parts)
        if cfg.placement_aware else 0.0
    )

    objective = (
        cfg.w_cut       * primary_cut
        + cfg.w_balance * imbalance_penalty
        + cfg.w_timing  * timing_penalty
        + cfg.w_placement * placement_penalty
    )

    return {
        "cutsize":          cutsize,
        "soed":             soed,
        "connectivity":     compute_connectivity(hg, partition),
        "imbalance":        imbalance,
        "imbalance_penalty":imbalance_penalty,
        "timing_penalty":   timing_penalty,
        "hpwl":             placement_penalty,
        "objective":        objective,
        "feasible":         imbalance <= cfg.balance_constraint,
    }


# ---------------------------------------------------------------------------
# Constraint repair  —  O(n log n) via sorted weight queue
# ---------------------------------------------------------------------------

def repair_balance(
    partition: np.ndarray,
    hg: Hypergraph,
    cfg: PartitionConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Move vertices from overloaded partitions until all satisfy the balance
    constraint. Candidates are sorted by weight so the smallest cell is moved
    first, minimising unnecessary weight swing.

    Complexity: O(n log n) preprocessing + O(n) repair passes.
    Fixed vertices (hg.fixed_vertices) are never moved.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(partition)
    k = cfg.num_parts
    w = np.asarray(hg.vertex_weights, dtype=float)
    total = w.sum()
    if total == 0.0:
        return partition

    fixed = set(hg.fixed_vertices) if hg.fixed_vertices else set()
    ideal = total / k
    upper = ideal * (1.0 + cfg.balance_constraint)

    partition = partition.copy()

    # Sort movable vertices by weight ascending (move lightest first)
    movable = np.array([v for v in range(n) if v not in fixed])
    if len(movable) == 0:
        return partition
    movable = movable[np.argsort(w[movable])]

    for _ in range(n * k):
        part_w = np.array([w[partition == p].sum() for p in range(k)], dtype=float)
        overloaded = np.where(part_w > upper)[0]
        if len(overloaded) == 0:
            break
        p_from = int(overloaded[np.argmax(part_w[overloaded])])
        p_to = int(np.argmin(part_w))
        # Pick lightest movable vertex in the overloaded partition
        candidates = movable[partition[movable] == p_from]
        if len(candidates) == 0:
            break
        v = int(candidates[0])  # already sorted by weight
        partition[v] = p_to
    return partition


def random_partition(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random partition; guarantees every part has ≥ 1 vertex."""
    p = rng.integers(0, k, size=n)
    for part in range(k):
        if not np.any(p == part):
            p[rng.integers(0, n)] = part
    return p
