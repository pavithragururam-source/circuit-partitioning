"""
Hypergraph data structures and the weighted scalar objective function.

Compatible with OpenROAD/TritonPart interface concepts:
  evaluate_hypergraph_solution / evaluate_part_design_solution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np


@dataclass
class Hypergraph:
    """Common internal hypergraph representation for all benchmarks."""
    num_vertices: int
    num_hyperedges: int
    hyperedges: List[List[int]]
    vertex_weights: List[float] = field(default_factory=list)
    hyperedge_weights: List[float] = field(default_factory=list)
    vertex_names: List[str] = field(default_factory=list)
    vertex_coords: Optional[List[tuple]] = None  # (x, y) for placement-aware mode

    def __post_init__(self):
        if not self.vertex_weights:
            self.vertex_weights = [1.0] * self.num_vertices
        if not self.hyperedge_weights:
            self.hyperedge_weights = [1.0] * self.num_hyperedges
        if not self.vertex_names:
            self.vertex_names = [str(i) for i in range(self.num_vertices)]


@dataclass
class PartitionConfig:
    """Mirror of OpenROAD/TritonPart partitioning parameters."""
    num_parts: int = 2
    balance_constraint: float = 0.10   # max allowed imbalance ratio
    timing_aware: bool = False
    placement_aware: bool = False
    global_net_threshold: int = 1000   # ignore nets larger than this
    solution_file: Optional[str] = None
    # Objective weights
    w_cut: float = 1.0
    w_balance: float = 10.0
    w_timing: float = 0.0
    w_placement: float = 0.0


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_cutsize(hg: Hypergraph, partition: np.ndarray) -> float:
    """Sum of weights of hyperedges that span more than one partition."""
    cut = 0.0
    threshold = hg.num_vertices  # global_net_threshold applied upstream
    for i, hedge in enumerate(hg.hyperedges):
        if len(hedge) > threshold:
            continue
        parts = set(int(partition[v]) for v in hedge)
        if len(parts) > 1:
            cut += hg.hyperedge_weights[i]
    return cut


def compute_imbalance(hg: Hypergraph, partition: np.ndarray, num_parts: int) -> float:
    """Max deviation of any partition's weight from the ideal (total/k)."""
    w = np.asarray(hg.vertex_weights, dtype=float)
    total = w.sum()
    if total == 0:
        return 0.0
    ideal = total / num_parts
    part_w = np.array([w[partition == p].sum() for p in range(num_parts)], dtype=float)
    return float(np.max(np.abs(part_w - ideal)) / ideal)


def compute_placement_penalty(hg: Hypergraph, partition: np.ndarray, num_parts: int) -> float:
    """Mean intra-partition spread (L2 distance from centroid) — zero if no coords."""
    if hg.vertex_coords is None:
        return 0.0
    coords = np.array(hg.vertex_coords, dtype=float)
    penalty = 0.0
    for p in range(num_parts):
        mask = partition == p
        if mask.sum() < 2:
            continue
        c = coords[mask]
        center = c.mean(axis=0)
        penalty += np.linalg.norm(c - center, axis=1).mean()
    return penalty / num_parts


# ---------------------------------------------------------------------------
# Combined evaluation (mirrors evaluate_hypergraph_solution)
# ---------------------------------------------------------------------------

def evaluate(hg: Hypergraph, partition: np.ndarray, cfg: PartitionConfig) -> Dict:
    """
    Evaluate a partition and return a dict of metrics + the scalar objective.

    F = w_c * cutsize
      + w_b * max(0, imbalance - balance_constraint)
      + w_t * timing_penalty
      + w_p * placement_penalty
    """
    cutsize = compute_cutsize(hg, partition)
    imbalance = compute_imbalance(hg, partition, cfg.num_parts)
    imbalance_penalty = max(0.0, imbalance - cfg.balance_constraint)
    timing_penalty = 0.0  # populated by OpenROAD adapter when timing_aware=True
    placement_penalty = (
        compute_placement_penalty(hg, partition, cfg.num_parts)
        if cfg.placement_aware else 0.0
    )

    objective = (
        cfg.w_cut * cutsize
        + cfg.w_balance * imbalance_penalty
        + cfg.w_timing * timing_penalty
        + cfg.w_placement * placement_penalty
    )

    return {
        "cutsize": cutsize,
        "imbalance": imbalance,
        "imbalance_penalty": imbalance_penalty,
        "timing_penalty": timing_penalty,
        "placement_penalty": placement_penalty,
        "objective": objective,
        "feasible": imbalance <= cfg.balance_constraint,
    }


# ---------------------------------------------------------------------------
# Constraint repair
# ---------------------------------------------------------------------------

def repair_balance(
    partition: np.ndarray,
    hg: Hypergraph,
    cfg: PartitionConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Greedily move vertices from overloaded to underloaded partitions until
    the balance constraint is satisfied or no further improvement is possible.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(partition)
    k = cfg.num_parts
    w = np.asarray(hg.vertex_weights, dtype=float)
    total = w.sum()
    if total == 0:
        return partition
    ideal = total / k
    upper = ideal * (1.0 + cfg.balance_constraint)

    partition = partition.copy()
    for _ in range(n * k):
        part_w = np.array([w[partition == p].sum() for p in range(k)], dtype=float)
        overloaded = np.where(part_w > upper)[0]
        if len(overloaded) == 0:
            break
        p_from = overloaded[np.argmax(part_w[overloaded])]
        p_to = int(np.argmin(part_w))
        candidates = np.where(partition == p_from)[0]
        if len(candidates) == 0:
            break
        # Move the lightest vertex to avoid unnecessary weight swing
        v = int(candidates[np.argmin(w[candidates])])
        partition[v] = p_to
    return partition


def random_partition(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random partition of n vertices into k parts."""
    p = rng.integers(0, k, size=n)
    # Guarantee every part has at least one vertex
    for part in range(k):
        if not np.any(p == part):
            p[rng.integers(0, n)] = part
    return p
