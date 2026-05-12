"""
Multilevel Hypergraph Coarsening via Heavy Edge Matching (HEM).

References
----------
- Karypis & Kumar, SIAM J. Sci. Comput. 1998  (METIS multilevel k-way)
- Alpert et al., ICCAD 1997  (multilevel partitioning for VLSI)

Overview
--------
Multilevel framework:
  1. COARSENING:  Repeatedly apply HEM to reduce |V| until |V| ≤ coarsen_to.
     At each level, matched vertex pairs are merged into super-vertices.
     The coarsened hypergraph is constructed from the cluster map.

  2. INITIAL PARTITION:  Run the metaheuristic on the coarsest hypergraph.

  3. UNCOARSENING (projection + refinement):
     Project the coarse partition back to each finer level.
     Apply FM refinement at every level to recover quality lost by coarsening.

Key design choices
------------------
* HEM uses net-weight-scaled adjacency: each pair (u,v) gets contribution
  w(n) / (|n| - 1) from every net n containing both u and v.
  This is equivalent to the "heavy clique" net model.
* Matching is greedy: vertices iterated in random order; each vertex is
  matched with its heaviest unmatched neighbour.
* Super-vertex weight = sum of constituent vertex weights (area-balanced).
* Hyperedges fully internal to a super-vertex are removed (self-loops).

Fixed vertices (hg.fixed_vertices) are never merged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from objective import (
    Hypergraph, PartitionConfig,
    repair_balance,
)


# ---------------------------------------------------------------------------
# Helper: build weighted adjacency from hypergraph
# ---------------------------------------------------------------------------

def _build_adj(hg: Hypergraph) -> Dict[int, Dict[int, float]]:
    """
    Return adj[u][v] = Σ_n w(n)/(|n|-1) for all nets n containing both u and v.
    Only nets with 2 ≤ |n| ≤ 50 contribute (ignore extremely large nets).
    """
    adj: Dict[int, Dict[int, float]] = {v: {} for v in range(hg.num_vertices)}

    for e_idx, he in enumerate(hg.hyperedges):
        sz = len(he)
        if sz < 2 or sz > 50:
            continue
        contrib = hg.hyperedge_weights[e_idx] / (sz - 1)
        for i in range(sz):
            for j in range(i + 1, sz):
                u, v = he[i], he[j]
                adj[u][v] = adj[u].get(v, 0.0) + contrib
                adj[v][u] = adj[v].get(u, 0.0) + contrib

    return adj


# ---------------------------------------------------------------------------
# One level of Heavy Edge Matching
# ---------------------------------------------------------------------------

def _hem_match(
    hg: Hypergraph,
    rng: np.random.Generator,
    fixed: set,
) -> np.ndarray:
    """
    Match vertices using HEM and return cluster_map[v] ∈ [0, n_clusters).
    """
    n = hg.num_vertices
    adj = _build_adj(hg)

    # Iterate vertices in random order to avoid systematic bias
    order = rng.permutation(n)
    matched   = np.full(n, -1, dtype=np.int32)
    cluster_id = 0
    cluster_map = np.full(n, -1, dtype=np.int32)

    for v in order:
        if matched[v] >= 0:
            continue
        if v in fixed:
            # Fixed vertex forms its own singleton cluster
            matched[v] = cluster_id
            cluster_map[v] = cluster_id
            cluster_id += 1
            continue

        # Find heaviest unmatched non-fixed neighbour
        best_u = -1
        best_w = -1.0
        for u, w in adj[v].items():
            if matched[u] < 0 and u not in fixed and w > best_w:
                best_w, best_u = w, u

        matched[v] = cluster_id
        cluster_map[v] = cluster_id
        if best_u >= 0:
            matched[best_u] = cluster_id
            cluster_map[best_u] = cluster_id
        cluster_id += 1

    return cluster_map


# ---------------------------------------------------------------------------
# Build coarsened hypergraph from cluster map
# ---------------------------------------------------------------------------

def _build_coarse_hg(
    hg: Hypergraph,
    cluster_map: np.ndarray,
) -> Hypergraph:
    """
    Contract hg according to cluster_map.  Self-loop nets (all vertices
    in same cluster) are removed.  Parallel nets are merged (weights summed).
    """
    n_coarse = int(cluster_map.max()) + 1

    # Super-vertex weights = sum of constituent vertex weights
    coarse_vw = np.zeros(n_coarse, dtype=float)
    for v in range(hg.num_vertices):
        coarse_vw[cluster_map[v]] += hg.vertex_weights[v]

    # Build coarse hyperedges, merging parallel nets
    net_dict: Dict[frozenset, float] = {}
    for e_idx, he in enumerate(hg.hyperedges):
        coarse_he = frozenset(int(cluster_map[v]) for v in he)
        if len(coarse_he) < 2:
            continue   # self-loop: all vertices coalesced into same cluster
        net_dict[coarse_he] = net_dict.get(coarse_he, 0.0) + hg.hyperedge_weights[e_idx]

    coarse_edges  = [list(s) for s in net_dict.keys()]
    coarse_ew     = list(net_dict.values())

    return Hypergraph(
        num_vertices=n_coarse,
        num_hyperedges=len(coarse_edges),
        hyperedges=coarse_edges,
        vertex_weights=list(coarse_vw),
        hyperedge_weights=coarse_ew,
    )


# ---------------------------------------------------------------------------
# Project coarse partition back to fine level
# ---------------------------------------------------------------------------

def _project_partition(
    coarse_partition: np.ndarray,
    cluster_map: np.ndarray,
) -> np.ndarray:
    """
    Map coarse-level partition labels to fine-level vertex assignments.
    cluster_map[v] = coarse vertex index for fine vertex v.
    """
    n_fine = len(cluster_map)
    fine_partition = np.empty(n_fine, dtype=np.int32)
    for v in range(n_fine):
        fine_partition[v] = coarse_partition[int(cluster_map[v])]
    return fine_partition


# ---------------------------------------------------------------------------
# Public API: CoarseningHierarchy
# ---------------------------------------------------------------------------

class CoarseningHierarchy:
    """
    Multi-level coarsening context.

    Usage
    -----
    hier = CoarseningHierarchy()
    coarsest_hg = hier.coarsen(hg, cfg, rng)

    # ... run metaheuristic on coarsest_hg to get coarse_partition ...

    final_partition = hier.uncoarsen(coarse_partition, hg, cfg, refiner)
    """

    def __init__(self):
        self._levels: List[Tuple[Hypergraph, np.ndarray]] = []
        # Each level: (fine_hg, cluster_map fine→coarse)

    def coarsen(
        self,
        hg: Hypergraph,
        cfg: PartitionConfig,
        rng: np.random.Generator,
    ) -> Hypergraph:
        """
        Iteratively coarsen hg until |V| ≤ cfg.coarsen_to or no reduction occurs.
        Returns the coarsest Hypergraph.
        """
        self._levels = []
        fixed = set(hg.fixed_vertices) if hg.fixed_vertices else set()
        current = hg

        while current.num_vertices > cfg.coarsen_to:
            cluster_map = _hem_match(current, rng, fixed)
            n_coarse = int(cluster_map.max()) + 1

            # Stop if coarsening ratio < 5 % (cannot compress further)
            if n_coarse >= int(0.95 * current.num_vertices):
                break

            coarse = _build_coarse_hg(current, cluster_map)
            self._levels.append((current, cluster_map))
            current = coarse

            # Update fixed set in coarse space
            fixed = {int(cluster_map[v]) for v in fixed}

        return current

    def uncoarsen(
        self,
        coarse_partition: np.ndarray,
        original_hg: Hypergraph,
        cfg: PartitionConfig,
        refiner=None,
    ) -> np.ndarray:
        """
        Project partition from coarsest level back to original and optionally
        apply FM refinement at each level.

        Parameters
        ----------
        coarse_partition : partition of the coarsest hypergraph
        original_hg      : the original (finest) hypergraph
        cfg              : partition config (used for FM and repair)
        refiner          : FMRefiner instance (or None to skip FM)
        """
        partition = coarse_partition.copy()

        for fine_hg, cluster_map in reversed(self._levels):
            partition = _project_partition(partition, cluster_map)
            # Repair balance (projection may violate constraint)
            partition = repair_balance(partition, fine_hg, cfg)
            # FM refinement at this level
            if refiner is not None:
                partition = refiner.refine(fine_hg, partition, cfg)

        # If no coarsening was done, levels is empty: just return the partition
        if not self._levels:
            return repair_balance(coarse_partition, original_hg, cfg)

        return partition

    @property
    def num_levels(self) -> int:
        return len(self._levels)


# ---------------------------------------------------------------------------
# Convenience function: full multilevel partitioning pipeline
# ---------------------------------------------------------------------------

def multilevel_partition(
    hg: Hypergraph,
    cfg: PartitionConfig,
    optimizer,
    refiner,
    rng: np.random.Generator,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Run the full multilevel pipeline:
      coarsen → metaheuristic → uncoarsen+FM → return (partition, history).

    Parameters
    ----------
    optimizer : BaseOptimizer instance (e.g. HHOOptimizer)
    refiner   : FMRefiner instance
    """
    hier = CoarseningHierarchy()
    coarsest = hier.coarsen(hg, cfg, rng)

    # Run metaheuristic on coarsest graph
    coarse_part, history = optimizer.optimize(coarsest, cfg, seed=seed)

    # Uncoarsen with FM at each level
    final_part = hier.uncoarsen(coarse_part, hg, cfg, refiner)

    return final_part, history
