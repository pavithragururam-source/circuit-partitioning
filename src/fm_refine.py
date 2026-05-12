"""
Fiduccia-Mattheyses (FM) k-way Hypergraph Partitioning Refinement.

References
----------
- Fiduccia & Mattheyses, DAC 1982  (original bisection FM)
- Sanchis, IEEE Trans. CAD 1989    (k-way extension)
- Karypis & Kumar, SIAM J. Sci. Comput. 1998  (multilevel k-way FM in METIS)

Algorithm Overview
------------------
One FM pass:
  1. Build gain_to[v][q] = SOED gain of moving vertex v to partition q.
     Uses SOED (Sum of External Degrees) gain model — exact for any k.
     Gain formula: gain(v, p→q) = Σ_n [ w(n) if pc[n][p]==1 ]
                                 - Σ_n [ w(n) if pc[n][q]==0 ]
  2. Insert all movable vertices into a max-heap keyed by best gain.
  3. Repeat until all vertices locked:
       a. Pop vertex v* with highest gain move (v*,p→q) respecting balance.
       b. Move v*; record (v*, p, q, gain) in move stack.
       c. Update gains of all unlocked vertices in nets containing v*
          using the EXACT incremental formulas derived from the SOED model.
  4. Find prefix of moves that minimises cumulative SOED.
  5. Accept that prefix; roll back the rest.
  6. Return (refined_partition, improved: bool).

Multi-pass until no improvement.

Complexity per pass: O(Σ|net_size| · k · log n)  with lazy-deletion heap.
Fixed vertices (hg.fixed_vertices) are never moved.
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

import numpy as np

from objective import (
    Hypergraph, PartitionConfig,
    build_cell_to_nets_index, build_partition_counts,
    compute_soed, repair_balance,
)


# ---------------------------------------------------------------------------
# Lazy-deletion max-heap  (heap stores negative gain for min-heap inversion)
# ---------------------------------------------------------------------------

class _GainHeap:
    """
    Max-heap for (gain, cell) pairs with lazy invalidation.
    Supports push/invalidate/pop_best in O(log n).
    """

    __slots__ = ("_heap", "_valid", "_seq")

    def __init__(self):
        self._heap: List[Tuple] = []   # (-gain, seq, cell, to_part)
        self._valid: dict = {}          # cell -> seq
        self._seq: int = 0

    def push(self, cell: int, gain: float, to_part: int):
        seq = self._seq
        self._seq += 1
        heapq.heappush(self._heap, (-gain, seq, cell, to_part))
        self._valid[cell] = seq

    def invalidate(self, cell: int):
        self._valid.pop(cell, None)

    def pop_best(self) -> Optional[Tuple[int, float, int]]:
        """Return (cell, gain, to_part) or None if empty."""
        while self._heap:
            neg_gain, seq, cell, to_part = heapq.heappop(self._heap)
            if self._valid.get(cell) == seq:
                del self._valid[cell]
                return cell, -neg_gain, to_part
        return None

    @property
    def empty(self) -> bool:
        return not self._valid


# ---------------------------------------------------------------------------
# Initial gain computation
# ---------------------------------------------------------------------------

def _init_gains(
    hg: Hypergraph,
    partition: np.ndarray,
    part_counts: List[np.ndarray],
    k: int,
    cell_to_nets: List[List[int]],
) -> np.ndarray:
    """
    gain_to[v, q] = SOED gain of moving vertex v (currently in partition[v]) to q.
    For q == partition[v]: gain is -inf (sentinel).
    Complexity: O(Σ|net| · k).
    """
    n = hg.num_vertices
    gain_to = np.full((n, k), -np.inf, dtype=float)

    for v in range(n):
        p = int(partition[v])
        for q in range(k):
            if q == p:
                continue
            g = 0.0
            for e_idx in cell_to_nets[v]:
                w   = hg.hyperedge_weights[e_idx]
                pc  = part_counts[e_idx]
                if pc[p] == 1:      # v is the sole cell of p in this net
                    g += w          # uncut from p → SOED decreases
                if pc[q] == 0:      # no cell in q for this net before move
                    g -= w          # new span added → SOED increases
            gain_to[v, q] = g

    return gain_to


# ---------------------------------------------------------------------------
# Incremental gain update after moving v from p to q
# ---------------------------------------------------------------------------

def _update_gains_after_move(
    v: int,
    from_part: int,
    to_part: int,
    hg: Hypergraph,
    partition: np.ndarray,
    part_counts: List[np.ndarray],
    gain_to: np.ndarray,
    locked: np.ndarray,
    cell_to_nets: List[List[int]],
    k: int,
) -> None:
    """
    Exact incremental gain update after v moves from_part → to_part.
    Derived from SOED gain differential — see module docstring.
    """
    p = from_part
    q = to_part

    for e_idx in cell_to_nets[v]:
        w  = hg.hyperedge_weights[e_idx]
        he = hg.hyperedges[e_idx]
        pc = part_counts[e_idx]

        old_pc_p = pc[p]   # includes v (before part_counts update)
        old_pc_q = pc[q]   # before part_counts update

        # Commit the part-count change for this net
        pc[p] -= 1
        pc[q] += 1

        for u in he:
            if u == v or locked[u]:
                continue
            pu = int(partition[u])

            # -----------------------------------------------------------------
            # ΔA = change in w(n)*(pc[pu]==1) term of gain_to[u, r] for all r≠pu
            # -----------------------------------------------------------------
            delta_A = 0.0
            if pu == p:
                # pc[p] just decreased by 1
                if old_pc_p == 2:        # was 2, now 1: u became sole member
                    delta_A = +w
                # (old_pc_p == 1 impossible: u and v both in p → pc_p ≥ 2)
            elif pu == q:
                # pc[q] just increased by 1
                if old_pc_q == 1:        # was 1 (only u), now 2: u loses sole status
                    delta_A = -w

            # -----------------------------------------------------------------
            # ΔB = change in w(n)*(pc[r]==0) term — per target partition r
            # -----------------------------------------------------------------
            # r == p: pc[p] changed
            delta_B_p = 0.0
            if old_pc_p == 1:            # pc[p]: 1 → 0, so (pc[r==p]==0) True
                delta_B_p = +w           # Δ in -w*(pc[r]==0) = -w*(+1) = -w
                                         # but we subtract ΔB so: gain -= (-w) nope

            # r == q: pc[q] changed
            delta_B_q = 0.0
            if old_pc_q == 0:            # pc[q]: 0 → 1, so (pc[r==q]==0) False
                delta_B_q = -w           # (was True, now False) → gain += w

            # Apply updates to all valid target partitions for u
            if delta_A != 0.0 or delta_B_p != 0.0 or delta_B_q != 0.0:
                for r in range(k):
                    if r == pu:
                        continue
                    dg = delta_A
                    if r == p:
                        dg -= delta_B_p  # gain = +w*(pc[pu]==1) - w*(pc[r]==0)
                    if r == q:
                        dg -= delta_B_q
                    if dg != 0.0:
                        old_val = gain_to[u, r]
                        if old_val > -np.inf:
                            gain_to[u, r] = old_val + dg


# ---------------------------------------------------------------------------
# Balance check helper
# ---------------------------------------------------------------------------

def _check_balance(
    part_w: np.ndarray,
    v_weight: float,
    from_p: int,
    to_p: int,
    upper: float,
    k: int,
) -> bool:
    """True if moving a cell of weight v_weight from from_p to to_p keeps balance."""
    new_to = part_w[to_p] + v_weight
    new_from = part_w[from_p] - v_weight
    return new_to <= upper and new_from >= 0.0


# ---------------------------------------------------------------------------
# Single FM pass
# ---------------------------------------------------------------------------

def _fm_pass(
    hg: Hypergraph,
    partition: np.ndarray,
    cfg: PartitionConfig,
    cell_to_nets: List[List[int]],
    fixed: set,
) -> Tuple[np.ndarray, bool]:
    """
    One full FM pass. Returns (new_partition, improved).
    """
    k = cfg.num_parts
    n = hg.num_vertices
    vw = np.asarray(hg.vertex_weights, dtype=float)
    total_w = vw.sum()
    ideal   = total_w / k
    upper   = ideal * (1.0 + cfg.balance_constraint)

    part_counts = build_partition_counts(hg, partition, k)
    gain_to     = _init_gains(hg, partition, part_counts, k, cell_to_nets)

    part_w = np.array([vw[partition == p].sum() for p in range(k)], dtype=float)

    # Build per-cell best-gain heap (one entry per cell: its best move)
    heap = _GainHeap()
    locked = np.zeros(n, dtype=bool)

    for v in range(n):
        if v in fixed:
            continue
        pu = int(partition[v])
        best_q = -1
        best_g = -np.inf
        for q in range(k):
            if q == pu:
                continue
            g = gain_to[v, q]
            if g > best_g and _check_balance(part_w, vw[v], pu, q, upper, k):
                best_g, best_q = g, q
        if best_q >= 0:
            heap.push(v, best_g, best_q)

    current_part = partition.copy()
    best_soed    = compute_soed(hg, current_part)
    best_idx     = -1
    move_stack: List[Tuple[int, int, int]] = []   # (cell, from_part, to_part)

    step = 0
    while not heap.empty:
        result = heap.pop_best()
        if result is None:
            break
        v, g, q = result
        p = int(current_part[v])

        # Re-validate balance (heap entry may be stale after part_w changed)
        if not _check_balance(part_w, vw[v], p, q, upper, k):
            # Find a valid target partition
            best_q2, best_g2 = -1, -np.inf
            for q2 in range(k):
                if q2 == p:
                    continue
                if gain_to[v, q2] > best_g2 and _check_balance(part_w, vw[v], p, q2, upper, k):
                    best_g2, best_q2 = gain_to[v, q2], q2
            if best_q2 < 0:
                locked[v] = True
                continue
            q = best_q2

        # Execute move
        current_part[v] = q
        part_w[p] -= vw[v]
        part_w[q] += vw[v]
        locked[v] = True
        move_stack.append((v, p, q))

        # Incremental gain update for cells in affected nets
        _update_gains_after_move(
            v, p, q, hg, current_part, part_counts,
            gain_to, locked, cell_to_nets, k,
        )

        # Re-insert unlocked neighbours with updated best moves
        affected_cells: set = set()
        for e_idx in cell_to_nets[v]:
            for u in hg.hyperedges[e_idx]:
                if not locked[u] and u not in fixed:
                    affected_cells.add(u)

        for u in affected_cells:
            heap.invalidate(u)
            pu = int(current_part[u])
            best_q3, best_g3 = -1, -np.inf
            for r in range(k):
                if r == pu:
                    continue
                g3 = gain_to[u, r]
                if g3 > best_g3 and _check_balance(part_w, vw[u], pu, r, upper, k):
                    best_g3, best_q3 = g3, r
            if best_q3 >= 0:
                heap.push(u, best_g3, best_q3)

        # Track cumulative best SOED
        cur_soed = compute_soed(hg, current_part)
        if cur_soed < best_soed:
            best_soed = cur_soed
            best_idx  = len(move_stack) - 1

        step += 1

    # Roll back moves after the best prefix
    if best_idx >= 0:
        for idx in range(len(move_stack) - 1, best_idx, -1):
            cell, frm, to = move_stack[idx]
            current_part[cell] = frm
        return current_part, True

    return partition, False   # no improvement


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FMRefiner:
    """
    k-way Fiduccia-Mattheyses local refinement.

    Usage
    -----
    refiner = FMRefiner()
    improved_partition = refiner.refine(hg, partition, cfg)
    """

    def __init__(self, max_passes: int = 10):
        self.max_passes = max_passes

    def refine(
        self,
        hg: Hypergraph,
        partition: np.ndarray,
        cfg: PartitionConfig,
    ) -> np.ndarray:
        """
        Run FM passes until no improvement or max_passes reached.
        Returns the refined partition.
        """
        cell_to_nets = build_cell_to_nets_index(hg)
        fixed = set(hg.fixed_vertices) if hg.fixed_vertices else set()

        for _ in range(self.max_passes):
            new_part, improved = _fm_pass(hg, partition, cfg, cell_to_nets, fixed)
            if not improved:
                break
            partition = new_part

        # Final balance repair (FM may leave marginal infeasibility)
        return repair_balance(partition, hg, cfg)
