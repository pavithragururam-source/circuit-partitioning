"""
BaseOptimizer — shared interface and helpers for every metaheuristic.

All algorithms:
  1. Accept a Hypergraph + PartitionConfig.
  2. Return (best_partition: np.ndarray, history: list[float]).
  3. Call repair_balance after every solution update.

Expert additions
----------------
* run()            — public entry point that wraps optimize() with optional
                     multilevel coarsening (HEM) and FM post-refinement.
* _spectral_init_one() — seeds one population member from the Fiedler
                         spectral embedding (clique-expansion Laplacian +
                         k-means). Falls back to random for graphs > 2000 V.
* _init_population()   — uses spectral seed as the first population member
                         when the graph is small enough.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from objective import (
    Hypergraph, PartitionConfig,
    evaluate, repair_balance, random_partition,
)


class BaseOptimizer(ABC):
    """Abstract base for all circuit-partitioning metaheuristics."""

    def __init__(self, pop_size: int = 30, max_iter: int = 200, **_):
        self.pop_size = pop_size
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # Public entry point — call this from main.py
    # ------------------------------------------------------------------

    def run(
        self,
        hg: Hypergraph,
        cfg: PartitionConfig,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Full pipeline:
          [optional multilevel coarsening]
          → metaheuristic optimize()
          → [optional FM post-refinement]
          → return (partition, history)

        Controlled by cfg.multilevel and cfg.fm_refinement.
        """
        rng = np.random.default_rng(seed)

        if cfg.multilevel and hg.num_vertices > cfg.coarsen_to:
            return self._run_multilevel(hg, cfg, rng, seed)

        partition, history = self.optimize(hg, cfg, seed=seed)

        if cfg.fm_refinement:
            from fm_refine import FMRefiner
            partition = FMRefiner(max_passes=cfg.fm_max_passes).refine(
                hg, partition, cfg
            )

        return partition, history

    # ------------------------------------------------------------------
    # Abstract algorithm method — implemented by each subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def optimize(
        self,
        hg: Hypergraph,
        cfg: PartitionConfig,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Run the metaheuristic on hg.

        Returns
        -------
        best_partition : np.ndarray of shape (n,), dtype int
        history        : list of best objective values per iteration
        """

    # ------------------------------------------------------------------
    # Multilevel pipeline helper
    # ------------------------------------------------------------------

    def _run_multilevel(
        self,
        hg: Hypergraph,
        cfg: PartitionConfig,
        rng: np.random.Generator,
        seed: Optional[int],
    ) -> Tuple[np.ndarray, List[float]]:
        """
        HEM coarsening → optimize on coarsest graph → uncoarsen + FM.
        """
        from coarsening import CoarseningHierarchy
        from fm_refine import FMRefiner

        refiner = FMRefiner(max_passes=cfg.fm_max_passes) if cfg.fm_refinement else None
        hier = CoarseningHierarchy()
        coarsest = hier.coarsen(hg, cfg, rng)

        coarse_part, history = self.optimize(coarsest, cfg, seed=seed)
        final_part = hier.uncoarsen(coarse_part, hg, cfg, refiner)

        return final_part, history

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _obj(hg: Hypergraph, part: np.ndarray, cfg: PartitionConfig) -> float:
        return evaluate(hg, part, cfg)["objective"]

    @staticmethod
    def _repair(part: np.ndarray, hg: Hypergraph, cfg: PartitionConfig,
                rng: np.random.Generator) -> np.ndarray:
        return repair_balance(part, hg, cfg, rng)

    def _init_population(
        self,
        hg: Hypergraph,
        cfg: PartitionConfig,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (pop, obj_values) — pop shape (pop_size, n).

        The first slot is filled with a spectral seed (Fiedler embedding)
        when the graph is small enough (≤ 2000 vertices).  The remainder
        are uniformly random, balance-repaired.
        """
        n = hg.num_vertices
        solutions: List[np.ndarray] = []

        # One spectral seed — gives the population a high-quality start
        if n <= 2000:
            spec = self._spectral_init_one(hg, cfg, rng)
            solutions.append(self._repair(spec, hg, cfg, rng))

        while len(solutions) < self.pop_size:
            solutions.append(
                self._repair(random_partition(n, cfg.num_parts, rng), hg, cfg, rng)
            )

        pop = np.stack(solutions)
        obj = np.array([self._obj(hg, pop[i], cfg) for i in range(self.pop_size)])
        return pop, obj

    @staticmethod
    def _spectral_init_one(
        hg: Hypergraph,
        cfg: PartitionConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Build a partition from the Fiedler spectral embedding.

        Steps
        -----
        1. Construct clique-expansion weighted adjacency (same as HEM's _build_adj).
        2. Form the normalized graph Laplacian L = D - A.
        3. Compute the k smallest non-trivial eigenvectors (spectral embedding).
        4. k-means on the embedding (20 iterations) → initial partition.

        Falls back silently to a random partition on any error or if
        the graph is too large (>2000 V) for dense eigen-decomposition.
        """
        try:
            from coarsening import _build_adj
            k = cfg.num_parts
            n = hg.num_vertices

            if n > 2000:
                raise ValueError("graph too large for dense spectral init")

            adj = _build_adj(hg)

            # Dense adjacency matrix
            A = np.zeros((n, n), dtype=float)
            for u, nbrs in adj.items():
                for v, w in nbrs.items():
                    A[u, v] = w
            deg = A.sum(axis=1)
            L = np.diag(deg) - A

            # k smallest eigenvalues / eigenvectors (skip index 0: zero mode)
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Take columns 1..k as spectral embedding (n × (k-1))
            emb = eigenvectors[:, 1:k] if k > 1 else eigenvectors[:, :1]

            if emb.shape[1] == 0:
                raise ValueError("degenerate embedding (k=1?)")

            # k-means (20 hard iterations, random seeding)
            centers = emb[rng.choice(n, k, replace=False)]
            labels = np.zeros(n, dtype=np.int32)
            for _ in range(20):
                dists = np.stack([
                    np.sum((emb - centers[c]) ** 2, axis=1) for c in range(k)
                ])
                labels = np.argmin(dists, axis=0).astype(np.int32)
                new_centers = np.array([
                    emb[labels == c].mean(axis=0)
                    if np.any(labels == c)
                    else emb[rng.integers(n)]
                    for c in range(k)
                ])
                if np.allclose(centers, new_centers, atol=1e-9):
                    break
                centers = new_centers

            # Guarantee every partition has at least one vertex
            for part in range(k):
                if not np.any(labels == part):
                    labels[int(rng.integers(n))] = part

            return labels

        except Exception:
            return random_partition(hg.num_vertices, cfg.num_parts, rng)

    @staticmethod
    def _best(pop: np.ndarray, obj: np.ndarray):
        idx = int(np.argmin(obj))
        return pop[idx].copy(), float(obj[idx])

    @staticmethod
    def _discrete_move(
        src: np.ndarray,
        ref: np.ndarray,
        k: int,
        step_rate: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Blend two integer solutions probabilistically.
        Each locus takes ref[i]'s value with probability step_rate.
        """
        mask = rng.random(len(src)) < step_rate
        new = src.copy()
        new[mask] = ref[mask]
        return new

    @staticmethod
    def _random_swap(part: np.ndarray, k: int, rng: np.random.Generator,
                     n_swaps: int = 1) -> np.ndarray:
        """Perturb a solution by reassigning n_swaps random loci."""
        new = part.copy()
        idx = rng.integers(0, len(part), size=n_swaps)
        new[idx] = rng.integers(0, k, size=n_swaps)
        return new
