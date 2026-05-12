"""
BaseOptimizer — shared interface and helpers for every metaheuristic.

All algorithms:
  1. Accept a Hypergraph + PartitionConfig.
  2. Return (best_partition: np.ndarray, history: list[float]).
  3. Call repair_balance after every solution update.
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
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def optimize(
        self,
        hg: Hypergraph,
        cfg: PartitionConfig,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Run the optimisation.

        Returns
        -------
        best_partition : np.ndarray of shape (n,), dtype int
        history        : list of best objective values per iteration
        """

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
        """Return (pop, obj_values) — pop shape (pop_size, n)."""
        n = hg.num_vertices
        pop = np.stack([
            self._repair(random_partition(n, cfg.num_parts, rng), hg, cfg, rng)
            for _ in range(self.pop_size)
        ])
        obj = np.array([self._obj(hg, pop[i], cfg) for i in range(self.pop_size)])
        return pop, obj

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
