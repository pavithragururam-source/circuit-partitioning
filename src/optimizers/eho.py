"""
Elephant Herding Optimization (EHO) for hypergraph partitioning.

Reference: Wang et al. (2016)  doi:10.1109/ISCBI.2015.8

Biological metaphor:
  - Clan updating: each elephant in a clan moves toward the clan matriarch
    (the best elephant in its clan) with a coefficient alpha ∈ (0,1).
  - Separating: the worst elephant of the worst clan is removed from the
    clan and replaced with a new random individual (alpha_s update).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer
from objective import Hypergraph, PartitionConfig, random_partition


class EHOOptimizer(BaseOptimizer):
    """Elephant Herding Optimization."""

    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 200,
        n_clans: int = 5,       # number of elephant clans
        alpha: float = 0.5,     # clan updating coefficient
        beta: float = 0.1,      # matriarch contribution coefficient
        **kwargs,
    ):
        super().__init__(pop_size=pop_size, max_iter=max_iter, **kwargs)
        self.n_clans = min(n_clans, pop_size)
        self.alpha = alpha
        self.beta = beta

    def optimize(
        self,
        hg: Hypergraph,
        cfg: PartitionConfig,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        rng = np.random.default_rng(seed)
        k = cfg.num_parts
        n = hg.num_vertices

        pop, obj = self._init_population(hg, cfg, rng)
        best_part, best_obj = self._best(pop, obj)
        history: List[float] = [best_obj]

        # Assign each elephant to a clan (round-robin by sorted rank)
        clan_size = self.pop_size // self.n_clans

        for _ in range(self.max_iter):
            # Sort by objective, then assign clans in rank order
            order = np.argsort(obj)
            pop, obj = pop[order], obj[order]

            new_pop = pop.copy()

            # ---- Clan Updating operator -------------------------------------
            for c in range(self.n_clans):
                start = c * clan_size
                end = start + clan_size if c < self.n_clans - 1 else self.pop_size
                clan_idx = np.arange(start, end)
                matriarch_idx = clan_idx[0]     # best elephant in clan

                for i in clan_idx:
                    # Blend current elephant with matriarch stochastically
                    mask = rng.random(n) < self.alpha
                    new_x = pop[i].copy()
                    new_x[mask] = pop[matriarch_idx, mask]
                    # Matriarch itself uses clan centroid-like update
                    if i == matriarch_idx:
                        # Replace matriarch loci with random values based on beta
                        m_mask = rng.random(n) < self.beta
                        new_x[m_mask] = rng.integers(0, k, size=int(m_mask.sum()))
                    new_pop[i] = self._repair(new_x, hg, cfg, rng)

            # ---- Separating operator ----------------------------------------
            # Replace the worst elephant in the worst clan with a random one
            worst_idx = self.pop_size - 1
            new_pop[worst_idx] = self._repair(
                random_partition(n, k, rng), hg, cfg, rng
            )

            pop = new_pop
            obj = np.array([self._obj(hg, pop[i], cfg) for i in range(self.pop_size)])

            iter_best, iter_obj = self._best(pop, obj)
            if iter_obj < best_obj:
                best_obj = iter_obj
                best_part = iter_best
            history.append(best_obj)

        return best_part, history
