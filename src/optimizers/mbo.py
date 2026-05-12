"""
Monarch Butterfly Optimization (MBO) for hypergraph partitioning.

Reference: Wang et al. (2019)  doi:10.1007/s00521-015-1923-y

The population is split into two subpopulations (Land 1, Land 2).
Two operators are applied per generation:

  Migration operator   — butterflies in Land 1 update positions by sampling
                         from the migration ratio p (bar-tailed godwit flight).
  Butterfly-adjusting  — butterflies in Land 2 walk toward the best individual.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer
from objective import Hypergraph, PartitionConfig, random_partition


class MBOOptimizer(BaseOptimizer):
    """Monarch Butterfly Optimization."""

    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 200,
        p: float = 5.0 / 12.0,   # migration ratio (fraction assigned to Land 1)
        period: float = 1.2,      # migration period
        bar: float = 5.0 / 12.0, # bar (same as p by convention)
        smax: int = 1,            # max step for butterfly-adjusting walk
        **kwargs,
    ):
        super().__init__(pop_size=pop_size, max_iter=max_iter, **kwargs)
        self.p = p
        self.period = period
        self.bar = bar
        self.smax = smax

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
        # Sort by objective so subpopulation split is deterministic
        order = np.argsort(obj)
        pop, obj = pop[order], obj[order]

        best_part, best_obj = pop[0].copy(), float(obj[0])
        history: List[float] = [best_obj]

        n1 = max(1, int(np.ceil(self.p * self.pop_size)))  # Land 1 size
        n2 = self.pop_size - n1                             # Land 2 size

        for t in range(1, self.max_iter + 1):
            # Sort population (elitism)
            order = np.argsort(obj)
            pop, obj = pop[order], obj[order]

            new_pop = pop.copy()

            # ---- Migration operator (Land 1) --------------------------------
            for i in range(n1):
                new_x = pop[i].copy()
                for j in range(n):
                    r1 = rng.random()
                    if r1 <= self.p:
                        # Pick from Land 1 uniformly
                        r_idx = rng.integers(0, n1)
                        new_x[j] = pop[r_idx, j]
                    else:
                        # Pick from Land 2
                        r_idx = rng.integers(n1, self.pop_size)
                        new_x[j] = pop[r_idx, j]
                new_pop[i] = self._repair(new_x, hg, cfg, rng)

            # ---- Butterfly-adjusting operator (Land 2) ----------------------
            smax_t = max(1, int(self.smax * (1.0 - t / self.max_iter) + 1))
            for i in range(n1, self.pop_size):
                new_x = pop[i].copy()
                for j in range(n):
                    r2 = rng.random()
                    if r2 <= self.bar:
                        # Walk toward global best
                        new_x[j] = best_part[j]
                    else:
                        # Random walk (Levy-like: random integer step)
                        step = rng.integers(0, smax_t + 1)
                        if step > 0:
                            new_x[j] = rng.integers(0, k)
                new_pop[i] = self._repair(new_x, hg, cfg, rng)

            pop = new_pop
            obj = np.array([self._obj(hg, pop[i], cfg) for i in range(self.pop_size)])

            iter_best, iter_obj = self._best(pop, obj)
            if iter_obj < best_obj:
                best_obj = iter_obj
                best_part = iter_best
            history.append(best_obj)

        return best_part, history
