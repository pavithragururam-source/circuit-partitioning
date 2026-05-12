"""
Artificial Bee Colony (ABC) for hypergraph partitioning.

Reference: Karaboga & Basturk (2007) — adapted for discrete combinatorial search.

Phases
------
Employed bees  : each food source is locally perturbed; greedy selection.
Onlooker bees  : fitness-proportionate selection → local perturbation.
Scout bees     : abandoned food sources (stagnant for `limit` trials)
                 are replaced by fresh random solutions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer
from objective import Hypergraph, PartitionConfig


class ABCOptimizer(BaseOptimizer):
    """Artificial Bee Colony optimiser."""

    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 200,
        limit: int = 50,       # abandonment threshold per food source
        perturb_rate: float = 0.15,  # fraction of loci changed per trial
        **kwargs,
    ):
        super().__init__(pop_size=pop_size, max_iter=max_iter, **kwargs)
        self.limit = limit
        self.perturb_rate = perturb_rate

    def optimize(
        self,
        hg: Hypergraph,
        cfg: PartitionConfig,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        rng = np.random.default_rng(seed)
        k = cfg.num_parts
        n_swaps = max(1, int(self.perturb_rate * hg.num_vertices))

        pop, obj = self._init_population(hg, cfg, rng)
        trial = np.zeros(self.pop_size, dtype=int)
        best_part, best_obj = self._best(pop, obj)
        history: List[float] = [best_obj]

        for _ in range(self.max_iter):

            # ---- Employed bee phase -----------------------------------------
            for i in range(self.pop_size):
                cand = self._repair(
                    self._random_swap(pop[i], k, rng, n_swaps), hg, cfg, rng
                )
                c_obj = self._obj(hg, cand, cfg)
                if c_obj <= obj[i]:
                    pop[i], obj[i] = cand, c_obj
                    trial[i] = 0
                else:
                    trial[i] += 1

            # ---- Onlooker bee phase -----------------------------------------
            # Fitness = 1/(1+obj); normalise to probabilities
            fitness = 1.0 / (1.0 + obj)
            prob = fitness / fitness.sum()
            for _ in range(self.pop_size):
                i = int(rng.choice(self.pop_size, p=prob))
                cand = self._repair(
                    self._random_swap(pop[i], k, rng, n_swaps), hg, cfg, rng
                )
                c_obj = self._obj(hg, cand, cfg)
                if c_obj <= obj[i]:
                    pop[i], obj[i] = cand, c_obj
                    trial[i] = 0
                else:
                    trial[i] += 1

            # ---- Scout bee phase --------------------------------------------
            for i in range(self.pop_size):
                if trial[i] >= self.limit:
                    from objective import random_partition
                    pop[i] = self._repair(
                        random_partition(hg.num_vertices, k, rng), hg, cfg, rng
                    )
                    obj[i] = self._obj(hg, pop[i], cfg)
                    trial[i] = 0

            # ---- Track global best ------------------------------------------
            iter_best, iter_obj = self._best(pop, obj)
            if iter_obj < best_obj:
                best_obj = iter_obj
                best_part = iter_best
            history.append(best_obj)

        return best_part, history
