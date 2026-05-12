"""
Earthworm Optimization Algorithm (EWA) for hypergraph partitioning.

Reference: Wang et al. (2018)  doi:10.1155/2018/6319898

Biological metaphor:
  - Reproduction: an earthworm produces two offspring (Caenorhabditis-style)
    by crossing-over between its position and the global best.
  - Cauchy mutation: random long-jump perturbation drawn from a Cauchy dist.
  - Sigmoid-based selection of the better offspring.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer
from objective import Hypergraph, PartitionConfig


class EWAOptimizer(BaseOptimizer):
    """Earthworm Optimization Algorithm."""

    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 200,
        beta: float = 0.98,       # similarity factor (crossover weight)
        gamma: float = 0.9,       # Cauchy mutation decay
        alpha: float = 0.99,      # similarity factor decay
        **kwargs,
    ):
        super().__init__(pop_size=pop_size, max_iter=max_iter, **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha

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

        beta = self.beta

        for t in range(1, self.max_iter + 1):
            new_pop = pop.copy()
            new_obj = obj.copy()

            for i in range(self.pop_size):
                # ---- Reproduction: crossover with global best ---------------
                #  For each locus: keep own gene with prob beta; take best's otherwise
                mask = rng.random(n) < beta
                x1 = pop[i].copy()
                x1[~mask] = best_part[~mask]
                x1 = self._repair(x1, hg, cfg, rng)
                obj1 = self._obj(hg, x1, cfg)

                # ---- Cauchy mutation ----------------------------------------
                # Discrete analogue: randomly reassign loci with probability
                # proportional to Cauchy spread; heavier tail than Gaussian.
                cauchy_scale = max(1, int(self.gamma ** t * n * 0.2))
                x2 = pop[i].copy()
                n_flip = min(n, rng.integers(1, cauchy_scale + 1))
                flip_idx = rng.choice(n, size=n_flip, replace=False)
                x2[flip_idx] = rng.integers(0, k, size=n_flip)
                x2 = self._repair(x2, hg, cfg, rng)
                obj2 = self._obj(hg, x2, cfg)

                # Select better child; tie-break to existing if equal
                if obj1 <= obj2:
                    cand, c_obj = x1, obj1
                else:
                    cand, c_obj = x2, obj2

                if c_obj <= obj[i]:
                    new_pop[i] = cand
                    new_obj[i] = c_obj

            pop, obj = new_pop, new_obj
            beta *= self.alpha   # decay similarity factor

            iter_best, iter_obj = self._best(pop, obj)
            if iter_obj < best_obj:
                best_obj = iter_obj
                best_part = iter_best
            history.append(best_obj)

        return best_part, history
