"""
Slime Mould Algorithm (SMA) for hypergraph partitioning.

Reference: Li et al. (2020)  doi:10.1016/j.future.2020.03.055

Biological metaphor:
  Physarum polycephalum (slime mould) propagates its vein network toward food.
  The algorithm models three behaviours:

    Approach food   — move toward the best (food-rich) position.
    Wrap food       — oscillate around the best position with vein weights W.
    Grope for food  — random search when food is scarce.

Discrete adaptation:
  - vein weight W[i] ∈ [0, 1] controls how strongly individual i mimics best.
  - At each step, every locus is either copied from the best or randomised.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer
from objective import Hypergraph, PartitionConfig, random_partition


class SMAOptimizer(BaseOptimizer):
    """Slime Mould Algorithm."""

    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 200,
        z: float = 0.03,    # probability threshold for random exploration
        **kwargs,
    ):
        super().__init__(pop_size=pop_size, max_iter=max_iter, **kwargs)
        self.z = z

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

        for t in range(1, self.max_iter + 1):
            # ---- Compute vein weights ----------------------------------------
            # Rank individuals by objective (lower = better = rank 0)
            sort_idx = np.argsort(obj)
            obj_best = obj[sort_idx[0]]
            obj_worst = obj[sort_idx[-1]]

            W = np.ones(self.pop_size)
            half = self.pop_size // 2
            for rank, idx in enumerate(sort_idx):
                if obj_worst == obj_best:
                    W[idx] = 1.0
                elif rank < half:   # better half: W > 1
                    W[idx] = 1.0 + rng.random() * np.log10(
                        1.0 + (obj_best - obj[idx]) / (obj_worst - obj_best + 1e-12)
                    )
                else:               # worse half: W < 1
                    W[idx] = 1.0 - rng.random() * np.log10(
                        1.0 + (obj[idx] - obj_best) / (obj_worst - obj_best + 1e-12)
                    )

            # ---- Update step ------------------------------------------------
            a = np.arctanh(1.0 - t / self.max_iter)   # decreasing oscillation amplitude

            for i in range(self.pop_size):
                if rng.random() < self.z:
                    # Grope: full random solution
                    pop[i] = self._repair(random_partition(n, k, rng), hg, cfg, rng)
                    obj[i] = self._obj(hg, pop[i], cfg)
                    continue

                p = np.tanh(abs(obj[i] - obj_best))   # ∈ [0, 1]

                new_x = pop[i].copy()
                r = rng.random(n)

                if rng.random() < p:
                    # Approach food: oscillate toward best with weight W[i]
                    copy_rate = W[i] * a
                    copy_rate = float(np.clip(copy_rate, 0.0, 1.0))
                    mask = r < copy_rate
                    new_x[mask] = best_part[mask]
                else:
                    # Wrap: blend two random individuals
                    a_idx = rng.integers(0, self.pop_size)
                    b_idx = rng.integers(0, self.pop_size)
                    mask_a = r < 0.5
                    new_x[mask_a] = pop[a_idx, mask_a]
                    new_x[~mask_a] = pop[b_idx, ~mask_a]

                new_x = self._repair(new_x, hg, cfg, rng)
                new_obj = self._obj(hg, new_x, cfg)
                if new_obj <= obj[i]:
                    pop[i] = new_x
                    obj[i] = new_obj

            iter_best, iter_obj = self._best(pop, obj)
            if iter_obj < best_obj:
                best_obj = iter_obj
                best_part = iter_best
            history.append(best_obj)

        return best_part, history
