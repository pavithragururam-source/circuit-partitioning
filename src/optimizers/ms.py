"""
Moth Search (MS) Algorithm for hypergraph partitioning.

Reference: Wang (2018)  doi:10.1016/j.ins.2018.04.046

Biological metaphor:
  - Moths fly toward a light source (global best) using a Levy-flight spiral
    in the moth-flight mode (exploitation).
  - Dispersed moths perform random Levy-flight jumps for global exploration.
  - A threshold based on the moth's fitness selects the operative mode.

Discrete adaptation:
  - Levy-flight step → probability of copying a locus from the best solution.
  - Larger step lengths increase the number of loci that "fly toward" the best.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer
from objective import Hypergraph, PartitionConfig


def _levy_step(rng: np.random.Generator, beta: float = 1.5) -> float:
    """Generate a scalar Levy flight step magnitude (Mantegna's algorithm)."""
    import math
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (num / den) ** (1 / beta)
    u = rng.normal(0, sigma_u)
    v = rng.normal(0, 1)
    step = u / (abs(v) ** (1 / beta))
    return float(np.clip(abs(step), 0.0, 1.0))


class MSOptimizer(BaseOptimizer):
    """Moth Search optimiser."""

    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 200,
        levy_beta: float = 1.5,    # Levy exponent
        p_levy: float = 0.3,       # fraction of moths using Levy exploration
        **kwargs,
    ):
        super().__init__(pop_size=pop_size, max_iter=max_iter, **kwargs)
        self.levy_beta = levy_beta
        self.p_levy = p_levy

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

        n_levy = max(1, int(self.p_levy * self.pop_size))

        for t in range(1, self.max_iter + 1):
            # Adaptive light-source attraction: decreases as iterations progress
            attraction = 1.0 - t / self.max_iter   # 1 → 0

            for i in range(self.pop_size):
                if i < n_levy:
                    # ---- Levy-flight exploration ----------------------------
                    step = _levy_step(rng, self.levy_beta)
                    n_flip = max(1, int(step * n))
                    new_x = pop[i].copy()
                    flip_idx = rng.choice(n, size=n_flip, replace=False)
                    new_x[flip_idx] = rng.integers(0, k, size=n_flip)
                else:
                    # ---- Moth-flight: spiral toward light (global best) -----
                    copy_rate = attraction + _levy_step(rng, self.levy_beta) * 0.2
                    copy_rate = float(np.clip(copy_rate, 0.0, 1.0))
                    mask = rng.random(n) < copy_rate
                    new_x = pop[i].copy()
                    new_x[mask] = best_part[mask]

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
