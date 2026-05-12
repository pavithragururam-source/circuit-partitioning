"""
Harris Hawks Optimization (HHO) for hypergraph partitioning.

Reference: Heidari et al. (2019)  doi:10.1016/j.future.2019.02.028

Biological metaphor:
  Harris hawks cooperatively hunt prey (rabbit) using four attack strategies
  that switch based on the rabbit's escape energy E:

    |E| >= 1   — Exploration (random perching or rabbit-tracking)
    0.5 ≤ |E| < 1 — Soft besiege (encircle the rabbit, no rapid dive)
    |E| < 0.5  — Hard besiege (closing encirclement) with or without
                  progressive rapid dives (Levy flight)

Discrete adaptation:
  - Each locus is copied from the rabbit (best solution) or a random hawk,
    with a probability proportional to escape energy and the strategy chosen.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer
from objective import Hypergraph, PartitionConfig, random_partition


def _levy(rng: np.random.Generator, n: int, beta: float = 1.5) -> np.ndarray:
    """Return an n-length vector of Levy flight magnitudes in [0, 1]."""
    import math
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (num / den) ** (1 / beta)
    u = rng.normal(0, sigma, n)
    v = rng.normal(0, 1, n)
    steps = np.abs(u / (np.abs(v) ** (1 / beta)))
    return np.clip(steps / (steps.max() + 1e-12), 0.0, 1.0)


class HHOOptimizer(BaseOptimizer):
    """Harris Hawks Optimization."""

    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 200,
        **kwargs,
    ):
        super().__init__(pop_size=pop_size, max_iter=max_iter, **kwargs)

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
        rabbit, rabbit_obj = self._best(pop, obj)
        history: List[float] = [rabbit_obj]

        for t in range(1, self.max_iter + 1):
            # Escaping energy (decreases from 2 to 0 over iterations)
            E0 = 2.0 * rng.random() - 1.0      # initial energy ∈ [-1, 1]
            E = 2.0 * E0 * (1.0 - t / self.max_iter)

            for i in range(self.pop_size):
                E_abs = abs(E)

                if E_abs >= 1.0:
                    # ---- Exploration ----------------------------------------
                    if rng.random() < 0.5:
                        # Random tall perch: copy from random hawk
                        r_idx = rng.integers(0, self.pop_size)
                        mask = rng.random(n) < rng.random()
                        new_x = pop[i].copy()
                        new_x[mask] = pop[r_idx, mask]
                    else:
                        # Rabbit-tracking: partial copy from rabbit
                        r_rabbit = random_partition(n, k, rng)
                        mask = rng.random(n) < rng.random()
                        new_x = pop[i].copy()
                        new_x[mask] = r_rabbit[mask]

                elif rng.random() >= 0.5:
                    # ---- Soft besiege (no rapid dive) -----------------------
                    copy_rate = E_abs * rng.random()
                    mask = rng.random(n) < copy_rate
                    new_x = pop[i].copy()
                    new_x[mask] = rabbit[mask]

                elif E_abs >= 0.5:
                    # ---- Soft besiege with Levy-flight rapid dive -----------
                    levy = _levy(rng, n)
                    mask = rng.random(n) < levy
                    new_x = pop[i].copy()
                    new_x[mask] = rabbit[mask]

                else:
                    # ---- Hard besiege with Levy-flight rapid dive -----------
                    # Strong convergence — copy most loci from rabbit
                    levy = _levy(rng, n)
                    copy_rate = 1.0 - E_abs + 0.5 * rng.random()
                    mask = (rng.random(n) < copy_rate) | (rng.random(n) < levy)
                    new_x = pop[i].copy()
                    new_x[mask] = rabbit[mask]

                new_x = self._repair(new_x, hg, cfg, rng)
                new_obj = self._obj(hg, new_x, cfg)
                if new_obj <= obj[i]:
                    pop[i] = new_x
                    obj[i] = new_obj

            iter_best, iter_obj = self._best(pop, obj)
            if iter_obj < rabbit_obj:
                rabbit_obj = iter_obj
                rabbit = iter_best
            history.append(rabbit_obj)

        return rabbit, history
