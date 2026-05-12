"""
Krill Herd (KH) for hypergraph partitioning.

Reference: Gandomi & Alavi (2012) — adapted for discrete search.

Each krill individual maintains a continuous position vector x ∈ [0, k)^n.
The discrete partition is obtained by: partition[i] = floor(x[i]) % k.

Motion components (simplified for combinatorial space):
  N  — induced motion from neighbours (move toward best-ranked neighbours)
  F  — foraging motion (move toward food = centroid of all positions, weighted
        by fitness; also toward personal best)
  D  — physical diffusion (random walk)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer
from objective import Hypergraph, PartitionConfig


def _decode(x: np.ndarray, k: int) -> np.ndarray:
    """Map continuous position to integer partition."""
    return np.floor(np.clip(x, 0, k - 1e-9)).astype(int)


class KHOptimizer(BaseOptimizer):
    """Krill Herd optimiser."""

    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 200,
        n_max: float = 0.01,   # max inertia-scaled neighbourhood motion
        v_f: float = 0.02,     # foraging speed
        d_max: float = 0.005,  # max diffusion amplitude
        **kwargs,
    ):
        super().__init__(pop_size=pop_size, max_iter=max_iter, **kwargs)
        self.n_max = n_max
        self.v_f = v_f
        self.d_max = d_max

    def optimize(
        self,
        hg: Hypergraph,
        cfg: PartitionConfig,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        rng = np.random.default_rng(seed)
        k = cfg.num_parts
        n = hg.num_vertices

        # Continuous positions in [0, k)
        X = rng.uniform(0, k, (self.pop_size, n))
        N = np.zeros_like(X)   # induced motion
        F = np.zeros_like(X)   # foraging motion

        # Evaluate initial population
        parts = np.stack([_decode(X[i], k) for i in range(self.pop_size)])
        parts = np.stack([self._repair(parts[i], hg, cfg, rng) for i in range(self.pop_size)])
        obj = np.array([self._obj(hg, parts[i], cfg) for i in range(self.pop_size)])

        best_part, best_obj = self._best(parts, obj)
        pb_X = X.copy()          # personal best continuous positions
        pb_obj = obj.copy()
        history: List[float] = [best_obj]

        for t in range(1, self.max_iter + 1):
            inertia = 0.9 - 0.5 * t / self.max_iter  # linearly decreasing

            # Objective-based fitness (lower obj -> higher fitness)
            fit_max, fit_min = obj.max(), obj.min()
            if fit_max == fit_min:
                fit_norm = np.ones(self.pop_size)
            else:
                fit_norm = (fit_max - obj) / (fit_max - fit_min + 1e-12)

            # ---- Induced motion: toward best neighbour in Euclidean space ---
            best_idx = int(np.argmin(obj))
            best_X = X[best_idx]
            alpha_best = 2.0 * rng.random(n)            # attraction to global best
            for i in range(self.pop_size):
                alpha_local = 0.0
                local_dir = np.zeros(n)
                dists = np.linalg.norm(X - X[i], axis=1)
                dists[i] = np.inf
                nb_idx = np.argsort(dists)[:5]
                for j in nb_idx:
                    dij = np.linalg.norm(X[j] - X[i]) + 1e-12
                    k_ij = (fit_norm[j] - fit_norm[i]) / dij
                    local_dir += k_ij * (X[j] - X[i])
                    alpha_local += abs(k_ij)
                dir_i = alpha_local * local_dir + alpha_best * (best_X - X[i])
                N[i] = inertia * N[i] + self.n_max * dir_i

            # ---- Foraging motion: toward food (fitness-weighted centroid) ---
            food = np.average(X, axis=0, weights=fit_norm + 1e-12)
            for i in range(self.pop_size):
                beta_food = 2.0 * rng.random(n) * (fit_norm[i] + 1e-12)
                beta_pb = 2.0 * rng.random(n)
                F[i] = inertia * F[i] + self.v_f * (
                    beta_food * (food - X[i]) + beta_pb * (pb_X[i] - X[i])
                )

            # ---- Physical diffusion: random walk ----------------------------
            D = self.d_max * (2.0 * rng.random((self.pop_size, n)) - 1.0)

            # ---- Update positions -------------------------------------------
            X = X + N + F + D
            X = np.clip(X, 0, k - 1e-9)

            # Decode, repair, evaluate
            parts = np.stack([_decode(X[i], k) for i in range(self.pop_size)])
            parts = np.stack([self._repair(parts[i], hg, cfg, rng) for i in range(self.pop_size)])
            obj = np.array([self._obj(hg, parts[i], cfg) for i in range(self.pop_size)])

            # Update personal bests
            improved = obj < pb_obj
            pb_X[improved] = X[improved]
            pb_obj[improved] = obj[improved]

            # Track global best
            iter_best, iter_obj = self._best(parts, obj)
            if iter_obj < best_obj:
                best_obj = iter_obj
                best_part = iter_best
            history.append(best_obj)

        return best_part, history
