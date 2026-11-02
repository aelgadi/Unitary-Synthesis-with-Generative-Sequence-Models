"""Pareto archive P — UpdateArchive of Algorithm 2 (README).

Keeps the circuits that are non-dominated under the dominance test of the
README: fidelity
up, CNOT count, depth and total gate count down, with strict improvement in at
least one. Three specialisations of plain Pareto dominance: a circuit is admitted only
if F >= fidelity_floor; fidelities within fidelity_tol count as tied, so
neither circuit dominates the other; and circuits with the same canonical hash
(the same simplified gate stream) are kept once, preferring the better
representative. Past max_size the entry in the most crowded region of
objective space is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Circuit:
    """An archive entry: a circuit's gates, fidelity F, two-qubit count C and depth D."""

    fidelity: float
    cnot_count: int
    depth: int
    total_gates: int
    tokens: np.ndarray      # BOS-prefixed token row
    angles: np.ndarray      # one angle per action position, aligned with tokens[1:]
    epoch: int
    canonical_hash: str | None = None


class ParetoArchive:
    def __init__(self, max_size: int, fidelity_floor: float, fidelity_tol: float):
        self.max_size = max_size
        self.fidelity_floor = fidelity_floor
        self.fidelity_tol = float(fidelity_tol)
        self._circuits: list[Circuit] = []

    def __len__(self) -> int:
        return len(self._circuits)

    @property
    def circuits(self) -> list[Circuit]:
        """Entries in insertion order."""
        return list(self._circuits)

    def sorted(self) -> list[Circuit]:
        """Entries by fidelity, ties ordered by compactness."""
        return sorted(
            self._circuits, key=lambda c: (-c.fidelity, c.cnot_count, c.depth, c.total_gates)
        )

    def best_by_fidelity(self) -> Circuit | None:
        if not self._circuits:
            return None
        best = max(c.fidelity for c in self._circuits)
        tied = [c for c in self._circuits if np.isclose(c.fidelity, best, rtol=0.0, atol=1.0e-7)]
        return min(tied, key=lambda c: (c.cnot_count, c.depth, c.total_gates, -c.fidelity))

    def best_by_cnot(self, min_fidelity: float) -> Circuit | None:
        """Lowest-CNOT entry among those with fidelity >= ``min_fidelity``."""
        eligible = [c for c in self._circuits if c.fidelity >= min_fidelity]
        if not eligible:
            return None
        return min(eligible, key=lambda c: (c.cnot_count, c.depth, c.total_gates, -c.fidelity))

    @staticmethod
    def _prefer(candidate: Circuit, incumbent: Circuit, tol: float) -> bool:
        """Whether a same-hash candidate is the better representative."""
        if candidate.fidelity > incumbent.fidelity + tol:
            return True
        if incumbent.fidelity > candidate.fidelity + tol:
            return False
        return (
            (candidate.cnot_count, candidate.depth, candidate.total_gates, -candidate.fidelity, candidate.epoch)
            < (incumbent.cnot_count, incumbent.depth, incumbent.total_gates, -incumbent.fidelity, incumbent.epoch)
        )

    def add(self, circuits: list[Circuit]) -> None:
        """UpdateArchive: insert admissible candidates and keep the non-dominated
        set of the archive plus candidates, deduplicated by canonical hash."""
        candidates = [c for c in circuits if c.fidelity >= self.fidelity_floor]
        if not candidates:
            return

        combined: list[Circuit] = []
        hash_to_idx: dict[str, int] = {}
        for c in [*self._circuits, *candidates]:
            h = c.canonical_hash
            if h is None:
                combined.append(c)
                continue
            j = hash_to_idx.get(h)
            if j is None:
                hash_to_idx[h] = len(combined)
                combined.append(c)
            elif self._prefer(c, combined[j], self.fidelity_tol):
                combined[j] = c

        n = len(combined)
        f = np.fromiter((c.fidelity for c in combined), dtype=np.float32, count=n)
        d = np.fromiter((c.depth for c in combined), dtype=np.int32, count=n)
        g = np.fromiter((c.total_gates for c in combined), dtype=np.int32, count=n)
        k = np.fromiter((c.cnot_count for c in combined), dtype=np.int32, count=n)
        f_i, d_i, g_i, k_i = f[:, None], d[:, None], g[:, None], k[:, None]
        f_j, d_j, g_j, k_j = f[None, :], d[None, :], g[None, :], k[None, :]
        tied = np.abs(f_i - f_j) <= np.float32(self.fidelity_tol)
        better_or_equal = (f_j >= f_i) & (d_j <= d_i) & (g_j <= g_i) & (k_j <= k_i)
        strictly_better = (f_j > f_i) | (d_j < d_i) | (g_j < g_i) | (k_j < k_i)
        dominated = (better_or_equal & strictly_better & ~tied).any(axis=1)
        self._circuits = [combined[i] for i in np.flatnonzero(~dominated)]

        while self.max_size > 0 and len(self._circuits) > self.max_size:
            self._prune_by_crowding()

    def _prune_by_crowding(self) -> None:
        """Drop the entry with the smallest crowding distance (extremes are kept)."""
        n = len(self._circuits)
        crowding = np.zeros(n, dtype=np.float64)
        for vals in (
            [c.fidelity for c in self._circuits],
            [float(c.depth) for c in self._circuits],
            [float(c.total_gates) for c in self._circuits],
            [float(c.cnot_count) for c in self._circuits],
        ):
            vals = np.array(vals)
            order = np.argsort(vals)
            val_range = float(vals[order[-1]] - vals[order[0]])
            if val_range == 0.0:
                continue
            crowding[order[0]] = np.inf
            crowding[order[-1]] = np.inf
            for i in range(1, n - 1):
                crowding[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
        self._circuits.pop(int(np.argmin(crowding)))
