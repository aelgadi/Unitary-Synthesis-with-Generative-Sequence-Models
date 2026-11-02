"""Closed-form angle sweep — Algorithm 4 of the README.

With every other angle held fixed, the process fidelity is a single-frequency
sinusoid in the angle of the rotation at position p, maximised in closed form
by theta_p* = atan2(2 Im(conj(A) B), |A|^2 - |B|^2) with
A = Tr(L_pref U^dag R_rem) and B = Tr(L_pref U^dag R_rem P_p). One pass visits
every position left to right in Gauss–Seidel fashion: L_pref grows with the
freshly updated gate while R_rem is peeled with the pre-pass angle, so each
step is non-decreasing in fidelity.
K passes run on the whole batch of skeletons at once.

The sweep runs inside every rollout (K = cfg.inner_sweeps) and once more over
every archived circuit after training (K = cfg.post_sweeps).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from helpers import BOS, CNOT, DEVICE, REAL_DTYPE, CircuitEvaluator, pad_batch


class CoordinateSweep:
    """``refine(tokens, angles)`` runs ``n_sweeps`` passes over every circuit
    of a batch and returns the fitted angles with their fidelities."""

    def __init__(self, evaluator: CircuitEvaluator, n_sweeps: int):
        if n_sweeps <= 0:
            raise ValueError("n_sweeps must be positive")
        self.evaluator = evaluator
        self.n_sweeps = int(n_sweeps)

    @torch.no_grad()
    def _one_pass(self, tokens: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        ev = self.evaluator
        B, L = tokens.shape

        # R_rem = G_{L-1} ... G_1 from the angles entering this pass.
        R_rem = ev.eye.expand(B, ev.dim, ev.dim)
        for j in range(1, L):
            R_rem = ev.gate(tokens[:, j], angles[:, j]) @ R_rem

        # Forward Gauss–Seidel pass: closed-form update at every rotation,
        # advance L_pref with the new gate, peel the next old gate off R_rem.
        L_pref = ev.eye.expand(B, ev.dim, ev.dim)
        new_angles = angles.clone()
        for p in range(L):
            tok = tokens[:, p]
            gtype = ev.tok_gate_type[tok]
            is_rot = gtype < CNOT
            P_p = ev.pauli_full[torch.where(is_rot, gtype, 0), ev.tok_qubit0[tok]]

            M = L_pref @ ev.u_target_dag @ R_rem
            A = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)
            Bp = torch.einsum("bij,bji->b", M, P_p)                # Tr(M P_p)
            X = (A.abs() ** 2 - Bp.abs() ** 2).to(REAL_DTYPE)
            Y = (2.0 * (A.conj() * Bp).imag).to(REAL_DTYPE)
            new_theta = torch.where(is_rot, torch.atan2(Y, X), angles[:, p])
            new_angles[:, p] = new_theta

            L_pref = ev.gate(tok, new_theta) @ L_pref
            if p + 1 < L:
                G_next_old = ev.gate(tokens[:, p + 1], angles[:, p + 1])
                R_rem = R_rem @ G_next_old.conj().transpose(-2, -1)
        return new_angles

    def refine(self, tokens: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(fidelities, refined_angles)`` after ``n_sweeps`` passes."""
        if tokens.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros(angles.shape, dtype=np.float32)
        tok = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=DEVICE)
        ang = torch.as_tensor(np.asarray(angles), dtype=REAL_DTYPE, device=DEVICE)
        for _ in range(self.n_sweeps):
            ang = self._one_pass(tok, ang)
        refined = ang.to(torch.float32).cpu().numpy()
        fids = self.evaluator.fidelity_batch(tokens, refined)
        return fids, refined


def refine_circuits(circuits, evaluator: CircuitEvaluator, n_sweeps: int):
    """Re-sweep every archived circuit from its stored angles; keep the swept
    angles where they do not lower the fidelity. Structure is unchanged, so the
    gate counts carry over; the canonical hash is left for the next simplify."""
    if not circuits:
        return []
    tokens, angles = pad_batch(circuits)
    original = evaluator.fidelity_batch(tokens, angles)
    refined_fids, refined = CoordinateSweep(evaluator, n_sweeps).refine(tokens, angles)
    stored = np.where((refined_fids >= original)[:, None], refined, angles)
    fids = evaluator.fidelity_batch(tokens, stored)
    bos = np.asarray([BOS], dtype=np.int32)
    return [
        replace(
            c, fidelity=float(fids[i]),
            tokens=np.concatenate([bos, tokens[i]], axis=0),
            angles=np.asarray(stored[i], dtype=np.float32), canonical_hash=None,
        )
        for i, c in enumerate(circuits)
    ]
