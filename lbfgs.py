"""L-BFGS angle optimisation — Algorithm 6 of the README.

The coordinate sweep is exact per coordinate but cannot leave a stationary
point of the joint objective where no single-angle move improves the fidelity.
After simplification, every archived circuit's angle vector is therefore
refined jointly: ``lbfgs_steps`` iterations of limited-memory BFGS (Liu and
Nocedal, 1989) on J(theta) = 1 - F(V(sigma, theta); U), with the gradient by
automatic differentiation through the circuit.

The archive is padded to one width and refined as a batch: every circuit is
an independent problem, so the two-loop recursion, the curvature history and
the step length are all kept per row. The step length comes from a
backtracking line search on the Armijo condition, and a pair (s, y) enters a
row's history only if it satisfies the curvature condition s.y > 0.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from helpers import DEVICE, REAL_DTYPE, CircuitEvaluator, pad_batch


def _value_and_grad(loss_fn, x):
    x = x.detach().requires_grad_(True)
    f = loss_fn(x)
    (g,) = torch.autograd.grad(f.sum(), x)
    return f.detach(), g


def lbfgs(loss_fn, x0: torch.Tensor, steps: int, memory: int, line_search_steps: int):
    """Batched L-BFGS. ``loss_fn`` maps angles ``(B, W)`` to the per-row losses
    ``(B,)`` of B independent problems. Returns the refined angles and losses."""
    x = x0.detach().clone()
    f, g = _value_and_grad(loss_fn, x)
    S, Y, RHO = [], [], []
    gamma = torch.ones_like(f)                   # scale of the initial inverse Hessian
    for _ in range(steps):
        # Two-loop recursion: r ~ H^{-1} g from the stored (s, y) pairs.
        q = g.clone()
        alphas = []
        for s, y, rho in zip(reversed(S), reversed(Y), reversed(RHO)):
            a = rho * (s * q).sum(-1)
            q = q - a[:, None] * y
            alphas.append(a)
        r = gamma[:, None] * q
        for s, y, rho, a in zip(S, Y, RHO, reversed(alphas)):
            b = rho * (y * r).sum(-1)
            r = r + (a - b)[:, None] * s
        d = -r
        slope = (g * d).sum(-1)
        descent = slope < 0                      # fall back to steepest descent otherwise
        d = torch.where(descent[:, None], d, -g)
        slope = torch.where(descent, slope, -(g * g).sum(-1))

        # Backtracking line search on the Armijo condition, one step per row.
        t = torch.ones_like(f)
        with torch.no_grad():
            for _ in range(line_search_steps):
                ok = loss_fn(x + t[:, None] * d) <= f + 1e-4 * t * slope
                if bool(ok.all()):
                    break
                t = torch.where(ok, t, 0.5 * t)
        x_new = x + t[:, None] * d
        f_new, g_new = _value_and_grad(loss_fn, x_new)
        accept = f_new <= f

        # Curvature pair; rows that failed the curvature condition keep it inert.
        s = torch.where(accept[:, None], x_new - x, 0.0)
        y = torch.where(accept[:, None], g_new - g, 0.0)
        sy = (s * y).sum(-1)
        curved = sy > 1e-10
        safe_sy = torch.where(curved, sy, 1.0)
        S.append(s)
        Y.append(y)
        RHO.append(torch.where(curved, 1.0 / safe_sy, 0.0))
        gamma = torch.where(curved, safe_sy / torch.where(curved, (y * y).sum(-1), 1.0), gamma)
        if len(S) > memory:
            S.pop(0), Y.pop(0), RHO.pop(0)

        x = torch.where(accept[:, None], x_new, x)
        f = torch.where(accept, f_new, f)
        g = torch.where(accept[:, None], g_new, g)
    return x, f


def refine_circuits(circuits, evaluator: CircuitEvaluator, cfg):
    """Lift every archived circuit's angles from their stored values; keep the
    lifted angles only where they strictly raise the fidelity. Structure is
    unchanged, so the gate counts carry over; the canonical hash is left for
    the next simplify."""
    if not circuits:
        return []
    tokens, angles = pad_batch(circuits)
    original = evaluator.fidelity_batch(tokens, angles)
    refined = np.array(angles, copy=True)
    refined_fids = np.zeros(len(circuits), dtype=np.float32)
    for start in range(0, len(circuits), cfg.lbfgs_chunk):      # chunked to bound memory
        rows = slice(start, start + cfg.lbfgs_chunk)
        tok = torch.as_tensor(tokens[rows], dtype=torch.long, device=DEVICE)
        x, loss = lbfgs(
            lambda a: 1.0 - evaluator.fidelity(evaluator.unitary(tok, a)),
            torch.as_tensor(angles[rows], dtype=REAL_DTYPE, device=DEVICE),
            cfg.lbfgs_steps, cfg.lbfgs_memory, cfg.lbfgs_line_search_steps,
        )
        refined[rows] = x.to(torch.float32).cpu().numpy()
        refined_fids[rows] = (1.0 - loss).to(torch.float32).cpu().numpy()
    stored = np.where((refined_fids > original)[:, None], refined, angles)
    fids = evaluator.fidelity_batch(tokens, stored)
    print(f"  L-BFGS (K={cfg.lbfgs_steps}): {int((fids > original).sum())}/{len(circuits)} circuits lifted, "
          f"best F {float(original.max()):.6f} -> {float(fids.max()):.6f}")
    return [
        replace(
            c, fidelity=float(fids[i]),
            tokens=np.asarray(c.tokens, dtype=np.int32).copy(),
            angles=np.asarray(stored[i, :np.asarray(c.tokens).size - 1], dtype=np.float32),
            canonical_hash=None,
        )
        for i, c in enumerate(circuits)
    ]
