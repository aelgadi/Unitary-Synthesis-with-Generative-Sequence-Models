"""Training loop — TrainingIteration (Algorithm 2 of the README) for T iterations.

Each iteration samples a group of N skeletons from the policy (Algorithm 3),
fits every skeleton's angles with the closed-form sweep (Algorithm 4), scores
it with the cost (cost.py), takes the GRPO step (grpo.py) and retains the
non-dominated circuits in the Pareto archive (archive.py). The
policy is trained from scratch for every target and never sees the target
itself; the reward alone shapes it.

Two implementation details beyond the pseudocode. Every sampled row is
canonicalised by Algorithm 5 before it is scored and archived, so the CNOT
count entering the reward and the archive is that of the simplified skeleton.
And the batch is truncated to the length of its longest circuit before the
sweep and the GRPO pass; this changes no number, because the PAD suffix is the
identity for the unitary and the sweep and is excluded from the loss, but it
keeps the work proportional to the circuits actually sampled.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

import grpo
from archive import Circuit, ParetoArchive
from cost import cost
from helpers import (
    DEVICE, CircuitEvaluator, cnot_lower_bound, cnot_upper_bound, context_tokens,
    sequence_lengths,
)
from policy import GatePolicy, sample, token_tables
from simplify import simplify_sequence
from sweep import CoordinateSweep


def inverse_temperature(epoch: int, cfg) -> float:
    """Cyclic cosine schedule of the inverse temperature, with a period of half the run;
    the first iteration samples at the midpoint of [beta_min, beta_max]."""
    mid = (cfg.beta_max + cfg.beta_min) / 2
    if epoch == 0:
        return mid
    period = max(1, cfg.max_epochs // 2)
    return mid - (cfg.beta_max - cfg.beta_min) / 2 * math.cos(2 * math.pi * epoch / period)


def train(cfg, evaluator: CircuitEvaluator) -> tuple[ParetoArchive, list[dict]]:
    """Train a policy from scratch against ``evaluator.u_target``; return the
    Pareto archive at the end of training and the per-epoch log."""
    vocab = evaluator.vocab
    n = cfg.num_qubits
    context = context_tokens(n, cfg.max_context_tokens)
    min_cnots, max_cnots = cnot_lower_bound(n), cnot_upper_bound(n)
    N, B = cfg.group_size, cfg.batch_size

    torch.manual_seed(cfg.training_seed)                         # policy initialisation
    model = GatePolicy(cfg, vocab.size, context).to(DEVICE)
    optimizer = grpo.make_optimizer(model, cfg)
    generator = torch.Generator(device=DEVICE)                   # rollout sampling
    generator.manual_seed(cfg.training_seed)
    batch_rng = np.random.default_rng(cfg.training_seed)         # mini-batch order
    tables = token_tables(vocab)
    fit_angles = CoordinateSweep(evaluator, cfg.inner_sweeps)
    archive = ParetoArchive(cfg.max_archive_size, cfg.fidelity_floor, cfg.fidelity_tol)

    print(f"n={n}: |V|={vocab.size}, context={context} tokens, CNOT band [{min_cnots}, {max_cnots}], device={DEVICE}")
    log: list[dict] = []
    best_seen = -1.0
    t_run = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        t_epoch = time.perf_counter()
        beta = inverse_temperature(epoch, cfg)

        # Propose a group of N skeletons (Algorithm 3).
        out = sample(model, cfg, tables, context, min_cnots, max_cnots, beta, generator)
        tokens = out["tokens"].cpu().numpy().astype(np.int32)
        init_angles = out["angles"].cpu().numpy().astype(np.float32)
        log_p = out["log_p"].cpu().numpy().astype(np.float32)
        width = int(out["lengths"].max())                      # longest circuit of the group

        # Fit the angles (Algorithm 4), canonicalise (Algorithm 5), score (cost.py).
        fids, swept = fit_angles.refine(tokens[:, 1:width + 1], init_angles[:, :width])
        angles = init_angles.copy()
        angles[:, :width] = swept
        rows = [simplify_sequence(tokens[i], angles[i], vocab, cfg) for i in range(N)]
        simp_tokens = np.stack([r[0] for r in rows]).astype(np.int32)
        simp_angles = np.stack([r[1] for r in rows]).astype(np.float32)
        depths = np.array([r[2] for r in rows], dtype=np.int32)
        totals = np.array([r[3] for r in rows], dtype=np.int32)
        cnots = np.array([r[4] for r in rows], dtype=np.int32)
        costs = cost(fids, cnots, cfg)
        adv = grpo.advantages(costs)

        # Retain the non-dominated circuits (archive.py).
        archive.add([
            Circuit(float(fids[i]), int(cnots[i]), int(depths[i]), int(totals[i]),
                    simp_tokens[i].copy(), simp_angles[i].copy(), epoch, rows[i][5])
            for i in np.flatnonzero(fids >= cfg.fidelity_floor).tolist()
        ])

        # GRPO step: inner_passes passes over the group in mini-batches of B.
        order = batch_rng.permutation(np.arange(N * cfg.inner_passes, dtype=np.int32))
        losses = []
        for b in range(N * cfg.inner_passes // B):
            sel = order[b * B:(b + 1) * B] % N
            losses.append(grpo.update(
                model, optimizer,
                torch.as_tensor(tokens[sel, :width + 1], dtype=torch.long, device=DEVICE),
                torch.as_tensor(adv[sel], dtype=torch.float32, device=DEVICE),
                beta,
                torch.as_tensor(log_p[sel, :width], dtype=torch.float32, device=DEVICE),
                tables, min_cnots, max_cnots, cfg,
            ))

        # Log.
        i_best = int(np.argmax(fids))
        best_seen = max(best_seen, float(fids[i_best]))
        lengths = sequence_lengths(torch.as_tensor(simp_tokens[:, 1:])).numpy()
        top = archive.best_by_fidelity()
        compact = archive.best_by_cnot(cfg.fidelity_threshold)
        entry = {
            "epoch": epoch,
            "beta": float(beta),
            "loss": float(np.mean(losses)),
            "mean_cost": float(costs.mean()),
            "min_cost": float(costs.min()),
            "best_fidelity": float(fids[i_best]),
            "cnot_at_best_fidelity": int(cnots[i_best]),
            "depth_at_best_fidelity": int(depths[i_best]),
            "best_seen_fidelity": best_seen,
            "mean_length": float(lengths.mean()),
            "archive_size": len(archive),
            "archive_best_fidelity": None if top is None else float(top.fidelity),
            "archive_cnot_at_best_fidelity": None if top is None else int(top.cnot_count),
            "archive_min_cnot_at_threshold": None if compact is None else int(compact.cnot_count),
            "epoch_time_sec": time.perf_counter() - t_epoch,
            "elapsed_sec": time.perf_counter() - t_run,
        }
        log.append(entry)
        print(
            f"epoch {epoch + 1:3d}/{cfg.max_epochs} | beta={beta:.3f} | loss={entry['loss']:.4f}"
            f" | cost={entry['mean_cost']:.3f} | F_best={entry['best_fidelity']:.4f}"
            f" (C={entry['cnot_at_best_fidelity']}) | F_seen={best_seen:.4f}"
            f" | archive={len(archive)} | {entry['epoch_time_sec']:.1f}s"
        )
    return archive, log
