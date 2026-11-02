"""GRPO update — GrpoUpdate of Algorithm 2 (README).

Group-relative advantages, the clipped surrogate on the length-normalised
sequence likelihood ratio, and the AdamW optimiser with gradient-norm
clipping, all as written out in the README. The policy is scored under
the same masked, inverse-temperature-beta distribution its rollouts were
sampled from, so the ratio is a valid importance weight.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from helpers import sequence_lengths
from policy import GatePolicy, TokenTables, mask_logits, replay_last_token


def advantages(costs: np.ndarray) -> np.ndarray:
    """Group-relative advantage of every rollout: (mean cost - cost) / (sample std + 1e-8)."""
    costs = np.asarray(costs, dtype=np.float32)
    mean = costs.mean(dtype=np.float32)
    std = costs.std(ddof=1, dtype=np.float32) if costs.size > 1 else np.float32(0.0)
    return ((mean - costs) / (std + np.float32(1e-8))).astype(np.float32)


def clipped_loss(new_log_p, old_log_p, adv, clip_ratio: float):
    """Clipped surrogate loss, averaged over the mini-batch."""
    ratio = torch.exp(new_log_p - old_log_p)
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    return -torch.mean(torch.minimum(ratio * adv, clipped * adv))


def make_optimizer(model: GatePolicy, cfg) -> torch.optim.Optimizer:
    """AdamW on the policy parameters; the gradient is clipped to
    ``cfg.grad_norm_clip`` in global norm before every step."""
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def update(model: GatePolicy, optimizer, tokens, adv, beta: float, old_log_p,
           tables: TokenTables, min_cnots: int, max_cnots: int, cfg) -> float:
    """One gradient step on a mini-batch of BOS-prefixed token rows ``(B, T+1)``
    with their advantages and the log-probabilities recorded at sampling time.
    Returns the loss."""
    actions = tokens[:, 1:]
    T = actions.shape[1]
    positions = torch.arange(T, device=tokens.device)
    valid = positions[None, :] < sequence_lengths(actions)[:, None]
    last_token = replay_last_token(actions, tables)
    is_cnot = tables.is_cnot[actions].long()
    cnots_before = torch.cumsum(is_cnot, dim=1) - is_cnot

    logits = model(tokens)[:, :T]
    logits = mask_logits(logits, positions, last_token, cnots_before, tables, min_cnots, max_cnots)
    log_p = F.log_softmax(-beta * logits, dim=-1).gather(-1, actions[..., None]).squeeze(-1)
    denom = valid.sum(dim=-1).to(log_p.dtype).clamp(min=1.0)
    zero = torch.zeros((), dtype=log_p.dtype, device=log_p.device)
    seq_new = torch.where(valid, log_p, zero).sum(dim=-1) / denom
    seq_old = torch.where(valid, old_log_p, zero).sum(dim=-1) / denom
    loss = clipped_loss(seq_new, seq_old, adv.detach(), cfg.clip_ratio)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
    optimizer.step()
    return float(loss.item())
