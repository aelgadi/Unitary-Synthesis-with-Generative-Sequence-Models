"""Transformer policy pi_phi and Sample — Algorithm 3 of the README.

A decoder-only transformer maps the emitted token prefix to a
distribution over the next circuit token; it takes no representation of the
target. Tokens are drawn from softmax(-beta * logits) after the masks
listed in the README: BOS and PAD are never sampled, STOP is blocked at the first
position, the R1 (rotation-merge) and R2 (CNOT-involution) redundancy masks
block the token that would cancel against the last gate on a qubit, and the
CNOT-band masks keep every circuit inside [L_SMB, L_QSD]. Rotation angles are
not learned: each sampled rotation receives a uniform angle in [-pi, pi] that
Algorithm 4 then fits.

The same masked distribution is used by the sampler here and by the policy
scoring inside the GRPO update, so the importance ratio of the GRPO update is
well defined.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from helpers import BOS, DEVICE, PAD, STOP, Vocabulary, sequence_lengths


# ---------------------------------------------------------------------------
# The network.
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, cache=None, pos=None):
        B, T, C = x.shape
        head_dim = C // self.n_head
        q, k, v = self.c_attn(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        if cache is not None:                       # one new position; attend over the prefix
            k_cache, v_cache = cache
            k_cache[:, :, pos] = k[:, :, 0]
            v_cache[:, :, pos] = v[:, :, 0]
            k, v = k_cache[:, :, :pos + 1], v_cache[:, :, :pos + 1]
        scores = (q @ k.transpose(-2, -1)) * head_dim ** -0.5
        if cache is None:                           # full sequence: causal mask
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        out = F.softmax(scores, dim=-1) @ v
        return self.c_proj(out.transpose(1, 2).reshape(B, T, C))


class _Block(nn.Module):
    """Pre-norm decoder block: masked self-attention, then a GELU feed-forward."""

    def __init__(self, n_head: int, n_embd: int, eps: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=eps)
        self.attn = _Attention(n_head, n_embd)
        self.ln_2 = nn.LayerNorm(n_embd, eps=eps)
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x, cache=None, pos=None):
        x = x + self.attn(self.ln_1(x), cache, pos)
        return x + self.c_proj(F.gelu(self.c_fc(self.ln_2(x)), approximate="tanh"))


class GatePolicy(nn.Module):
    """Causal transformer producing gate-token logits; the output layer reuses
    the token-embedding weights."""

    def __init__(self, cfg, vocab_size: int, n_positions: int):
        super().__init__()
        self.n_head, self.n_embd, self.n_positions = cfg.n_head, cfg.n_embd, n_positions
        self.wte = nn.Embedding(vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(n_positions, cfg.n_embd)
        self.h = nn.ModuleList(
            [_Block(cfg.n_head, cfg.n_embd, cfg.layer_norm_eps) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=cfg.init_stddev)
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, input_ids):
        """Logits ``(B, T, V)`` for every position of the token rows ``(B, T)``."""
        T = input_ids.shape[1]
        if T > self.n_positions:
            raise ValueError(f"Sequence length {T} exceeds the context window {self.n_positions}")
        x = self.wte(input_ids) + self.wpe(torch.arange(T, device=input_ids.device))
        for block in self.h:
            x = block(x)
        return self.ln_f(x) @ self.wte.weight.T

    def new_cache(self, batch_size: int, total_len: int):
        shape = (batch_size, self.n_head, total_len, self.n_embd // self.n_head)
        return [
            (torch.zeros(shape, dtype=self.wte.weight.dtype, device=self.wte.weight.device),
             torch.zeros(shape, dtype=self.wte.weight.dtype, device=self.wte.weight.device))
            for _ in self.h
        ]

    def step(self, token, pos: int, cache):
        """Logits ``(B, V)`` for the next token after ``token`` at position ``pos``,
        using and extending the KV ``cache``."""
        x = (self.wte(token) + self.wpe.weight[pos])[:, None, :]
        for block, layer_cache in zip(self.h, cache):
            x = block(x, layer_cache, pos)
        return (self.ln_f(x) @ self.wte.weight.T)[:, 0]


# ---------------------------------------------------------------------------
# The sampling masks.
# ---------------------------------------------------------------------------

class TokenTables(NamedTuple):
    """Per-token metadata as device tensors, indexed by token id."""
    qubit0: torch.Tensor       # qubit acted on (control for CNOT)
    qubit1: torch.Tensor       # CNOT target (0 for other tokens; gated by is_cnot)
    is_rotation: torch.Tensor  # R1 candidates
    is_cnot: torch.Tensor      # R2 candidates
    is_noop: torch.Tensor      # BOS / STOP / PAD: never touch a qubit
    n_qubits: int
    size: int


def token_tables(vocab: Vocabulary) -> TokenTables:
    def t(a, dtype):
        return torch.as_tensor(a, dtype=dtype, device=DEVICE)

    return TokenTables(
        qubit0=t(vocab.qubit0, torch.long), qubit1=t(vocab.qubit1, torch.long),
        is_rotation=t(vocab.is_rotation, torch.bool), is_cnot=t(vocab.is_cnot, torch.bool),
        is_noop=t(vocab.is_noop, torch.bool), n_qubits=int(vocab.num_qubits), size=int(vocab.size),
    )


def initial_last_token(batch_size: int, n_qubits: int) -> torch.Tensor:
    """Per-qubit last-touch state before any gate: -1 on every qubit."""
    return torch.full((batch_size, n_qubits), -1, dtype=torch.long, device=DEVICE)


def update_last_token(last_token, emitted, tables: TokenTables):
    """Record ``emitted`` (shape ``(B,)``) as the last token on the qubits it
    touches; control tokens leave the state unchanged."""
    is_real = ~tables.is_noop[emitted]
    q0 = tables.qubit0[emitted]
    q1 = tables.qubit1[emitted]
    is_cnot = tables.is_cnot[emitted]
    q_range = torch.arange(tables.n_qubits, device=last_token.device)
    on_q0 = q_range[None, :] == q0[:, None]
    on_q1 = q_range[None, :] == q1[:, None]
    update = is_real[:, None] & (on_q0 | (is_cnot[:, None] & on_q1))
    return torch.where(update, emitted[:, None], last_token)


def replay_last_token(actions, tables: TokenTables):
    """Last-touch state in force when each action of a ``(B, T)`` batch was
    sampled, i.e. the state after the actions before it: shape ``(B, T, n)``."""
    B, T = actions.shape
    pos = torch.arange(T, device=actions.device)
    q0 = tables.qubit0[actions]
    q1 = tables.qubit1[actions]
    is_real = ~tables.is_noop[actions]
    is_cnot = tables.is_cnot[actions]
    state = torch.full((B, T, tables.n_qubits), -1, dtype=actions.dtype, device=actions.device)
    none = torch.full((B, 1), -1, dtype=torch.long, device=actions.device)
    for q in range(tables.n_qubits):
        touch = is_real & ((q0 == q) | (is_cnot & (q1 == q)))
        last = torch.cummax(torch.where(touch, pos, -1), dim=1).values   # last touch at or before t
        prev = torch.cat([none, last[:, :-1]], dim=1)                      # last touch strictly before t
        state[:, :, q] = torch.where(prev >= 0, actions.gather(1, prev.clamp(min=0)), -1)
    return state


def mask_logits(logits, position, last_token, cnot_counts, tables: TokenTables,
                min_cnots: int, max_cnots: int):
    """Apply every sampling mask to ``logits`` of shape ``(..., V)``.

    A masked token's logit is set to +inf, which is probability zero under
    softmax(-beta * logits). ``position`` is the sequence position of the
    leading axes (an int at rollout, ``arange(T)`` at the update),
    ``last_token`` the ``(..., n)`` last-touch state and ``cnot_counts`` the
    ``(...)`` CNOTs emitted before this position.
    """
    tok_ids = torch.arange(tables.size, device=logits.device)
    is_stop = tok_ids == STOP
    position = torch.as_tensor(position, device=logits.device)
    mask = (tok_ids == BOS) | (tok_ids == PAD)                                  # never sampled
    mask = mask | ((position == 0)[..., None] & is_stop)                        # no empty circuit
    match_q0 = last_token[..., tables.qubit0] == tok_ids                        # R1 / R2
    match_q1 = last_token[..., tables.qubit1] == tok_ids
    mask = mask | (match_q0 & tables.is_rotation) | (match_q0 & match_q1 & tables.is_cnot)
    mask = mask | ((cnot_counts >= max_cnots)[..., None] & tables.is_cnot)      # no CNOT past L_QSD
    mask = mask | ((cnot_counts < min_cnots)[..., None] & is_stop)              # no STOP below L_SMB
    return logits.masked_fill(mask, float("inf"))


# ---------------------------------------------------------------------------
# Sample — Algorithm 3.
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(model: GatePolicy, cfg, tables: TokenTables, total_len: int,
           min_cnots: int, max_cnots: int, beta: float, generator: torch.Generator) -> dict:
    """Draw ``cfg.group_size`` skeletons autoregressively with a KV cache.

    Returns BOS-prefixed token rows of length ``total_len`` (``"tokens"``), a
    proposal angle per action position (``"angles"``), the per-token
    log-probabilities under the masked policy (``"log_p"``) and each row's
    active length (``"lengths"``). Rows are PAD-filled after their STOP.
    """
    B = cfg.group_size
    n_steps = total_len - 1
    dtype = model.wte.weight.dtype
    cache = model.new_cache(B, total_len)
    prev = torch.full((B,), BOS, dtype=torch.long, device=DEVICE)
    stopped = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    toks = torch.full((B, n_steps), PAD, dtype=torch.long, device=DEVICE)
    angles = torch.zeros((B, n_steps), dtype=dtype, device=DEVICE)
    log_p = torch.zeros((B, n_steps), dtype=dtype, device=DEVICE)
    cnots = torch.zeros(B, dtype=torch.long, device=DEVICE)
    last_tok = initial_last_token(B, tables.n_qubits)

    for step in range(n_steps):
        logits = model.step(prev, step, cache)
        logits = mask_logits(logits, step, last_tok, cnots, tables, min_cnots, max_cnots)
        scaled = -beta * logits
        sampled = torch.multinomial(F.softmax(scaled, dim=-1), 1, generator=generator).squeeze(1)
        tok_next = torch.where(stopped, PAD, sampled)
        log_p_tok = F.log_softmax(scaled, dim=-1).gather(1, sampled[:, None]).squeeze(1)
        log_p_tok = torch.where(stopped, 0.0, log_p_tok)
        angle = torch.rand(B, generator=generator, dtype=dtype, device=DEVICE) * (2 * math.pi) - math.pi
        angle = torch.where(stopped, 0.0, angle)

        toks[:, step] = tok_next
        angles[:, step] = angle
        log_p[:, step] = log_p_tok
        cnots = cnots + tables.is_cnot[tok_next].long()
        last_tok = update_last_token(last_tok, tok_next, tables)
        stopped = stopped | (tok_next == STOP)
        prev = tok_next
        if bool(stopped.all()):
            break

    tokens = torch.cat([torch.full((B, 1), BOS, dtype=torch.long, device=DEVICE), toks], dim=1)
    return {"tokens": tokens, "angles": angles, "log_p": log_p, "lengths": sequence_lengths(toks)}
