"""Simplify — Algorithm 5 of the README.

Five algebraic rules, applied in a single left-to-right pass that is repeated
until no rule fires:

  1. merge consecutive same-axis rotations on a qubit, R_P(a) R_P(b) = R_P(a + b);
  2. cancel consecutive identical CNOTs, CNOT(c,t) CNOT(c,t) = I;
  3. remove rotations with |theta| < angle_tol;
  4. reduce every angle to (-pi, pi]: a 2 pi shift is a global phase;
  5. move an R_z rightward past the control of a CNOT, which creates new
     opportunities for rules 1-3.

"Consecutive" means adjacent on the qubits involved: the scan looks back past
gates on other qubits, and past CNOT controls for an R_z (rule 5). The rules
are exact identities, so the fidelity is unchanged.

The same routine canonicalises every sampled rollout during training, which
gives the archive its canonical hash and the reward its CNOT count.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np

from helpers import BOS, GATE_OFFSET, PAD, STOP, Vocabulary, parse_token

_TWO_PI = 2.0 * math.pi
_HASH_ANGLE_SCALE = 1_000_000


@dataclass(frozen=True)
class _Op:
    kind: str            # "RY" | "RZ" | "CNOT"
    qubits: tuple        # (q,) for a rotation, (control, target) for a CNOT
    angle: float = 0.0   # rotations only


def _reduce_angle(theta: float) -> float:
    """Rule 4: canonicalise to (-pi, pi]."""
    theta = math.fmod(theta, _TWO_PI)
    if theta > math.pi:
        theta -= _TWO_PI
    elif theta <= -math.pi:
        theta += _TWO_PI
    return theta


def _decode(tokens: np.ndarray, angles: np.ndarray, vocab: Vocabulary) -> list[_Op]:
    """Token row (BOS at position 0, angles in the same layout) -> gate list."""
    ops: list[_Op] = []
    for pos in range(1, len(tokens)):
        tok = int(tokens[pos])
        if tok == STOP:
            break
        if tok < GATE_OFFSET:
            continue
        kind, qubits = parse_token(vocab.names[tok])
        ops.append(_Op(kind, qubits, float(angles[pos]) if kind != "CNOT" else 0.0))
    return ops


def _encode(ops: list[_Op], vocab: Vocabulary, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Gate list -> BOS-prefixed token row of ``length`` and action-aligned angles."""
    toks: list[int] = [BOS]
    angs: list[float] = []
    for op in ops:
        if len(toks) >= length:
            break
        name = f"{op.kind}_q{op.qubits[0]}" if op.kind != "CNOT" else f"CNOT_q{op.qubits[0]}_q{op.qubits[1]}"
        toks.append(vocab.id[name])
        angs.append(op.angle)
    if len(toks) < length:
        toks.append(STOP)
        angs.append(0.0)
    while len(toks) < length:
        toks.append(PAD)
        angs.append(0.0)
    while len(angs) < length - 1:
        angs.append(0.0)
    return np.array(toks[:length], dtype=np.int32), np.array(angs[:max(length - 1, 0)], dtype=np.float64)


def _one_pass(ops: list[_Op], angle_tol: float) -> tuple[list[_Op], bool]:
    """One left-to-right scan applying rules 1, 2 and 5, then rules 3 and 4."""
    out: list[_Op] = []
    changed = False
    for op in ops:
        placed = False
        for j in range(len(out) - 1, -1, -1):
            prev = out[j]
            if not any(q in prev.qubits for q in op.qubits):
                continue                                             # does not touch these qubits
            if op.kind == "RZ" and prev.kind == "CNOT" and op.qubits == (prev.qubits[0],):
                continue                                             # rule 5: commute past the control
            if prev.kind == op.kind and prev.qubits == op.qubits:
                if op.kind == "CNOT":
                    out.pop(j)                                       # rule 2
                else:
                    out[j] = _Op(op.kind, op.qubits, prev.angle + op.angle)  # rule 1
                changed = True
                placed = True
            break                                                    # blocked, merged or cancelled
        if not placed:
            out.append(op)

    filtered: list[_Op] = []
    for op in out:
        if op.kind == "CNOT":
            filtered.append(op)
            continue
        r = _reduce_angle(op.angle)                                  # rule 4
        if abs(r) < angle_tol:                                       # rule 3
            changed = True
            continue
        if r != op.angle:
            changed = True
        filtered.append(_Op(op.kind, op.qubits, r))
    return filtered, changed


def _simplify_ops(ops: list[_Op], angle_tol: float) -> list[_Op]:
    for _ in range(200):
        ops, changed = _one_pass(ops, angle_tol)
        if not changed:
            break
    return ops


def _metrics(ops: list[_Op], num_qubits: int) -> tuple[int, int, int]:
    """(depth, total gates, CNOT count) of a gate list."""
    qubit_time = [0] * num_qubits
    for op in ops:
        layer = max(qubit_time[q] for q in op.qubits)
        for q in op.qubits:
            qubit_time[q] = layer + 1
    depth = max(qubit_time) if any(qubit_time) else 0
    return depth, len(ops), sum(1 for op in ops if op.kind == "CNOT")


def _canonical_hash(ops: list[_Op]) -> str:
    h = hashlib.sha256()
    h.update(b"gqe-simplified-v1")
    for op in ops:
        h.update(op.kind.encode("ascii"))
        h.update(b":")
        h.update(",".join(str(int(q)) for q in op.qubits).encode("ascii"))
        if op.kind != "CNOT":
            h.update(f":{int(round(_reduce_angle(op.angle) * _HASH_ANGLE_SCALE))}".encode("ascii"))
        h.update(b";")
    return h.hexdigest()


def simplify_sequence(tokens: np.ndarray, angles: np.ndarray | None, vocab: Vocabulary, cfg):
    """Simplify one BOS-prefixed token row.

    ``angles`` may be aligned with the full row or with the action positions
    (``tokens[1:]``). Returns ``(tokens, angles, depth, total_gates, cnot_count,
    canonical_hash)``; the returned angles are aligned with the action positions.
    """
    tokens = np.asarray(tokens, dtype=np.int32)
    aligned = np.zeros(len(tokens), dtype=np.float64)
    if angles is not None:
        raw = np.asarray(angles, dtype=np.float64)
        if raw.size == len(tokens):
            aligned = raw
        else:
            k = min(raw.size, max(len(tokens) - 1, 0))
            aligned[1:1 + k] = raw[:k]

    ops = _simplify_ops(_decode(tokens, aligned, vocab), cfg.angle_tol)
    depth, total, cnots = _metrics(ops, vocab.num_qubits)
    new_tokens, new_angles = _encode(ops, vocab, len(tokens))
    return new_tokens, new_angles, depth, total, cnots, _canonical_hash(ops)


def simplify_circuits(circuits, vocab: Vocabulary, cfg):
    """Simplify every archived circuit; fidelity is preserved, the gate counts
    are recomputed and the canonical hash set."""
    out = []
    for c in circuits:
        tokens, angles, depth, total, cnots, h = simplify_sequence(c.tokens, c.angles, vocab, cfg)
        out.append(replace(
            c, cnot_count=cnots, depth=depth, total_gates=total,
            tokens=tokens, angles=angles, canonical_hash=h,
        ))
    return out
