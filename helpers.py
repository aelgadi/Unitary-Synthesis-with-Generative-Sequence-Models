"""Shared helpers.

The device and precision defaults, the CNOT bounds L_SMB and L_QSD, the token
vocabulary, the circuit evaluator that turns a token row and its angles into a
unitary and scores it with the process fidelity, and the small utilities that
more than one stage of the pipeline relies on.
"""

from __future__ import annotations

import json
import math
from itertools import permutations

import numpy as np
import torch

# Single precision on the GPU (the benchmark setting), double on the CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_DTYPE = torch.float32 if DEVICE.type == "cuda" else torch.float64
COMPLEX_DTYPE = torch.complex64 if DEVICE.type == "cuda" else torch.complex128


# ---------------------------------------------------------------------------
# Resource bounds on the CNOT count.
# ---------------------------------------------------------------------------

def cnot_lower_bound(num_qubits: int) -> int:
    """L_SMB(n), the Shende–Markov–Bullock lower bound ceil((4^n - 3n - 1) / 4)."""
    n = int(num_qubits)
    if n <= 1:
        return 0
    return max(0, math.ceil((4 ** n - 3 * n - 1) / 4))


def cnot_upper_bound(num_qubits: int) -> int:
    """L_QSD(n), the CNOT count of the Quantum Shannon Decomposition,
    ceil((23/48) 4^n - (3/2) 2^n + 4/3)."""
    n = int(num_qubits)
    if n <= 1:
        return 0
    val = (23.0 / 48.0) * (4 ** n) - (3.0 / 2.0) * (2 ** n) + (4.0 / 3.0)
    return max(0, math.ceil(val))


# ---------------------------------------------------------------------------
# Token vocabulary.
# ---------------------------------------------------------------------------
BOS, STOP, PAD, GATE_OFFSET = 0, 1, 2, 3   # control tokens, then the gate tokens
RY, RZ, CNOT, NOOP = 0, 1, 2, 3            # gate types; RY/RZ index the Pauli stack [Y, Z]


def parse_token(name: str) -> tuple[str, tuple[int, ...]]:
    """``"RZ_q0"`` -> ``("RZ", (0,))``, ``"CNOT_q0_q1"`` -> ``("CNOT", (0, 1))``."""
    if name.startswith("<"):
        return "NOOP", ()
    parts = name.split("_")
    return parts[0], tuple(int(p[1:]) for p in parts[1:])


class Vocabulary:
    """The |V|(n) = n^2 + n + 3 tokens of the vocabulary, with per-token metadata.

    Ids: BOS = 0, STOP = 1, PAD = 2, then R_z and R_y on every qubit, then one
    CNOT token per ordered (control, target) pair. Qubit 0 is the leftmost
    tensor factor.
    """

    def __init__(self, num_qubits: int):
        n = int(num_qubits)
        self.num_qubits = n
        self.names = ["<BOS>", "<STOP>", "<PAD>"]
        for q in range(n):
            for axis in ("RZ", "RY"):
                self.names.append(f"{axis}_q{q}")
        for c, t in permutations(range(n), 2):
            self.names.append(f"CNOT_q{c}_q{t}")
        self.size = len(self.names)
        self.id = {name: i for i, name in enumerate(self.names)}
        self.cnot_pairs = list(permutations(range(n), 2))

        self.gate_type = np.full((self.size,), NOOP, dtype=np.int32)
        self.qubit0 = np.zeros((self.size,), dtype=np.int32)     # qubit acted on (control for CNOT)
        self.qubit1 = np.zeros((self.size,), dtype=np.int32)     # CNOT target, 0 otherwise
        self.cnot_pair = np.zeros((self.size,), dtype=np.int32)  # index into cnot_pairs
        for tok, name in enumerate(self.names):
            kind, qubits = parse_token(name)
            if kind == "NOOP":
                continue
            self.gate_type[tok] = {"RY": RY, "RZ": RZ, "CNOT": CNOT}[kind]
            self.qubit0[tok] = qubits[0]
            if kind == "CNOT":
                self.qubit1[tok] = qubits[1]
                self.cnot_pair[tok] = self.cnot_pairs.index(qubits)
        self.is_rotation = self.gate_type < CNOT
        self.is_cnot = self.gate_type == CNOT
        self.is_noop = self.gate_type == NOOP


def sequence_lengths(tokens: torch.Tensor, stop_id: int = STOP) -> torch.Tensor:
    """Active length of each token row: position of the first STOP plus one,
    or the full length when the row has no STOP."""
    b, n = tokens.shape
    padded = torch.cat(
        [tokens == stop_id, torch.ones((b, 1), dtype=torch.bool, device=tokens.device)], dim=1
    )
    first_stop = padded.to(torch.int32).argmax(dim=1)
    return torch.clamp(first_stop + 1, max=n)


def context_tokens(num_qubits: int, max_tokens: int) -> int:
    """Token window of a rollout: 7.5 L_SMB(n) action positions, at least 32
    and at most ``max_tokens - 1``, plus the BOS position."""
    budget = max(32, min(int(max_tokens) - 1, int(round(7.5 * cnot_lower_bound(num_qubits)))))
    return budget + 1


def pad_batch(circuits) -> tuple[np.ndarray, np.ndarray]:
    """Stack archived circuits into a (tokens without BOS, angles) batch of one
    common width, PAD-padding the tokens and zero-padding the angles."""
    width = max(max(np.asarray(c.tokens).size - 1, np.asarray(c.angles).size) for c in circuits)
    tokens = np.full((len(circuits), width), PAD, dtype=np.int32)
    angles = np.zeros((len(circuits), width), dtype=np.float32)
    for i, c in enumerate(circuits):
        row = np.asarray(c.tokens, dtype=np.int32)[1:]
        tokens[i, :row.size] = row
        a = np.asarray(c.angles, dtype=np.float32)
        angles[i, :a.size] = a
    return tokens, angles


# ---------------------------------------------------------------------------
# Circuit evaluator: token row -> unitary -> process fidelity |Tr(U^dag V)|^2 / d^2.
# ---------------------------------------------------------------------------
PAULI_X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PAULI_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def embed_single_qubit(gate_2x2: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    """Embed a one-qubit gate at ``qubit`` (qubit 0 is the leftmost tensor factor)."""
    eye = np.eye(2, dtype=gate_2x2.dtype)
    factors = [gate_2x2 if q == qubit else eye for q in range(num_qubits)]
    out = factors[0]
    for f in factors[1:]:
        out = np.kron(out, f)
    return out


def embed_cnot(control: int, target: int, num_qubits: int) -> np.ndarray:
    """CNOT on an ordered (control, target) pair: |0><0|_c (x) I + |1><1|_c (x) X_t."""
    p0 = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p1 = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    return (embed_single_qubit(p0, control, num_qubits)
            + embed_single_qubit(p1, control, num_qubits) @ embed_single_qubit(PAULI_X, target, num_qubits))


class CircuitEvaluator:
    """Unitary construction and process fidelity for batches of token rows.

    Every gate is applied as a full d x d matrix product on the batch, with the
    rotation generators and CNOTs pre-embedded per qubit and qubit pair. The
    same gate builder serves the coordinate sweep and, under autograd, the
    L-BFGS stage.
    """

    def __init__(self, u_target: np.ndarray, vocab: Vocabulary):
        n = vocab.num_qubits
        self.num_qubits = n
        self.dim = 2 ** n
        self.vocab = vocab
        self.u_target = torch.as_tensor(np.asarray(u_target), dtype=COMPLEX_DTYPE, device=DEVICE)
        self.u_target_dag = self.u_target.conj().transpose(0, 1).contiguous()
        self.eye = torch.eye(self.dim, dtype=COMPLEX_DTYPE, device=DEVICE)
        # Full embeddings, indexed [gate type (RY=0 -> Y, RZ=1 -> Z), qubit] and [CNOT pair].
        self.pauli_full = torch.as_tensor(
            np.stack([
                np.stack([embed_single_qubit(P, q, n) for q in range(n)])
                for P in (PAULI_Y, PAULI_Z)
            ]),
            dtype=COMPLEX_DTYPE, device=DEVICE,
        )
        self.cnot_full = torch.as_tensor(
            np.stack([embed_cnot(c, t, n) for (c, t) in vocab.cnot_pairs]),
            dtype=COMPLEX_DTYPE, device=DEVICE,
        )
        self.tok_gate_type = torch.as_tensor(vocab.gate_type, dtype=torch.long, device=DEVICE)
        self.tok_qubit0 = torch.as_tensor(vocab.qubit0, dtype=torch.long, device=DEVICE)
        self.tok_cnot_pair = torch.as_tensor(vocab.cnot_pair, dtype=torch.long, device=DEVICE)

    def gate(self, tokens: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Full unitary of every token in ``tokens`` (shape ``(B,)``) at ``angles``:
        cos(theta/2) I - i sin(theta/2) P for a rotation, the embedded CNOT for
        a CNOT token, the identity for BOS/STOP/PAD."""
        gtype = self.tok_gate_type[tokens]
        is_rot = gtype < CNOT
        P = self.pauli_full[torch.where(is_rot, gtype, 0), self.tok_qubit0[tokens]]
        half = angles.to(REAL_DTYPE) * 0.5
        c = torch.cos(half).to(COMPLEX_DTYPE)[:, None, None]
        s = torch.sin(half).to(COMPLEX_DTYPE)[:, None, None]
        rot = c * self.eye - 1j * s * P
        cnot = self.cnot_full[self.tok_cnot_pair[tokens]]
        return torch.where(
            is_rot[:, None, None], rot,
            torch.where((gtype == CNOT)[:, None, None], cnot, self.eye),
        )

    def unitary(self, tokens: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Compose token rows ``(B, L)`` (no BOS) with angles ``(B, L)`` into
        ``(B, d, d)`` unitaries, gate by gate in execution order."""
        U = self.eye.expand(tokens.shape[0], self.dim, self.dim)
        for t in range(tokens.shape[1]):
            U = self.gate(tokens[:, t], angles[:, t]) @ U
        return U

    def fidelity(self, U: torch.Tensor) -> torch.Tensor:
        """Process fidelity |Tr(U_target^dagger U)|^2 / d^2 of every unitary in ``U``."""
        overlap = (self.u_target.conj() * U).sum(dim=(-2, -1))
        return torch.clamp(overlap.abs() ** 2 / self.dim ** 2, 0.0, 1.0)

    @torch.no_grad()
    def fidelity_batch(self, tokens: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Fidelity of every row of a token/angle batch, as host float32."""
        if len(tokens) == 0:
            return np.zeros((0,), dtype=np.float32)
        tok = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=DEVICE)
        ang = torch.as_tensor(np.asarray(angles), dtype=REAL_DTYPE, device=DEVICE)
        return self.fidelity(self.unitary(tok, ang)).to(torch.float32).cpu().numpy()


# ---------------------------------------------------------------------------
# Artifacts.
# ---------------------------------------------------------------------------

def write_json(path, payload) -> None:
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=default)
