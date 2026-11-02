"""Target U — the starting point of the pipeline.

A Haar-random unitary on ``num_qubits`` qubits drawn from a seed: the QR decomposition of a complex Ginibre matrix with the phases
of the diagonal of R absorbed into Q, which makes Q Haar-distributed. The
target enters the method only through the fidelity F(V; U).
"""

import numpy as np


def haar_random_unitary(num_qubits: int, seed: int) -> np.ndarray:
    d = 2 ** int(num_qubits)
    rng = np.random.default_rng(int(seed))
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    phase = np.diag(r) / np.abs(np.diag(r))
    u = q * phase[np.newaxis, :]
    deviation = float(np.max(np.abs(u @ u.conj().T - np.eye(d))))
    if deviation > 1e-8:
        raise ValueError(f"target seed={seed} is not unitary (max deviation {deviation:.2e})")
    return u
