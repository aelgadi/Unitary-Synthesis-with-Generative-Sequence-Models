"""Cost (reward) — the cost formula of the README.

    cost(F, C) = max(C - L_SMB, 0) / dL + sqrt(max(1 - F, 0)) / sqrt(1 - F_target),
    dL = L_QSD - L_SMB + 1,

and the reward is R = -cost. The structure-quality term charges one unit for
traversing the feasible band [L_SMB, L_QSD], so the two terms are on the same
scale at every n; the fidelity term is one unit at F = F_target and its slope
diverges as F -> 1, sharpening the approach to the per-skeleton ceiling.
"""

import numpy as np

from helpers import cnot_lower_bound, cnot_upper_bound


def cost(fidelities: np.ndarray, cnot_counts: np.ndarray, cfg) -> np.ndarray:
    n = cfg.num_qubits
    F = np.asarray(fidelities).astype(np.float32)
    C = np.asarray(cnot_counts).astype(np.float32)
    l_smb = np.float32(cnot_lower_bound(n))
    delta_l = np.float32(max(cnot_upper_bound(n) - cnot_lower_bound(n) + 1, 1))
    infidelity_scale = np.float32(np.sqrt(max(1.0 - float(cfg.fidelity_target), 1e-12)))
    cnot_term = np.maximum(C - l_smb, np.float32(0.0)) / delta_l
    fidelity_term = np.sqrt(np.maximum(np.float32(1.0) - F, np.float32(0.0))) / infidelity_scale
    return (cnot_term + fidelity_term).astype(np.float32)
