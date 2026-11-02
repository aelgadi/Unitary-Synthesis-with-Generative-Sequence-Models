"""Configuration — every hyperparameter of the method (the table in the README).

Every number the method depends on is a field of :class:`Config`, and the
defaults are the benchmark protocol. ``compile.py`` overrides the target (qubit
count and seed), the epoch budget and the output directory from the command
line; nothing else varies between runs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Target: a Haar-random unitary on ``num_qubits`` qubits drawn
    # from ``target_seed``. The training seed is held fixed across targets, so
    # variation between runs reflects the target alone.
    num_qubits: int = 2
    target_seed: int = 1
    training_seed: int = 42

    # Policy network: decoder-only transformer, the "small" size.
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384               # feed-forward width is 4 * n_embd = 1536
    max_context_tokens: int = 2048  # cap on the token window W (see helpers.context_tokens)
    init_stddev: float = 0.02       # every weight matrix ~ N(0, init_stddev^2)
    layer_norm_eps: float = 1.0e-5

    # GRPO training.
    max_epochs: int = 50            # training iterations T
    group_size: int = 256           # rollouts per iteration N
    batch_size: int = 16            # mini-batch size B
    inner_passes: int = 4           # passes over each group: 4N/B = 64 updates per epoch
    lr: float = 3.0e-4              # AdamW learning rate
    weight_decay: float = 0.01
    grad_norm_clip: float = 1.0
    clip_ratio: float = 0.2         # epsilon_clip of the clipped surrogate

    # Sampling inverse temperature beta: cyclic cosine between beta_min and
    # beta_max with a period of half the run, i.e. two full cycles.
    beta_min: float = 0.6
    beta_max: float = 1.4

    # Angle fitting inside every rollout (Algorithm 4): passes K.
    inner_sweeps: int = 20

    # Reward and Pareto archive.
    fidelity_target: float = 0.999  # F_target
    fidelity_floor: float = 0.7     # archive admission floor F_adm
    fidelity_tol: float = 1.0e-4    # Pareto fidelity tolerance epsilon_F
    max_archive_size: int = 200

    # Post-training refinement.
    post_sweeps: int = 100          # coordinate-sweep passes over every archived circuit
    lbfgs_steps: int = 200          # L-BFGS iterations per archived circuit
    lbfgs_memory: int = 10          # curvature pairs kept per circuit
    lbfgs_line_search_steps: int = 20
    lbfgs_chunk: int = 64           # circuits refined at once (bounds the autograd memory)
    angle_tol: float = 1.0e-9       # Rule 3 of Algorithm 5: |theta| below this is removed

    # Reporting only. The per-epoch log records the most compact archived
    # circuit with F >= fidelity_threshold; the final report lists the most
    # compact circuit at each reporting threshold.
    fidelity_threshold: float = 0.99
    report_thresholds: tuple = (1.0 - 1e-9, 0.9999, 0.999)
