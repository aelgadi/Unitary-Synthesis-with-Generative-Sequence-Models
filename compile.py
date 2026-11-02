"""Compile — Algorithm 1 of the README, the whole pipeline.

    python compile.py --n-qubits 3 --target-seed 46 [--max-epochs 3] [--output-dir runs]

Builds the target (target.py), trains a policy from scratch (train.py) and
then refines every archived circuit: its angles are re-swept (Algorithm 4),
its structure simplified (Algorithm 5), its angles lifted by L-BFGS
(Algorithm 6) and simplified once more, with the Pareto front re-validated
after every stage (ParetoFront). The final front and the training log are
written to ``<output-dir>/n<n>_seed<seed>/`` and the most compact circuit at
each reporting threshold is printed.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np

import lbfgs
import sweep
from archive import Circuit
from config import Config
from helpers import CircuitEvaluator, Vocabulary, cnot_lower_bound, cnot_upper_bound, write_json
from revalidate import pareto_front
from simplify import simplify_circuits
from target import haar_random_unitary
from train import train


def most_compact(circuits: list[Circuit], threshold: float) -> Circuit | None:
    """Lowest-CNOT circuit with F >= threshold (ties: depth, gates, fidelity)."""
    eligible = [c for c in circuits if c.fidelity >= threshold]
    if not eligible:
        return None
    return min(eligible, key=lambda c: (c.cnot_count, c.depth, c.total_gates, -c.fidelity))


def report(front: list[Circuit], cfg: Config, wall: float) -> None:
    n = cfg.num_qubits
    print(
        f"\nn={n} seed={cfg.target_seed}: {len(front)} circuits on the final Pareto front, "
        f"band [L_SMB, L_QSD] = [{cnot_lower_bound(n)}, {cnot_upper_bound(n)}], {wall:.0f}s"
    )
    print(f"{'operating point':<18}{'fidelity':>12}{'CNOT':>7}{'depth':>7}{'gates':>7}")
    rows = [("F = 1" if t >= 1 - 1e-8 else f"F >= {t:g}", most_compact(front, t)) for t in cfg.report_thresholds]
    rows.append(("best fidelity", front[0] if front else None))
    for label, c in rows:
        if c is None:
            print(f"{label:<18}{'-':>12}{'-':>7}{'-':>7}{'-':>7}")
        else:
            print(f"{label:<18}{c.fidelity:>12.6f}{c.cnot_count:>7}{c.depth:>7}{c.total_gates:>7}")


def compile_target(cfg: Config, output_dir: Path) -> list[Circuit]:
    """Run Algorithm 1 for one target and return its final Pareto front."""
    out = Path(output_dir) / f"n{cfg.num_qubits}_seed{cfg.target_seed}"
    out.mkdir(parents=True, exist_ok=True)
    u_target = haar_random_unitary(cfg.num_qubits, cfg.target_seed)
    np.save(out / "target_unitary.npy", u_target)
    vocab = Vocabulary(cfg.num_qubits)
    evaluator = CircuitEvaluator(u_target, vocab)

    t0 = time.perf_counter()
    archive, log = train(cfg, evaluator)                  # training phase (Algorithm 2)
    if len(archive):                                      # post-training phase
        print(f"Refining the {len(archive)} archived circuits ...")
        archive = pareto_front(sweep.refine_circuits(archive.circuits, evaluator, cfg.post_sweeps), cfg)
        archive = pareto_front(simplify_circuits(archive.sorted(), vocab, cfg), cfg)
        archive = pareto_front(lbfgs.refine_circuits(archive.circuits, evaluator, cfg), cfg)
        archive = pareto_front(simplify_circuits(archive.sorted(), vocab, cfg), cfg)
    wall = time.perf_counter() - t0

    front = archive.sorted()
    write_json(out / "run.json", {
        "config": dataclasses.asdict(cfg),
        "L_SMB": cnot_lower_bound(cfg.num_qubits),
        "L_QSD": cnot_upper_bound(cfg.num_qubits),
        "vocabulary": vocab.names,
        "wall_time_sec": wall,
        "training_log": log,
        "pareto_front": [dataclasses.asdict(c) for c in front],
    })
    report(front, cfg, wall)
    print(f"Artifacts: {out}")
    return front


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-qubits", type=int, default=Config.num_qubits)
    parser.add_argument("--target-seed", type=int, default=Config.target_seed)
    parser.add_argument("--max-epochs", type=int, default=Config.max_epochs)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)
    cfg = Config(num_qubits=args.n_qubits, target_seed=args.target_seed, max_epochs=args.max_epochs)
    compile_target(cfg, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
