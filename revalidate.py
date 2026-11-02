"""Re-validate the front — ParetoFront of Algorithm 1 (README).

The post-training stages act on each archived circuit independently, so the
dominance relations established during training no longer hold once angles
are lifted and gates removed. The front is recomputed from scratch from the
refined fidelities and simplified gate counts by inserting every circuit into
a fresh archive; it is applied after every post-training stage.
"""

from archive import Circuit, ParetoArchive


def pareto_front(circuits: list[Circuit], cfg) -> ParetoArchive:
    front = ParetoArchive(cfg.max_archive_size, cfg.fidelity_floor, cfg.fidelity_tol)
    front.add(circuits)
    return front
