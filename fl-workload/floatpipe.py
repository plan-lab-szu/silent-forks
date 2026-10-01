"""The prior-work-style floating-point aggregation pipeline (C1 core).

OJ-COMS 2024 aggregated on chain by calling a floating-point ML backend from a
precompiled contract. This module is that design's aggregation step, run on the
same client updates as the integer path, so the two can be compared round by
round on the same workload -- which is the sharpest available form of C1:
the failure mode is demonstrated on the very design being characterized,
using its own task family.

Reuses the strategy set from dfrl-agg/pyref/fp_divergence.py rather than
redefining it, so the workload experiment and the standalone demonstration
cannot drift apart.

The supported claim is that reproducibility depends on summation strategy,
library version, and build configuration, none of which the consensus protocol
specifies. Some strategies agree, so this does not establish universal
floating-point divergence.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dfrl-agg")),
    "pyref"))

import fp_divergence as FP  # noqa: E402


def aggregate_all(updates, weights):
    """Aggregate one round under every float strategy.

    updates: list of float32 vectors, one per client
    weights: list of numbers (the same s_n the integer path uses)

    Returns {strategy: {"root": hex, "vec": float32 array}}.
    """
    theta = np.asarray(updates, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    out = {}
    for name, vec in FP.strategies(theta, w).items():
        v = np.asarray(vec, dtype=np.float32)
        out[name] = {"root": FP.root(v), "vec": v}
    return out


def divergence_report(results, base="seq-fwd"):
    """Per-round summary: distinct roots, and how far each strategy is from the
    reference one. Counting distinct roots is the consensus-relevant number --
    each distinct root is a fork.
    """
    ref = results[base]["vec"]
    roots = {}
    per_strategy = {}
    for name, r in results.items():
        roots.setdefault(r["root"], []).append(name)
        differs = int(np.count_nonzero(r["vec"] != ref))
        maxabs = float(np.max(np.abs(r["vec"].astype(np.float64) - ref.astype(np.float64)))) if differs else 0.0
        per_strategy[name] = {"root": r["root"], "differs": differs, "max_abs_diff": maxabs}
    return {
        "distinct_roots": len(roots),
        "n_strategies": len(results),
        "root_groups": {k: v for k, v in roots.items()},
        "per_strategy": per_strategy,
    }
