"""Client-side corruption models for the Phase 4 / U3 robustness illustration.

Deliberately the simplest textbook attack (amplified sign-flip), not a novel
or adaptive one: the point of robustness_experiment.py is to show the
determinism methodology generalizes to a second aggregation rule commonly
used to bound outlier influence, not to claim Byzantine fault tolerance
against adaptive or colluding adversaries. DETERMINISM_SPEC.md's non-goals
list already excludes majority-collusion resistance; this experiment does not
attempt to change that scope.
"""

import numpy as np


def sign_flip(updates, adversarial_indices, scale=10.0):
    """Negate and amplify the local updates at the given client indices.

    A fixed, non-adaptive attack: adversarial clients always negate and scale
    their true local update by the same factor, every round, regardless of
    what honest clients are doing. This is the simplest way to make an
    update a clear outlier at every coordinate simultaneously.
    """
    out = []
    for i, u in enumerate(updates):
        if i in adversarial_indices:
            out.append((-scale * np.asarray(u, dtype=np.float32)).astype(np.float32))
        else:
            out.append(u)
    return out
