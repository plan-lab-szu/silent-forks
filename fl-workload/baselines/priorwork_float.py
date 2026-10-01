"""Float arm: what happens when validators aggregate in floating point.

This is not a divergence *measurement* on one round -- floatpipe.py already does
that. This simulates the consequence: several validators, each running an equally
conforming float aggregation, all fed identical client updates from identical
seeds, each maintaining its own global model.

The experimental control is total. Same initialization, same data split, same
client-side local training with the same per-(round, client) seeds. The ONLY
difference between chains is the order in which one sum is associated. If they
end up in different states, nothing but that can be responsible.

What this shows that a single-round measurement cannot: after the first round the
chains hold different bytes, so from round two the clients on different chains
are training from different global models, and the divergence compounds. A fork
is not a one-off 1e-08 discrepancy; it is a permanent split.

Reported per round: each chain's state root, how many distinct roots exist, the
largest pairwise parameter distance between chains, and each chain's accuracy.
"""

import itertools

import numpy as np

import fedcore
import floatpipe
import models


def run(source="mnist", data_root=None, clients=10, alpha=0.5, seed=1,
        model="logreg", hidden=314, lr=0.1, batch=64, local_epochs=1,
        rounds=15, strategies=None, **_ignored):
    Xtr, ytr, Xte, yte, shards = fedcore.make_split(source, data_root, clients, alpha, seed)
    net = models.build(model, hidden=hidden, seed=seed)
    cs = fedcore.make_clients(Xtr, ytr, shards, model, hidden, seed, lr, batch, local_epochs)

    init = net.flat()
    probe = floatpipe.aggregate_all([init, init], [1, 1])
    names = list(strategies or probe.keys())
    chains = {name: init.copy() for name in names}

    rows = []
    for rnd in range(rounds):
        roots, accs = {}, {}
        for name in names:
            updates, weights, _ = fedcore.local_updates(cs, chains[name], seed, rnd)
            res = floatpipe.aggregate_all(updates, weights)[name]
            chains[name] = res["vec"]
            roots[name] = res["root"]
            accs[name] = fedcore.evaluate(net, chains[name], Xte, yte)["test_acc"]

        pairs = list(itertools.combinations(names, 2))
        dist = max((float(np.max(np.abs(chains[a].astype(np.float64)
                                       - chains[b].astype(np.float64))))
                    for a, b in pairs), default=0.0)
        spread = max(accs.values()) - min(accs.values())
        rows.append({
            "round": rnd, "arm": "fed-float",
            "distinct_roots": len(set(roots.values())),
            "n_chains": len(names),
            "max_param_distance": dist,
            "accuracy_spread": spread,
            "roots": {k: v[:16] for k, v in roots.items()},
            "accuracies": accs,
            # The reference chain's accuracy, so this arm has one comparable
            # test_acc column like the others. Which chain is "the" chain is
            # itself undefined -- that is the defect, not a reporting choice.
            "test_acc": accs[names[0]],
        })
    return rows
