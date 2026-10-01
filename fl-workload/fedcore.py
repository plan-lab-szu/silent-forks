"""Shared mechanics for every arm of the comparison.

Extracted so that the integer arm, the float arm and the centralized arm differ
ONLY in the step under study. If each arm carried its own copy of the training
loop, a difference in results could always be blamed on the loop, and the
comparison would prove nothing.

What each arm shares: data split, model init, client-side local training, the
per-round seeding, and evaluation. What differs: how the updates are combined.
"""

import numpy as np

import data
import models
from client import Client


def make_split(source, data_root, clients, alpha, seed):
    Xtr, ytr, Xte, yte = data.load(source, data_root, seed=seed)
    shards = data.dirichlet_split(ytr, clients, alpha, seed=seed)
    return Xtr, ytr, Xte, yte, shards


def make_clients(Xtr, ytr, shards, kind, hidden, seed, lr, batch, local_epochs):
    return [
        Client(cid, Xtr[s], ytr[s], models.build(kind, hidden=hidden, seed=seed),
               lr, batch, local_epochs)
        for cid, s in enumerate(shards)]


def client_seed(seed, rnd, cid):
    """Per (run, round, client) seed.

    Fixed by construction rather than drawn from a shared generator, so an arm
    that performs fewer aggregation steps still gives its clients exactly the
    same local training as the other arms in the same round.
    """
    return seed * 1_000_003 + rnd * 1009 + cid


def local_updates(clients, flat, seed, rnd):
    updates, losses = [], []
    for c in clients:
        v, loss = c.train(flat, seed=client_seed(seed, rnd, c.cid))
        updates.append(v)
        losses.append(loss)
    weights = [c.n_samples for c in clients]
    return updates, weights, losses


def evaluate(model, flat, Xte, yte):
    model.load_flat(flat)
    return {"test_acc": model.accuracy(Xte, yte), "test_loss": model.loss(Xte, yte)}


def float64_reference(updates, weights):
    """The weighted mean computed in float64, used only as a measuring stick for
    quantization error. Not an arm: no validator would run this, and it is not
    reproducible across libraries either.
    """
    return np.average(np.asarray(updates, dtype=np.float64), axis=0,
                      weights=np.asarray(weights, dtype=np.float64))


def convergence_round(rows, threshold, key="test_acc"):
    """First round whose metric reaches the threshold, else None.

    R1-5b's complaint about "roughly a 45% reduction in convergence time" was
    that convergence was read off a curve by eye. So the criterion is explicit
    and applied identically to every arm.
    """
    for r in rows:
        if r.get(key) is not None and r[key] >= threshold:
            return r["round"]
    return None
