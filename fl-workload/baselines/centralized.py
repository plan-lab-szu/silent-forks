"""Centralized arm: pool all data, train on one machine.

This is the reference for learning quality, and it exists to make the modest
claim rather than the flattering one. R2-14 asked how much of the reported gain
comes from federated aggregation versus the rest of the stack. The answer this
arm supports is: the federated integer arm should TRACK this curve, not beat it.
A blockchain that appeared to improve accuracy would be evidence of a bug.

Compute is matched deliberately. The federated arm does `local_epochs` passes
over each shard per round, and the shards partition the training set, so one
federated round costs `local_epochs` passes over the whole set. This arm does
exactly that per round, with the same learning rate and batch size.
"""

import numpy as np

import fedcore
import models


def run(source="mnist", data_root=None, clients=10, alpha=0.5, seed=1,
        model="logreg", hidden=314, lr=0.1, batch=64, local_epochs=1,
        rounds=15, **_ignored):
    Xtr, ytr, Xte, yte, _ = fedcore.make_split(source, data_root, clients, alpha, seed)
    net = models.build(model, hidden=hidden, seed=seed)
    lr32 = np.float32(lr)
    n = len(ytr)
    rows = []
    for rnd in range(rounds):
        rng = np.random.default_rng(fedcore.client_seed(seed, rnd, 0))
        last = float("nan")
        for _ in range(local_epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                last, grad = net.loss_and_grad(Xtr[idx], ytr[idx])
                net.load_flat(net.flat() - lr32 * grad)
        row = {"round": rnd, "arm": "centralized", "train_loss": float(last)}
        row.update(fedcore.evaluate(net, net.flat(), Xte, yte))
        rows.append(row)
    return rows
