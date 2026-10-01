"""Federated arm with exact integer aggregation, computed off chain.

"Off chain" here means only "not inside a block execution": the aggregation is
performed by the same `dfrl-agg` code the precompile and the fused opcode will
call. That is the whole point -- when Phase 3 adds the on-chain arm, U6 (on-chain
equals off-chain) is a byte comparison between two callers of one implementation,
not a reconciliation between two implementations.

Every round records the state root, so this arm's output is directly comparable
to the float arm's.
"""

import fedcore
import metrics
import models
from aggbridge import Aggregator


def run(source="mnist", data_root=None, clients=10, alpha=0.5, seed=1,
        model="logreg", hidden=314, lr=0.1, batch=64, local_epochs=1,
        rounds=15, f=None, backend="go", crosscheck_every=0, **_ignored):
    Xtr, ytr, Xte, yte, shards = fedcore.make_split(source, data_root, clients, alpha, seed)
    net = models.build(model, hidden=hidden, seed=seed)
    dims = [net.n_params]
    cs = fedcore.make_clients(Xtr, ytr, shards, model, hidden, seed, lr, batch, local_epochs)
    agg = Aggregator(backend=backend, crosscheck_every=crosscheck_every)
    ff = agg.quantize.__defaults__[0] if f is None else f

    flat = net.flat()
    rows = []
    for rnd in range(rounds):
        updates, weights, losses = fedcore.local_updates(cs, flat, seed, rnd)
        quantized = [agg.quantize(v, ff) for v in updates]
        agg_data, root, detail = agg.fedavg(weights, quantized, dims, ff, round_index=rnd)
        flat = agg.dequantize(agg_data, ff)
        row = {"round": rnd, "arm": "fed-int", "state_root": root, "backend": detail,
               "train_loss": float(sum(losses) / len(losses)),
               "quant_error": metrics.quant_error(
                   fedcore.float64_reference(updates, weights), flat)}
        row.update(fedcore.evaluate(net, flat, Xte, yte))
        rows.append(row)
    return rows
