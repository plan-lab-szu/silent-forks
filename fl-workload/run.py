"""Federated round loop: the plain-FL workload driving dfrl-agg.

  python run.py --rounds 20 --clients 10 --model logreg --alpha 0.5

Every round: clients train locally in float32, quantize, submit; the integer
aggregate is computed by the SAME code the chain will run; the state root is
recorded. With --float-pipe the identical updates are also aggregated the way
the prior work did, so the two roots sit side by side per round.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data
import fedcore
import floatpipe
import metrics
import models
from aggbridge import Aggregator
from manifest import Manifest


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID concentration")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    p.add_argument("--hidden", type=int, default=314, help="MLP width; 314 gives ~250k params")
    p.add_argument("--target-params", type=int, default=0,
                   help="pick the MLP width that fits this |Theta| (for the X8 sweep)")
    p.add_argument("--f", type=int, default=Aggregator.quantize.__defaults__[0],
                   help="fixed-point fractional bits (protocol parameter)")
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--data", choices=["synthetic", "mnist"], default="synthetic")
    p.add_argument("--data-root", default=None)
    p.add_argument("--backend", choices=["go", "python", "both"], default="go")
    p.add_argument("--crosscheck-every", type=int, default=5,
                   help="verify the Go aggregate against the Python reference every N rounds")
    p.add_argument("--float-pipe", action="store_true",
                   help="also aggregate in float32 under every strategy (C1)")
    p.add_argument("--keep-submissions", action="store_true",
                   help="persist canonical per-client submissions (replay audit material)")
    p.add_argument("--timestamp", default=None, help="recorded in the manifest verbatim")
    p.add_argument("--runs-dir", default="runs")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.target_params:
        args.hidden = models.width_for_params(args.target_params)

    config = {k: v for k, v in vars(args).items() if k not in ("timestamp", "runs_dir")}
    man = Manifest(config, timestamp=args.timestamp, root=args.runs_dir)

    Xtr, ytr, Xte, yte, shards = fedcore.make_split(
        args.data, args.data_root, args.clients, args.alpha, args.seed)
    hist = data.label_distribution(ytr, shards)
    man.note("data_source", args.data)
    man.note("client_sizes", [len(s) for s in shards])
    man.note("heterogeneity", metrics.heterogeneity(hist))

    global_model = models.build(args.model, hidden=args.hidden, seed=args.seed)
    n_params = global_model.n_params
    dims = [n_params]
    man.note("n_params", n_params)

    clients = fedcore.make_clients(Xtr, ytr, shards, args.model, args.hidden,
                                   args.seed, args.lr, args.batch, args.local_epochs)

    # Weights must satisfy the spec's range (1 <= s_n < 2^19). A shard larger
    # than that is a configuration error, not something to silently truncate.
    for c in clients:
        if not 1 <= c.n_samples < (1 << 19):
            raise SystemExit("client %d weight %d outside the legal range" % (c.cid, c.n_samples))

    agg = Aggregator(backend=args.backend, crosscheck_every=args.crosscheck_every)
    man.note("aggregator_backend", agg.backend_detail)

    log = metrics.RoundLog(man.path("rounds.jsonl"))
    flat = global_model.flat()
    print("run %s | %s | %d params | %d clients | alpha=%g | backend=%s"
          % (man.id, args.model, n_params, args.clients, args.alpha, agg.backend_detail))

    int_roots, float_root_counts = [], []
    for rnd in range(args.rounds):
        updates, weights, losses = fedcore.local_updates(clients, flat, args.seed, rnd)

        quantized = [agg.quantize(v, args.f) for v in updates]
        if args.keep_submissions:
            agg.keep_submissions(man.path("submissions"), rnd, weights, quantized, dims, args.f)

        agg_data, root, detail = agg.fedavg(weights, quantized, dims, args.f, round_index=rnd)
        int_roots.append(root)
        new_flat = agg.dequantize(agg_data, args.f)

        # Reference float aggregate of the same updates, for the quantization
        # error figure (R2-9). This is measurement, not the consensus path.
        qerr = metrics.quant_error(fedcore.float64_reference(updates, weights), new_flat)

        row = {"round": rnd, "state_root": root, "backend": detail,
               "train_loss": float(np.mean(losses)),
               "quant_error": qerr}
        if args.float_pipe:
            rep = floatpipe.divergence_report(floatpipe.aggregate_all(updates, weights))
            row["float_distinct_roots"] = rep["distinct_roots"]
            row["float_per_strategy"] = rep["per_strategy"]
            float_root_counts.append(rep["distinct_roots"])

        flat = new_flat
        global_model.load_flat(flat)
        row["test_acc"] = global_model.accuracy(Xte, yte)
        row["test_loss"] = global_model.loss(Xte, yte)
        log.add(**row)

        extra = ""
        if args.float_pipe:
            extra = " | float roots=%d" % row["float_distinct_roots"]
            
        print("  round %3d  acc=%.4f  loss=%.4f  qerr(max)=%.2e  root=%s  [%s]%s"
              % (rnd, row["test_acc"], row["test_loss"], qerr["max"], root[:16], detail, extra))

    log.close()
    man.note("accuracy", metrics.summarize(log.rows, "test_acc"))
    man.note("int_roots_unique", len(set(int_roots)))
    if float_root_counts:
        man.note("float_distinct_roots_per_round",
                 {"min": min(float_root_counts), "max": max(float_root_counts)})
    out = man.save()
    print("\nwrote %s" % out)
    print("integer path: %d distinct roots over %d rounds (expected %d -- one per round, "
          "all reproducible)" % (len(set(int_roots)), args.rounds, args.rounds))
    if float_root_counts:
        print("float path:   %d-%d distinct roots WITHIN each single round"
              % (min(float_root_counts), max(float_root_counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
