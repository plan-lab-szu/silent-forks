"""Phase 4 / U3: FedAvg vs TrimmedMean under a fixed adversarial attack, on
real MNIST training -- the missing "real workload" half of Phase 4 (the
algorithmic half, dfrl-agg's TestTrimmedMeanBoundsOutlierInfluence, was
already a unit-level illustration with synthetic tensors; this repeats the
comparison with real per-round local updates from actual training).

  python robustness_experiment.py

Two arms, same honest clients, same seed, same per-round local training --
the ONLY difference is which clients are adversarial-injected and how the
result is aggregated:

  fedavg-attacked    -- FedAvg over honest + adversarial updates every round
  trimmed-attacked   -- TrimmedMean(k) over the SAME honest + adversarial
                        updates every round

Each arm's own aggregated model feeds its own next round's local training,
so this is two independent, fully-evolving federated runs under identical
attack conditions -- not a single-round snapshot.

Scope discipline (see attacks.py's docstring): the attack is the simplest
textbook case (fixed-fraction, non-adaptive, amplified sign-flip), not a
claim of Byzantine fault tolerance in general. DETERMINISM_SPEC.md's
non-goals list already excludes majority-collusion resistance and that is
unchanged here -- this experiment shows the SAME determinism methodology
(dfrl-agg golden vectors, permutation invariance, cross-language agreement)
extends to trimmed mean, a rule harder to make deterministic than FedAvg
(needs a total order / tie-break), and illustrates why you would want it,
nothing stronger.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fedcore
import models
from aggbridge import Aggregator, BridgeError
from aggbridge import R as dfrl_ref
from attacks import sign_flip

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "robustness")


def run_arm(name, aggregate_fn, clients, rounds, adversarial, net0, Xte, yte, seed, agg, f):
    """Runs up to `rounds` rounds. If the attack drives a parameter out of the
    quantizer's representable range, that is not a bug to work around -- it
    IS a result (the aggregate diverged badly enough to break the fixed-point
    format itself) -- so it is caught, recorded, and the arm stops early
    rather than the whole experiment crashing.

    net0 must be a model instance PRIVATE to this arm: fedcore.evaluate()
    mutates its internal parameters as a side effect (model.load_flat), so
    two arms sharing one model object would silently corrupt each other's
    starting point across calls -- this is why main() builds a fresh net
    per arm rather than reusing one.
    """
    flat = net0.flat()
    rows = []
    for rnd in range(rounds):
        updates, weights, losses = fedcore.local_updates(clients, flat, seed, rnd)
        corrupted = sign_flip(updates, adversarial)
        try:
            agg_data, root, detail = aggregate_fn(corrupted, weights, rnd)
        except (dfrl_ref.SpecError, BridgeError) as exc:
            print("  %-18s round %2d: DIVERGED -- %s" % (name, rnd, exc))
            rows.append({"round": rnd, "arm": name, "diverged": True, "error": str(exc)})
            break
        flat = agg.dequantize(agg_data, f)
        row = fedcore.evaluate(net0, flat, Xte, yte)
        row.update({"round": rnd, "arm": name, "backend": detail, "root": root,
                     "train_loss": float(sum(losses) / len(losses)), "diverged": False})
        rows.append(row)
        print("  %-18s round %2d: acc=%.4f loss=%.4f" % (name, rnd, row["test_acc"], row["test_loss"]))
    return rows


def main():
    clients_n, rounds, alpha, seed = 10, 15, 0.5, 1
    model_kind, hidden, lr, batch, local_epochs = "logreg", 314, 0.1, 64, 1
    adversarial = {2, 7}  # fixed, 2/10 = 20% of clients
    k = 2                 # trims 2 smallest + 2 largest per coordinate

    os.makedirs(OUT_DIR, exist_ok=True)

    Xtr, ytr, Xte, yte, shards = fedcore.make_split("mnist", None, clients_n, alpha, seed)
    net_fedavg = models.build(model_kind, hidden=hidden, seed=seed)
    net_trimmed = models.build(model_kind, hidden=hidden, seed=seed)  # separate instance, same init
    dims = [net_fedavg.n_params]
    print("model=%s |Theta|=%d clients=%d rounds=%d adversarial=%s (k=%d)"
          % (model_kind, net_fedavg.n_params, clients_n, rounds, sorted(adversarial), k))
    cs = fedcore.make_clients(Xtr, ytr, shards, model_kind, hidden, seed, lr, batch, local_epochs)
    agg = Aggregator(backend="go")
    f = agg.quantize.__defaults__[0]

    def quantize_all(updates):
        return [agg.quantize(v, f) for v in updates]

    def fedavg_fn(corrupted, weights, rnd):
        q = quantize_all(corrupted)
        return agg.fedavg(weights, q, dims, f, round_index=rnd)

    def trimmed_fn(corrupted, weights, rnd):
        q = quantize_all(corrupted)
        return agg.trimmed_mean(q, dims, k, f, round_index=rnd)

    print("\n--- arm: fedavg-attacked ---")
    rows_fedavg = run_arm("fedavg-attacked", fedavg_fn, cs, rounds, adversarial, net_fedavg, Xte, yte, seed, agg, f)
    print("\n--- arm: trimmed-attacked ---")
    rows_trimmed = run_arm("trimmed-attacked", trimmed_fn, cs, rounds, adversarial, net_trimmed, Xte, yte, seed, agg, f)

    with open(os.path.join(OUT_DIR, "rounds.jsonl"), "w", encoding="utf-8") as fh:
        for row in rows_fedavg + rows_trimmed:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def arm_outcome(rows):
        if rows[-1].get("diverged"):
            return {"diverged": True, "diverged_at_round": rows[-1]["round"],
                     "error": rows[-1]["error"], "final_test_acc": None}
        return {"diverged": False, "final_test_acc": rows[-1]["test_acc"]}

    summary = {
        "config": {"clients": clients_n, "rounds": rounds, "adversarial": sorted(adversarial), "k": k,
                    "model": model_kind, "dims": dims},
        "fedavg-attacked": arm_outcome(rows_fedavg),
        "trimmed-attacked": arm_outcome(rows_trimmed),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== summary ===")
    for arm in ("fedavg-attacked", "trimmed-attacked"):
        o = summary[arm]
        if o["diverged"]:
            print("%-18s DIVERGED at round %d (%s)" % (arm, o["diverged_at_round"], o["error"]))
        else:
            print("%-18s final test_acc=%.4f" % (arm, o["final_test_acc"]))
    print("wrote %s" % OUT_DIR)


if __name__ == "__main__":
    main()
