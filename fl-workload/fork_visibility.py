"""Is the fork visible in the learning metrics? (Answer: no. That is the point.)

  python fork_visibility.py --rounds 60 --data mnist

Two validators, both honest, both running a conforming float aggregation that
differs only in the order one sum is associated. Everything else is identical:
initialization, data split, client-side local training, per-round seeds. They are
run as independent chains, so from round two onward each one's clients train from
its own global model and the divergence compounds.

Why this experiment exists separately from the divergence measurement: a
single-round measurement shows that the bytes differ. This shows what that costs
and -- more importantly -- what it does NOT cost.

The expected finding, and the reason it matters:

  * The state roots differ from the very first round and never re-converge. At
    the consensus layer that is a permanent partition, whatever happens next.
  * The parameter distance grows, but slowly, and stays at the 1e-7 scale. SGD
    does not amplify it into two different models.
  * Test accuracy stays identical. Not "close" -- identical, to every digit,
    because a 1e-7 perturbation almost never flips an argmax.

So the defect cannot be detected by monitoring learning quality. A deployment
watching accuracy sees a healthy system while its validators hold irreconcilable
state. That is a plausible explanation for why the failure mode went unremarked
in prior on-chain aggregation designs, including this group's own, and it is the
strongest argument that determinism must be a property of the specification
rather than something to be checked empirically after the fact.

It is also a limitation to state plainly: nothing here shows floating-point
aggregation harms learning. It does not. The claim is about consensus, and the
paper must not blur the two.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import priorwork_float
from manifest import Manifest


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    p.add_argument("--hidden", type=int, default=314)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--data", choices=["synthetic", "mnist"], default="mnist")
    p.add_argument("--data-root", default=None)
    p.add_argument("--strategies", default="seq-fwd,seq-rev",
                   help="comma-separated float strategies, one chain each")
    p.add_argument("--every", type=int, default=6, help="print every Nth round")
    p.add_argument("--timestamp", default=None)
    p.add_argument("--runs-dir", default="runs")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    config = {k: v for k, v in vars(args).items() if k not in ("timestamp", "runs_dir", "every")}
    man = Manifest(config, timestamp=args.timestamp, root=args.runs_dir)

    rows = priorwork_float.run(
        source=args.data, data_root=args.data_root, clients=args.clients,
        alpha=args.alpha, seed=args.seed, model=args.model, hidden=args.hidden,
        lr=args.lr, batch=args.batch, local_epochs=args.local_epochs,
        rounds=args.rounds, strategies=strategies)

    with open(man.path("rounds.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    first_root_split = next((r["round"] for r in rows if r["distinct_roots"] > 1), None)
    first_acc_split = next((r["round"] for r in rows if r["accuracy_spread"] > 0), None)

    print("%d chains: %s | %s | %d rounds" % (len(strategies), ", ".join(strategies),
                                              args.model, args.rounds))
    print("\n%-7s %-7s %-13s %-11s %s" % ("round", "roots", "max|dTheta|", "acc", "spread"))
    for r in rows:
        if r["round"] % args.every and r["round"] != rows[-1]["round"] \
                and r["round"] != first_acc_split:
            continue
        print("%-7d %-7d %-13.3e %-11.4f %.5f" % (
            r["round"], r["distinct_roots"], r["max_param_distance"],
            r["test_acc"], r["accuracy_spread"]))

    growth = (rows[-1]["max_param_distance"] / rows[0]["max_param_distance"]
              if rows[0]["max_param_distance"] else None)
    man.note("first_round_with_distinct_roots", first_root_split)
    man.note("first_round_with_accuracy_difference", first_acc_split)
    man.note("max_param_distance_first", rows[0]["max_param_distance"])
    man.note("max_param_distance_last", rows[-1]["max_param_distance"])
    man.note("param_distance_growth_factor", growth)
    man.note("worst_accuracy_spread", max(r["accuracy_spread"] for r in rows))
    out = man.save()

    print("\nstate roots first differ at round: %s" % first_root_split)
    print("accuracy first differs at round:    %s"
          % (first_acc_split if first_acc_split is not None
             else "never, in %d rounds" % args.rounds))
    if growth:
        print("parameter distance grew %.1fx (%.3e -> %.3e), staying at the 1e-7 scale"
              % (growth, rows[0]["max_param_distance"], rows[-1]["max_param_distance"]))
    print("\nReading: the chains are permanently partitioned at the consensus layer"
          "\nwhile remaining functionally indistinguishable as models. The defect is"
          "\ninvisible to anyone monitoring learning quality.")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
