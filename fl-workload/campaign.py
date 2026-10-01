"""Multi-seed campaign across the comparison arms, with CIs and paired tests.

  python campaign.py --seeds 10 --rounds 12 --data mnist

R1-5b's complaint was not that the numbers were wrong but that they were
unsupported: single runs, convergence read off a curve by eye, no confidence
intervals, no significance testing. So this driver exists before any figure does.

Two comparisons, asking different questions. Keeping them apart matters, because
attributing the wrong gap to the blockchain is exactly the kind of overclaim the
reviewers penalised:

  fed-int vs centralized -- this measures the cost of FEDERATION under non-IID
      data. It is not the cost of the chain, and must never be reported as such:
      any federated deployment pays it, blockchain or not.
  fed-int vs fed-float -- this isolates the aggregation arithmetic, which is the
      only thing that differs. The learning metrics are expected to be
      indistinguishable. That is the finding, not a null result: the fork is
      invisible in accuracy and visible only in the state root.

The chain's own cost is therefore neither of these. It is (a) gas, latency and
on-chain bytes, measured in Phase 3, plus (b) quantization error, measured here
and bounded by 2^-F. Answering R2-14 means giving that decomposition, not one
headline number.

Everything is written to runs/<id>/ and every figure must be built from that,
never from a value still in memory.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fedcore
from baselines import centralized, fedavg_offchain, priorwork_float
from manifest import Manifest

ARMS = {
    "centralized": centralized.run,
    "fed-int": fedavg_offchain.run,
    "fed-float": priorwork_float.run,
}


def mean_ci(values, level=0.95):
    """Mean with a t-based confidence interval. Returns (mean, half_width, n)."""
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    n = a.size
    if n == 0:
        return None, None, 0
    if n == 1:
        return float(a[0]), None, 1
    from scipy import stats
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * a.std(ddof=1) / np.sqrt(n))
    return float(a.mean()), half, n


def paired(a, b, level=0.95):
    """Paired comparison of two arms over the same seeds.

    Reports the mean difference with a CI -- the interpretable quantity -- plus
    both a parametric and a rank-based p-value, since n is small and normality
    is not established.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    d = x - y
    m, half, n = mean_ci(d, level)
    out = {"n": n, "mean_diff": m, "ci_half_width": half,
           "ci": [m - half, m + half] if half is not None else None}
    if n >= 2:
        from scipy import stats
        out["ttest_rel_p"] = float(stats.ttest_rel(x, y).pvalue)
        if np.any(d != 0):
            try:
                out["wilcoxon_p"] = float(stats.wilcoxon(x, y).pvalue)
            except ValueError as exc:
                out["wilcoxon_p"] = None
                out["wilcoxon_note"] = str(exc)
        else:
            out["wilcoxon_p"] = None
            out["wilcoxon_note"] = "all paired differences are exactly zero"
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=10, help="seeds 1..N")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    p.add_argument("--hidden", type=int, default=314)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--data", choices=["synthetic", "mnist"], default="mnist")
    p.add_argument("--data-root", default=None)
    p.add_argument("--threshold", type=float, default=0.88,
                   help="accuracy defining convergence; explicit, applied to every arm")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--backend", choices=["go", "python", "both"], default="go")
    p.add_argument("--timestamp", default=None)
    p.add_argument("--runs-dir", default="runs")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit("unknown arm %r" % a)

    config = {k: v for k, v in vars(args).items() if k not in ("timestamp", "runs_dir")}
    man = Manifest(config, timestamp=args.timestamp, root=args.runs_dir)
    print("campaign %s | %s | %d seeds x %d rounds | arms: %s"
          % (man.id, args.model, args.seeds, args.rounds, ", ".join(arms)))

    per_arm = {a: {"final_acc": [], "conv_round": [], "extra": []} for a in arms}
    with open(man.path("rounds.jsonl"), "w", encoding="utf-8") as raw:
        for seed in range(1, args.seeds + 1):
            line = "  seed %2d " % seed
            for arm in arms:
                rows = ARMS[arm](
                    source=args.data, data_root=args.data_root, clients=args.clients,
                    alpha=args.alpha, seed=seed, model=args.model, hidden=args.hidden,
                    lr=args.lr, batch=args.batch, local_epochs=args.local_epochs,
                    rounds=args.rounds, backend=args.backend)
                for r in rows:
                    raw.write(json.dumps({"seed": seed, **r}, sort_keys=True) + "\n")
                raw.flush()
                per_arm[arm]["final_acc"].append(rows[-1]["test_acc"])
                conv = fedcore.convergence_round(rows, args.threshold)
                per_arm[arm]["conv_round"].append(conv)
                if arm == "fed-float":
                    per_arm[arm]["extra"].append({
                        "n_chains": rows[-1]["n_chains"],
                        "distinct_roots": rows[-1]["distinct_roots"],
                        "max_param_distance": rows[-1]["max_param_distance"],
                        "accuracy_spread": max(r["accuracy_spread"] for r in rows)})
                line += " %s=%.4f(conv=%s)" % (arm, rows[-1]["test_acc"], conv)
            print(line, flush=True)

    summary = {}
    for arm in arms:
        m, h, n = mean_ci(per_arm[arm]["final_acc"])
        cm, ch, cn = mean_ci([c for c in per_arm[arm]["conv_round"] if c is not None])
        summary[arm] = {
            "final_acc": {"mean": m, "ci95_half": h, "n": n},
            "conv_round": {"mean": cm, "ci95_half": ch, "n_converged": cn,
                           "n_total": len(per_arm[arm]["conv_round"])},
        }
        extra = per_arm[arm]["extra"]
        if extra:
            summary[arm]["float_detail"] = {
                "n_chains": sorted({e["n_chains"] for e in extra}),
                "distinct_roots_final": sorted({e["distinct_roots"] for e in extra}),
                "mean_max_param_distance_final": mean_ci(
                    [e["max_param_distance"] for e in extra])[0],
                "worst_accuracy_spread": max(e["accuracy_spread"] for e in extra),
            }

    comparisons = {}
    for x, y in (("fed-int", "centralized"), ("fed-int", "fed-float")):
        if x in per_arm and y in per_arm:
            comparisons["%s_vs_%s" % (x, y)] = paired(
                per_arm[x]["final_acc"], per_arm[y]["final_acc"])

    man.note("summary", summary)
    man.note("comparisons", comparisons)
    man.note("threshold", args.threshold)
    out = man.save()

    print("\n%-14s %-28s %s" % ("arm", "final acc (mean +/- 95% CI)",
                                "rounds to %.2f" % args.threshold))
    for arm in arms:
        s, c = summary[arm]["final_acc"], summary[arm]["conv_round"]
        ci = "n/a" if s["ci95_half"] is None else "%.4f +/- %.4f" % (s["mean"], s["ci95_half"])
        conv = "never" if c["n_converged"] == 0 else "%.1f (%d/%d seeds)" % (
            c["mean"], c["n_converged"], c["n_total"])
        print("%-14s %-28s %s" % (arm, ci, conv))

    for k, v in comparisons.items():
        ci = "n/a" if v["ci"] is None else "[%+.5f, %+.5f]" % tuple(v["ci"])
        print("\n%s: mean diff %+.5f, 95%% CI %s" % (k, v["mean_diff"], ci))
        print("    paired t p=%s   wilcoxon p=%s%s" % (
            _fmt(v.get("ttest_rel_p")), _fmt(v.get("wilcoxon_p")),
            "   (%s)" % v["wilcoxon_note"] if v.get("wilcoxon_note") else ""))

    fd = summary.get("fed-float", {}).get("float_detail")
    if fd:
        print("\nfed-float: %s chains -> %s distinct state roots"
              % (fd["n_chains"], fd["distinct_roots_final"]))
        print("    worst accuracy spread across chains: %.5f" % fd["worst_accuracy_spread"])
        print("    mean max pairwise parameter distance, final round: %.3e"
              % fd["mean_max_param_distance_final"])

    print("\nwrote %s" % out)
    return 0


def _fmt(p):
    return "n/a" if p is None else ("%.4g" % p)


if __name__ == "__main__":
    raise SystemExit(main())
