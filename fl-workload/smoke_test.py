"""End-to-end checks for the plain-FL workload. Run: python smoke_test.py

Kept as a script rather than a pytest suite because it is also the thing you run
to convince yourself the pipeline is intact after touching dfrl-agg. It asserts
properties, not numbers, so it does not need updating when the synthetic data or
the learning rate changes.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data
import floatpipe
import models
import run as runner
from aggbridge import Aggregator

FAIL = []


def check(name, cond, detail=""):
    print("  %-52s %s%s" % (name, "PASS" if cond else "FAIL", (" -- " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)


def roots_of(run_dir):
    with open(os.path.join(run_dir, "rounds.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line)["state_root"] for line in fh]


def main():
    tmp = tempfile.mkdtemp(prefix="fl-smoke-")
    try:
        print("1. models")
        check("logreg has 7,850 parameters", models.LogReg().n_params == 7850)
        check("MLP width 314 lands near 250k", 249_000 < models.MLP(314).n_params < 251_000,
              str(models.MLP(314).n_params))
        m = models.MLP(32)
        v = m.flat()
        m.load_flat(v)
        check("flat/load round trip is exact", np.array_equal(v, m.flat()))

        print("2. non-IID split")
        _, ytr, _, _ = data.load("synthetic", seed=0)
        skew = data.dirichlet_split(ytr, 10, 0.5, seed=1)
        iid = data.dirichlet_split(ytr, 10, 100.0, seed=1)
        import metrics
        tv_skew = metrics.heterogeneity(data.label_distribution(ytr, skew))["tv_mean"]
        tv_iid = metrics.heterogeneity(data.label_distribution(ytr, iid))["tv_mean"]
        check("small alpha is measurably more heterogeneous", tv_skew > 3 * tv_iid,
              "tv %.3f vs %.3f" % (tv_skew, tv_iid))
        check("every client gets at least one sample (s_n >= 1)",
              all(len(s) >= 1 for s in skew))

        print("3. aggregation bridge")
        agg = Aggregator(backend="both")
        check("Go toolchain available (else fell back)", agg.backend == "both",
              agg.backend_detail)
        q = [agg.quantize(np.array([0.5, -0.25, 1.0 / 3], dtype=np.float32)),
             agg.quantize(np.array([0.25, 0.5, -1.0 / 3], dtype=np.float32))]
        d, root, detail = agg.fedavg([3, 5], q, [3])
        check("Go and Python agree byte for byte", detail == "go+python-verified")
        try:
            agg.quantize(np.array([20.0], dtype=np.float32))
            rejected = False
        except Exception:
            rejected = True
        check("out-of-range parameter is rejected, not clamped", rejected)

        print("4. float pipeline (C1 negative control)")
        rng = np.random.default_rng(0)
        ups = [(rng.standard_normal(20000) * 0.05).astype(np.float32) for _ in range(10)]
        rep = floatpipe.divergence_report(floatpipe.aggregate_all(ups, [1000] * 10))
        check("float strategies produce >1 distinct root within one round",
              rep["distinct_roots"] > 1, "%d of %d" % (rep["distinct_roots"], rep["n_strategies"]))
        check("not all strategies diverge (red line 2: no overclaiming)",
              rep["distinct_roots"] < rep["n_strategies"])

        print("5. reproducibility of a full run")
        common = ["--rounds", "3", "--clients", "6", "--model", "logreg",
                  "--alpha", "0.5", "--seed", "7", "--crosscheck-every", "1",
                  "--timestamp", "fixed"]
        a_dir = os.path.join(tmp, "a")
        b_dir = os.path.join(tmp, "b")
        import contextlib
        import io as _io
        for d_ in (a_dir, b_dir):
            with contextlib.redirect_stdout(_io.StringIO()):
                runner.main(common + ["--runs-dir", d_])
        ra = roots_of(os.path.join(a_dir, os.listdir(a_dir)[0]))
        rb = roots_of(os.path.join(b_dir, os.listdir(b_dir)[0]))
        check("same seed gives byte-identical state roots every round", ra == rb,
              "%d rounds" % len(ra))
        check("roots differ between rounds (the model is actually moving)",
              len(set(ra)) == len(ra))

        print("6. quantization error budget")
        run_dir = os.path.join(a_dir, os.listdir(a_dir)[0])
        with open(os.path.join(run_dir, "rounds.jsonl"), encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        worst = max(r["quant_error"]["max"] for r in rows)
        # Budget: client quantization (<= 2^-(F+1)) plus one aggregate division
        # rounding (<= 2^-(F+1)), i.e. 2^-F at F=20.
        budget = 2.0 ** -20
        check("observed error within the two-stage budget 2^-F",
              worst <= budget, "%.3e <= %.3e" % (worst, budget))

        print("7. comparison arms (Phase 2)")
        from baselines import centralized, fedavg_offchain, priorwork_float
        arm_kw = dict(source="synthetic", clients=4, alpha=0.5, seed=3,
                      model="logreg", lr=0.1, rounds=2)
        cen = centralized.run(**arm_kw)
        fint = fedavg_offchain.run(backend="go", **arm_kw)
        ffl = priorwork_float.run(strategies=["seq-fwd", "seq-rev"], **arm_kw)
        check("all three arms return one row per round",
              len(cen) == len(fint) == len(ffl) == 2)
        check("integer arm records a state root", bool(fint[-1].get("state_root")))
        check("float arm forks from the first round", ffl[0]["distinct_roots"] == 2)
        check("float chains stay functionally identical",
              all(r["accuracy_spread"] == 0.0 for r in ffl),
              "worst spread %.2e" % max(r["accuracy_spread"] for r in ffl))
        # The two federated arms differ ONLY in the aggregation arithmetic, so at
        # round one -- before any divergence has compounded -- they must agree on
        # the learning metric to within the quantization error. If they do not,
        # something other than the aggregator differs and the comparison in
        # campaign.py would be measuring the wrong thing.
        gap = abs(fint[0]["test_acc"] - ffl[0]["test_acc"])
        check("federated int and float arms agree at round 1", gap <= 1e-3,
              "gap %.2e" % gap)

        print("8. manifest")
        man = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
        check("manifest records library versions", bool(man["libraries"].get("numpy")))
        check("manifest records platform", bool(man["platform"]["machine"]))
        check("manifest records aggregator backend",
              "aggregator_backend" in man["results"])
        check("manifest notes missing git commit rather than inventing one",
              "git_commit" in man)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAIL:
        print("%d check(s) FAILED: %s" % (len(FAIL), ", ".join(FAIL)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
