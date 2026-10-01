"""Statistics for the X9 precompile-vs-opcode latency redo.

  python analyze_x9.py

Reads every x9_results/n*_p*.json file (written by
TestX9PrecompileVsOpcodeLatency), and for each (n, p) point reports:

  - mean +/- 95% CI (t-based) per dispatch path
  - a PAIRED comparison of opcode vs precompile (same round index, both
    measured under interleaved-random order -- see the Go test's doc
    comment for why interleaving makes this pairing meaningful) using both
    a paired t-test and Wilcoxon signed-rank, exactly the two tests
    fl-workload/campaign.py already applies to the FL-side numbers. Same
    statistical standard, same honesty rule: if the CI on the paired
    difference straddles zero, that is reported as "no significant
    difference", not rounded off in either direction.
"""

import glob
import json
import os
import sys

import numpy as np
from scipy import stats


def mean_ci(values, level=0.95):
    a = np.asarray(values, dtype=np.float64)
    n = a.size
    if n < 2:
        return float(a.mean()) if n else None, None, n
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * a.std(ddof=1) / np.sqrt(n))
    return float(a.mean()), half, n


def paired(a, b, level=0.95):
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    d = x - y
    m, half, n = mean_ci(d, level)
    out = {"n": n, "mean_diff": m, "ci_half_width": half,
           "ci": [m - half, m + half] if half is not None else None}
    if n >= 2:
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


def _fmt_p(p):
    return "n/a" if p is None else ("%.4g" % p)


def analyze_one(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    n, p, rounds = d["n"], d["p"], d["rounds"]
    direct, via_pre, via_op = d["direct_ns"], d["via_precompile_ns"], d["via_opcode_ns"]

    print("\n=== n=%d p=%d (%d interleaved rounds) ===" % (n, p, rounds))
    for label, xs in (("direct pc.Run", direct),
                       ("EVM+STATICCALL(precompile)", via_pre),
                       ("EVM+FEDAVG(opcode)", via_op)):
        m, half, cnt = mean_ci(xs)
        print("  %-28s %12.0f +/- %8.0f ns/op  (n=%d)" % (label, m, half, cnt))

    cmp_op_pre = paired(via_op, via_pre)
    print("  opcode - precompile: mean diff %+.0f ns, 95%% CI [%+.0f, %+.0f]"
          % (cmp_op_pre["mean_diff"], cmp_op_pre["ci"][0], cmp_op_pre["ci"][1]))
    print("      paired t p=%s   wilcoxon p=%s%s" % (
        _fmt_p(cmp_op_pre.get("ttest_rel_p")), _fmt_p(cmp_op_pre.get("wilcoxon_p")),
        "  (%s)" % cmp_op_pre["wilcoxon_note"] if cmp_op_pre.get("wilcoxon_note") else ""))

    lo, hi = cmp_op_pre["ci"]
    if lo <= 0 <= hi:
        verdict = "NO STATISTICALLY SIGNIFICANT DIFFERENCE (95% CI straddles zero)"
    elif hi < 0:
        pct = -cmp_op_pre["mean_diff"] / mean_ci(via_pre)[0] * 100
        verdict = "opcode SIGNIFICANTLY FASTER than precompile-via-STATICCALL (~%.1f%%)" % pct
    else:
        pct = cmp_op_pre["mean_diff"] / mean_ci(via_pre)[0] * 100
        verdict = "opcode SIGNIFICANTLY SLOWER than precompile-via-STATICCALL (~%.1f%%)" % pct
    print("  VERDICT: %s" % verdict)
    return {"n": n, "p": p, "verdict": verdict, "comparison": cmp_op_pre}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "x9_results", "n*_p*.json")))
    if not files:
        print("no x9_results/*.json found -- run: go test ./core/vm/ -run TestX9PrecompileVsOpcodeLatency -v")
        return 1
    results = [analyze_one(f) for f in files]
    print("\n--- summary ---")
    for r in results:
        print("  n=%-3d p=%-6d %s" % (r["n"], r["p"], r["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
