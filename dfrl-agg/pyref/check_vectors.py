"""Cross-language check: does the Python reference land on Go's state roots?

Run from the module root:  python pyref/check_vectors.py

This is the check that gives contribution C1 its positive half. The negative
half (fp_divergence.py) shows that floating-point aggregation produces
different roots under equally reasonable implementations; this shows that the
integer specification produces the SAME root across two independent
implementations in two languages with different integer division semantics.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dfrl_ref as R  # noqa: E402

DEFAULT_VECTORS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vectors", "vectors.json")

# A path may be given on the command line so the same checker verifies both the
# nine curated golden vectors and the large randomized differential suite
# emitted by differential_test.go. The two files share one schema on purpose:
# the value of this check comes from the Python side sharing no code with the Go
# side, so it should not need Go-specific handling per suite.
VECTORS = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VECTORS


def run_case(c):
    addrs, weights, tensors = R.gen_updates(
        c["seed"], c["n"], c["p"], c["weights"], c["bound"], c["f"], c.get("dims")
    )
    if c["op"] == "trimmed":
        dims, f, data = R.trimmed_mean(addrs, weights, tensors, c.get("k", 0))
    elif c["op"] == "fedavg":
        dims, f, data = R.fedavg(weights, tensors)
    else:
        raise SystemExit("unknown op " + c["op"])
    return R.state_root(dims, f, data), len(R.encode(dims, f, data))


def main():
    with open(VECTORS, encoding="utf-8") as fh:
        doc = json.load(fh)
    print("spec:  %s" % doc["spec"])
    print("hash:  %s" % doc["hash"])
    print("%-28s %-18s %-10s %s" % ("case", "root[:16]", "bytes", "verdict"))
    print("-" * 72)
    bad = 0
    for c in doc["cases"]:
        root, n = run_case(c)
        ok = root == c["root"] and n == c["encoded_len"]
        bad += 0 if ok else 1
        print("%-28s %-18s %-10d %s" % (c["name"], root[:16], n, "MATCH" if ok else "MISMATCH"))
        if not ok:
            print("    go:     %s (%d bytes)" % (c["root"], c["encoded_len"]))
            print("    python: %s (%d bytes)" % (root, n))
    print("-" * 72)
    print("%d/%d cases byte-identical across Go and Python" % (len(doc["cases"]) - bad, len(doc["cases"])))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
