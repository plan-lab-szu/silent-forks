"""C1 negative control: floating-point aggregation is not reproducible.

This version provides three properties needed for reproducible comparison:

  1. Inputs come from splitmix64 through exactly representable powers of two
     (dfrl_ref.gen_float32), so the Go implementation can be fed provably
     identical bits. Any output difference is then attributable to the
     summation, not to the input generator.
  2. Parameterized in N, P, seed and dtype.
  3. Reports the same SHA3-256 root definition the Go side uses, so the Python
     and Go tables are directly comparable.

Some of these strategies agree and some do not, and NOTHING in the consensus protocol
determines which one a validator runs. The claim is not "float always
diverges"; it is that reproducibility here depends on summation strategy,
library version and build configuration, none of which the protocol specifies.

Usage:
  python pyref/fp_divergence.py [--n 10] [--p 250000] [--seed 20260823]
"""

import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dfrl_ref as R  # noqa: E402

PAPER_WEIGHTS = [1000, 1200, 900, 1100, 1050, 980, 1150, 1020, 970, 1130]


def root(a) -> str:
    """SHA3-256 over little-endian float32 bytes, as the prototype hashed."""
    return hashlib.sha3_256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def strategies(theta, w):
    """Each entry is a defensible implementation of the weighted average."""
    n, p = theta.shape
    S = w.sum(dtype=np.float32)
    out = {}

    acc = np.zeros(p, dtype=np.float32)
    for i in range(n):
        acc = acc + (w[i] * theta[i]).astype(np.float32)
    out["seq-fwd"] = acc / S

    acc = np.zeros(p, dtype=np.float32)
    for i in reversed(range(n)):
        acc = acc + (w[i] * theta[i]).astype(np.float32)
    out["seq-rev"] = acc / S

    out["pairwise (np.sum)"] = (w[:, None] * theta).sum(axis=0, dtype=np.float32) / S
    out["blas (np.dot)"] = (w @ theta).astype(np.float32) / S

    acc64 = np.zeros(p, dtype=np.float64)
    for i in range(n):
        acc64 += np.float64(w[i]) * theta[i].astype(np.float64)
    out["float64-acc"] = (acc64 / np.float64(S)).astype(np.float32)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=10)
    ap.add_argument("-p", "--p", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=20260823)
    a = ap.parse_args()

    theta = np.array(
        [R.gen_float32(a.seed + i * 0x1000, a.p) for i in range(a.n)], dtype=np.float32
    )
    w = np.array([PAPER_WEIGHTS[i % len(PAPER_WEIGHTS)] for i in range(a.n)], dtype=np.float32)

    res = strategies(theta, w)
    base = "seq-fwd"
    print("numpy %s | python %s | N=%d P=%s seed=%d"
          % (np.__version__, sys.version.split()[0], a.n, format(a.p, ","), a.seed))
    print("input digest (float32 LE bytes): %s\n" % root(theta)[:32])
    print("%-20s %-18s %12s %12s" % ("strategy", "root[:16]", "differs", "max |diff|"))
    print("-" * 66)
    roots = {}
    for k, v in res.items():
        d = int(np.count_nonzero(v != res[base]))
        m = float(np.max(np.abs(v.astype(np.float64) - res[base].astype(np.float64))))
        r = root(v)
        roots[r] = roots.get(r, 0) + 1
        print("%-20s %-18s %12s %12.3e" % (k, r[:16], format(d, ","), m))
    print("-" * 66)
    print("%d strategies -> %d distinct state roots" % (len(res), len(roots)))

    # The integer path, same inputs, for contrast.
    q = [[R.div_round_half_even(int(round(float(x) * (1 << R.DEFAULT_F))), 1) for x in row]
         for row in theta]
    tensors = [([a.p], R.DEFAULT_F, row) for row in q]
    weights = [int(x) for x in w]
    dims, f, agg = R.fedavg(weights, tensors)
    print("integer path (F=%d): single root %s" % (f, R.state_root(dims, f, agg)[:16]))


if __name__ == "__main__":
    raise SystemExit(main())
