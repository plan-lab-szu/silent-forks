"""C1, strongest form: idiomatic ML code in mainstream frameworks forks the chain.

  python framework_divergence.py

This supersedes floatpipe.py as the *headline* C1 evidence, and the difference
matters for how the contribution can be claimed.

floatpipe.py compares summation strategies we wrote ourselves (forward loop,
reverse loop, pairwise, blocked, float64-accumulate). A reviewer can fairly
respond: "you deliberately wrote five different reduction loops; of course they
differ -- no real implementation would do that."

This script removes that objection. It compares:

  * the TWO most idiomatic ways to express a weighted average in array code:
        w @ theta                      (matrix-vector product)
        (w[:, None] * theta).sum(0)    (broadcast-multiply then reduce)
    These are the same formula. No engineer would call them different
    algorithms; nobody would document which one their validator uses.

  * the TWO mainstream CPU frameworks a validator implementation would
    plausibly link against (numpy, torch), each with its own BLAS/oneDNN
    backend.

Result (see the table this prints): 2 formulations x 2 frameworks = **four
distinct state roots**, at both the paper's parameter scale (N=10, P=250k) and
the prior work's node count (N=400). The inputs are provably bit-identical --
generated through exactly-representable powers of two (same construction as
dfrl-agg/pyref) and their digest is printed so it can be checked.

So the claim is not "floating point is risky if you write unusual code." It is:
**the ordinary way to write this computation is unspecified in a way that
decides whether the chain forks.** Which BLAS your framework links against is
not something any consensus protocol states.

A THREAD-COUNT SWEEP IS ALSO RUN (thread_sweep(), called from main()), because
an earlier version of this docstring asserted thread count never matters
without any code that actually tested it -- os.environ["OMP_NUM_THREADS"]
set after numpy has already loaded its BLAS library has no effect, so a naive
sweep silently measures the same thread count five times. Once the sweep is
done properly (threadpoolctl.threadpool_limits for numpy, torch.set_num_threads
for torch -- see thread_sweep()'s docstring), the result is NOT uniformly
negative:

  numpy's "(w[:,None]*t).sum(0)" and both torch forms ARE thread-count
  invariant across nt in {1,2,4,8,16}, at both N=10 and N=400. Reason: FedAvg
  reduces over the CLIENT dimension, which is small (10-400), while these
  three code paths parallelize (if at all) over the PARAMETER dimension,
  which does not reorder any individual element's reduction.

  numpy's "w @ theta" -- the one BLAS-dispatched formulation, and the one
  used as the baseline column above -- IS thread-count dependent: 5 distinct
  roots across nt in {1,2,4,8,16} at N=10,P=250k, 4 distinct roots at
  N=400,P=20k. OpenBLAS's threaded sgemv kernel partitions the reduction
  into per-thread blocks whose count and boundaries depend on the thread
  count, so which roundings happen in which order changes with it. This is a
  FIFTH practical divergence axis, orthogonal to summation strategy, library
  version, framework choice, and CPU architecture -- and arguably the
  sharpest one, since thread count is universally treated as a pure
  performance knob with no semantic content.

  So "validators have different core counts" is a divergence source for
  operations that dispatch to a threaded BLAS kernel, and is not for
  operations that don't -- a real threat model has to say which
  configuration differences matter per code path, not attach the answer to
  a whole framework or a whole operation.

This does not establish that floating point always diverges. It shows that
reproducibility here depends on framework choice, expression form, and (for
BLAS-dispatched paths) thread count, none of which the protocol specifies.
"""

import hashlib

import numpy as np

MASK64 = (1 << 64) - 1


class SplitMix64:
    """Same generator as dfrl-agg/gen.go and pyref/dfrl_ref.py, so inputs are
    reproducible across every experiment in this project."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


def gen_float32(seed, p):
    """float32 values in [-2^-4, 2^-4), every one exactly representable, so the
    input carries no rounding of its own and cannot be blamed for a difference."""
    r = SplitMix64(seed)
    return [(float(r.next() >> 40) - 8388608.0) * (2.0**-27) for _ in range(p)]


def root(a):
    return hashlib.sha3_256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def make_inputs(n, p, seed=20260823):
    theta = np.array([gen_float32(seed + i * 0x1000, p) for i in range(n)], dtype=np.float32)
    w = np.array([1000 + (i % 7) * 37 for i in range(n)], dtype=np.float32)
    return theta, w


def variants(theta, w):
    """The four aggregations. Every one computes sum_n(w_n * theta_n) / sum(w)."""
    import torch

    S = w.sum(dtype=np.float32)
    tt, tw = torch.from_numpy(theta), torch.from_numpy(w)
    tS = tw.sum()
    return {
        "numpy  w @ theta": (w @ theta) / S,
        "numpy  (w[:,None]*t).sum(0)": (w[:, None] * theta).sum(axis=0, dtype=np.float32) / S,
        "torch  w @ theta": ((tw @ tt) / tS).numpy(),
        "torch  (w[:,None]*t).sum(0)": ((tw[:, None] * tt).sum(dim=0) / tS).numpy(),
    }


def thread_sweep(theta, w):
    """Actually vary the thread count and re-check each root, instead of only
    asserting a claim in print output. threadpoolctl.threadpool_limits is used
    for the numpy/BLAS path because setting OMP_NUM_THREADS/OPENBLAS_NUM_THREADS
    via os.environ after numpy has already loaded its BLAS library has no
    effect -- OpenBLAS reads its thread count once, at library load time, not
    per call. torch has its own intraop thread pool, controlled separately via
    torch.set_num_threads. This distinction is the reason an earlier version of
    this script only ever printed the claim without a test behind it: a naive
    os.environ-based sweep silently measures the same thread count five times
    and (mis)reports agreement."""
    import torch
    from threadpoolctl import threadpool_limits

    S = w.sum(dtype=np.float32)
    tt, tw = torch.from_numpy(theta), torch.from_numpy(w)
    tS = tw.sum()

    results = {name: {} for name in ("numpy  w @ theta", "numpy  (w[:,None]*t).sum(0)",
                                      "torch  w @ theta", "torch  (w[:,None]*t).sum(0)")}
    for nt in (1, 2, 4, 8, 16):
        with threadpool_limits(limits=nt):
            results["numpy  w @ theta"][nt] = root((w @ theta) / S)
            results["numpy  (w[:,None]*t).sum(0)"][nt] = root(
                (w[:, None] * theta).sum(axis=0, dtype=np.float32) / S)
        torch.set_num_threads(nt)
        results["torch  w @ theta"][nt] = root(((tw @ tt) / tS).numpy())
        results["torch  (w[:,None]*t).sum(0)"][nt] = root(((tw[:, None] * tt).sum(dim=0) / tS).numpy())
    return results


def main():
    import torch

    print("numpy %s | torch %s" % (np.__version__, torch.__version__))
    print()
    overall = {}
    all_theta, all_w = {}, {}
    for n, p in [(10, 250_000), (400, 20_000)]:
        theta, w = make_inputs(n, p)
        all_theta[(n, p)], all_w[(n, p)] = theta, w
        res = variants(theta, w)
        base = res["numpy  w @ theta"]
        print("N=%-4d P=%-7d   input digest %s" % (n, p, root(theta)[:24]))
        print("  %-30s %-26s %10s %12s" % ("aggregation", "state root[:24]", "differs", "max |diff|"))
        print("  " + "-" * 82)
        roots = {}
        for name, v in res.items():
            r = root(v)
            roots.setdefault(r, []).append(name)
            d = int(np.count_nonzero(v != base))
            m = float(np.max(np.abs(v.astype(np.float64) - base.astype(np.float64)))) if d else 0.0
            print("  %-30s %-26s %10s %12.3e" % (name, r[:24], format(d, ","), m))
        print("  " + "-" * 82)
        print("  %d formulations -> %d DISTINCT state roots" % (len(res), len(roots)))
        overall[(n, p)] = len(roots)
        print()

    print("=" * 84)
    for k, v in overall.items():
        print("N=%-4d P=%-7d -> %d distinct roots" % (k[0], k[1], v))
    print()

    print("=" * 84)
    print("Thread-count sweep (1/2/4/8/16), same inputs as above, each formulation")
    print("re-evaluated under threadpoolctl.threadpool_limits (numpy) / torch.set_num_threads (torch):")
    print()
    for (n, p) in overall:
        sweep = thread_sweep(all_theta[(n, p)], all_w[(n, p)])
        print("N=%-4d P=%-7d" % (n, p))
        for name, by_nt in sweep.items():
            distinct = len(set(by_nt.values()))
            flag = "VARIES WITH THREAD COUNT" if distinct > 1 else "invariant"
            print("  %-30s %d distinct root(s) across nt in {1,2,4,8,16}  -- %s" % (name, distinct, flag))
            if distinct > 1:
                for nt, r in by_nt.items():
                    print("      nt=%-2d  %s" % (nt, r[:24]))
        print()


if __name__ == "__main__":
    raise SystemExit(main())
