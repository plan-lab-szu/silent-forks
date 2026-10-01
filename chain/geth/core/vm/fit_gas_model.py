"""Fit the R2-10 gas model to the benchmark CSV in gas_bench.csv.

  python fit_gas_model.py

Kept as a checked-in script (not a one-off shell heredoc) so the fit is
reproducible and its method is inspectable rather than asserted.

Model: ns_per_op ~= C_base + mu1*p + mu2*n*p, matching the affine-in-|Theta|
model R2-10 asked for, extended with an N*|Theta| term for the same reason
R2-10 gave (the original single-variable affine model in the submitted
manuscript ignored client count).

Why NOT plain OLS on the raw nanosecond scale: the benchmark spans p from 16
to 250,000, so ns_per_op spans about 4 orders of magnitude (1,023 ns to
38,000,000 ns). Unweighted least-squares minimizes ABSOLUTE squared error,
which is completely dominated by the largest rows -- fitting unweighted here
gives an R^2 of 0.9985 that LOOKS excellent while mispredicting the small-p
rows by 40-60x (a p=16 point costs ~1,000 ns; the unweighted fit predicts an
intercept alone of ~69,000 ns). That is exactly the small-|Theta| regime
R1-6c and R2-10 care about (calldata-dominated cost, near the per-tx floor),
so getting it wrong there is not a rounding error, it is getting the part
that matters wrong.

Fix: weighted least squares with weight 1/ns^2, which is the standard
correction when the target spans orders of magnitude and every point should
count on comparable RELATIVE terms rather than absolute nanoseconds. Report
both fits so the difference is visible and the choice is not laundered.
"""

import numpy as np


def load(path="gas_bench.csv"):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n, p, inbytes, ns = (int(x) for x in line.split(","))
            rows.append((n, p, inbytes, ns))
    return np.array(rows, dtype=np.float64)


def fit(X, y, w=None):
    if w is None:
        Xw, yw = X, y
    else:
        sw = np.sqrt(w)
        Xw, yw = X * sw[:, None], y * sw
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    rel = np.abs(pred - y) / y
    return coef, r2, pred, rel


def main():
    rows = load()
    n, p, _, ns = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    X = np.column_stack([np.ones_like(p), p, n * p])

    print("=== unweighted OLS (dominated by large-p rows) ===")
    coef, r2, pred, rel = fit(X, ns)
    print("ns_per_op = %.1f + %.4f*p + %.4f*(n*p)   R^2=%.5f" % (*coef, r2))
    print("worst relative errors:")
    for i in np.argsort(-rel)[:5]:
        print("  n=%-3d p=%-7d actual=%-12.0f pred=%-12.0f rel_err=%.2fx"
              % (n[i], p[i], ns[i], pred[i], rel[i]))

    print("\n=== weighted least squares, weight = 1/ns^2 (relative-error fit) ===")
    coefw, r2w, predw, relw = fit(X, ns, w=1.0 / ns**2)
    print("ns_per_op = %.2f + %.5f*p + %.5f*(n*p)   R^2=%.5f" % (*coefw, r2w))
    print("worst relative errors:")
    for i in np.argsort(-relw)[:5]:
        print("  n=%-3d p=%-7d actual=%-12.0f pred=%-12.0f rel_err=%.2fx"
              % (n[i], p[i], ns[i], predw[i], relw[i]))
    print("median relative error: %.3f  (unweighted fit's was %.3f)"
          % (float(np.median(relw)), float(np.median(rel))))

    Cbase, mu1, mu2 = coefw
    print("\n=== converting to gas ===")
    print("Choosing 1 gas = 1 ns is a reporting convention, not a physical law:")
    print("it makes the RELATIVE cost structure (base vs per-param vs per-client-param)")
    print("directly readable while leaving the absolute scale to whoever calibrates")
    print("against a real fee market. Multiply by any constant k to retarget gas/sec.")
    print("dfrlGasBase      = %d" % round(Cbase))
    print("dfrlGasPerParam  = %d   (mu1, rounded)" % max(1, round(mu1)))
    print("dfrlGasPerNParam = %d   (mu2, rounded)" % max(1, round(mu2)))


if __name__ == "__main__":
    main()
