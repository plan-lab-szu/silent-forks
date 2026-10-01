"""Fit and plot the X8 contract-layer gas surface from x8_grid.csv.

  python fit_x8_gas_surface.py

Completes X8 (REWRITE_PLAN Phase 3c checklist): the precompile-layer
(N, |Theta|) gas grid already existed (fit_gas_model.py, Phase 3a); this
does the same for the CONTRACT layer (aggregateModel, both the commit-only
and full-storage variants), using the full 5x5 = 25-point grid from
TestX8ContractGasSurface instead of the 2-point U4 sample.

Same method as fit_gas_model.py and for the same reason: gas spans four
orders of magnitude across this grid (70K to 389M), so unweighted OLS would
be dominated by the largest rows. Weighted least squares (weight = 1/gas^2)
fits relative error instead.

Model: gas ~= C_base + mu1*p + mu2*n*p  -- same functional form as the
precompile-layer model, since aggregateModel's dominant cost (SLOAD/SSTORE
over n entries of size p, see HANDBOOK.md's "O(n^2) fix and diagnosis")
is architecturally the same shape: a fixed floor plus per-parameter and
per-(client,parameter) linear terms.
"""

import csv

import numpy as np


def load(path="x8_grid.csv"):
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append((int(row["n"]), int(row["p"]),
                         int(row["commit_gas"]), int(row["full_gas"]),
                         int(row["precompile_estimate"])))
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


def fit_and_report(label, n, p, y):
    print("\n=== %s: weighted least squares (weight=1/gas^2), 3-term (precompile-layer form) ===" % label)
    X3 = np.column_stack([np.ones_like(p), p, n * p])
    coef3, r23, pred3, rel3 = fit(X3, y, w=1.0 / y**2)
    print("gas = %.2f + %.5f*p + %.5f*(n*p)   R^2=%.5f   median_rel_err=%.3f   worst_rel_err=%.3f"
          % (*coef3, r23, np.median(rel3), rel3.max()))

    print("=== %s: weighted least squares, 4-term (+ standalone n term) ===" % label)
    X4 = np.column_stack([np.ones_like(p), p, n * p, n])
    coef4, r24, pred4, rel4 = fit(X4, y, w=1.0 / y**2)
    print("gas = %.2f + %.5f*p + %.5f*(n*p) + %.2f*n   R^2=%.5f   median_rel_err=%.3f   worst_rel_err=%.3f"
          % (*coef4, r24, np.median(rel4), rel4.max()))
    print("worst relative errors (4-term):")
    for i in np.argsort(-rel4)[:3]:
        print("  n=%-3d p=%-7d actual=%-14.0f pred=%-14.0f rel_err=%.3f"
              % (n[i], p[i], y[i], pred4[i], rel4[i]))

    if rel4.max() < rel3.max():
        print("-> 4-term reduces worst-case error (%.3f -> %.3f); a standalone per-N,"
              " P-independent cost is real at the contract layer (submitter-array/struct"
              " bookkeeping SSTORE), not captured by the precompile-layer's 3-term form."
              % (rel3.max(), rel4.max()))
        return coef4
    print("-> 4-term does not improve worst-case error; keeping 3-term model.")
    return coef3


def main():
    rows = load()
    n, p, commit, full, pcest = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]

    coef_commit = fit_and_report("commit-only", n, p, commit)
    coef_full = fit_and_report("full-storage", n, p, full)

    print("\n=== full-storage / commit-only ratio, by grid point ===")
    ratio = full / commit
    print("min=%.3fx (n=%d,p=%d)   max=%.3fx (n=%d,p=%d)   mean=%.3fx" % (
        ratio.min(), n[np.argmin(ratio)], p[np.argmin(ratio)],
        ratio.max(), n[np.argmax(ratio)], p[np.argmax(ratio)],
        ratio.mean()))
    # Report the ratio's actual argmin/argmax rather than a hand-written
    # characterization. An earlier version of this line asserted "lowest at
    # large N/small P", which contradicted the argmin it printed one line
    # above (the minimum sits at the LARGEST p); the paper copied that wrong
    # characterization. What the grid actually shows: the ratio is governed
    # by N and is weak and non-monotonic in p.
    for lo, hi in [(1, 1), (20, 20)]:
        row = ratio[n == lo]
        print("  N=%-2d: ratio ranges %.2fx--%.2fx over p" % (lo, row.min(), row.max()))
    print("NOT constant: driven by N (~%.1fx at N=%d down to ~%.1fx at N=%d); "
          "dependence on P is weak and non-monotonic."
          % (ratio.max(), n[np.argmax(ratio)], ratio.min(), n[np.argmin(ratio)]))
    print("Why: full-storage adds one fixed-size-in-N extra SSTORE of the P-sized result;")
    print("commit-only cost already scales with N*P, so the extra term matters relatively")
    print("less as N grows and more as N shrinks toward 1.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available -- skipping plot)")
        return

    ns_sorted = sorted(set(n.astype(int)))
    ps_sorted = sorted(set(p.astype(int)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, (label, y) in zip(axes[:2], [("commit-only", commit), ("full-storage", full)]):
        for nn in ns_sorted:
            mask = n == nn
            order = np.argsort(p[mask])
            ax.plot(p[mask][order], y[mask][order], marker="o", label="N=%d" % nn)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|Theta| (params)")
        ax.set_ylabel("aggregateModel gas")
        ax.set_title("X8: %s" % label)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    for nn in ns_sorted:
        mask = n == nn
        order = np.argsort(p[mask])
        ax.plot(p[mask][order], ratio[mask][order], marker="o", label="N=%d" % nn)
    ax.set_xscale("log")
    ax.set_xlabel("|Theta| (params)")
    ax.set_ylabel("full-storage / commit-only gas ratio")
    ax.set_title("X8: storage tax ratio (U4)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("x8_gas_surface.png", dpi=150)
    print("\nwrote x8_gas_surface.png")


if __name__ == "__main__":
    main()
