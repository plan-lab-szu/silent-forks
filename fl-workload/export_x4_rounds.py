"""Export real MNIST fed-int rounds as canonical dfrl-agg bytes, for X4.

  python export_x4_rounds.py
  python export_x4_rounds.py --model mlp --seed 2 --out runs/scope_export/mlp_seed2

The core check (on-chain result byte-identical to a direct dfrl-agg call) was
already verified in chain/geth/core/vm/contracts_dfrl_contract_test.go, but
only on SYNTHETIC random tensors. This exporter enables the same check on REAL local updates coming out of
actual MNIST training -- both because "synthetic tensor shaped like a model"
and "an actual model's gradient/weight update" are not guaranteed to hit the
same code paths (e.g. value-range edge cases quantize() might reject) and
because it lets X8's gas surface fit be checked at a real, paper-relevant
|Theta| (7,850, the LogReg parameter count) instead of only at the grid
points the fit was trained on.

Writes, under runs/x4_export/round%04d/:
  client%03d_w%d.bin  -- canonical-encoded quantized update, one per client
                          (same format keep_submissions already produces for
                          the C3/X7 replay audit, reused here)
  expected.bin         -- this round's aggregate, canonical-encoded, computed
                          by the SAME go-backend dfrl-agg call the fed-int arm
                          already uses -- this is what the on-chain result
                          must match bit-for-bit
  expected.root        -- that aggregate's state root (hex), for a second,
                          independent comparison point

Also writes runs/x4_export/manifest.json with per-round metadata (weights in
upload order, dims, f) so the Go-side test does not need to re-derive them
from filenames.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fedcore
import models
from aggbridge import Aggregator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "runs", "x4_export")


def parse_args():
    """Defaults reproduce the original X4 export exactly; flags exist for the
    scope re-verification grid (second model, several seeds) agreed 2026-09-17.
    Nothing here changes what `python export_x4_rounds.py` produces."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logreg", choices=("logreg", "mlp"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=314)
    ap.add_argument("--backend", default="go", choices=("go", "python", "both"),
                    help="python avoids the Go binary; needed where an OS "
                         "application-control policy blocks executing it")
    ap.add_argument("--out", default=OUT_DIR)
    return ap.parse_args()


def main():
    args = parse_args()
    clients, rounds, alpha, seed = args.clients, args.rounds, args.alpha, args.seed
    model_kind, hidden, lr, batch, local_epochs = args.model, args.hidden, 0.1, 64, 1
    out_dir = args.out

    os.makedirs(out_dir, exist_ok=True)

    Xtr, ytr, Xte, yte, shards = fedcore.make_split("mnist", None, clients, alpha, seed)
    net = models.build(model_kind, hidden=hidden, seed=seed)
    dims = [net.n_params]
    print("model=%s |Theta|=%d clients=%d rounds=%d" % (model_kind, net.n_params, clients, rounds))
    cs = fedcore.make_clients(Xtr, ytr, shards, model_kind, hidden, seed, lr, batch, local_epochs)
    agg = Aggregator(backend=args.backend)
    f = agg.quantize.__defaults__[0]

    flat = net.flat()
    manifest = {"clients": clients, "rounds": rounds, "dims": dims, "f": f,
                "model": model_kind, "seed": seed, "alpha": alpha,
                "backend": agg.backend_detail,
                "hidden": hidden if model_kind == "mlp" else None,
                "lr": lr, "batch": batch, "local_epochs": local_epochs,
                "round_meta": []}
    for rnd in range(rounds):
        updates, weights, losses = fedcore.local_updates(cs, flat, seed, rnd)
        quantized = [agg.quantize(v, f) for v in updates]

        round_dir = agg.keep_submissions(out_dir, rnd, weights, quantized, dims, f)

        agg_data, root, detail = agg.fedavg(weights, quantized, dims, f, round_index=rnd)
        with open(os.path.join(round_dir, "expected.bin"), "wb") as fh:
            import dfrl_ref as R
            fh.write(R.encode(list(dims), f, agg_data))
        with open(os.path.join(round_dir, "expected.root"), "w", encoding="utf-8") as fh:
            fh.write(root.strip() + "\n")

        flat = agg.dequantize(agg_data, f)
        acc = fedcore.evaluate(net, flat, Xte, yte)["test_acc"]
        manifest["round_meta"].append({
            "round": rnd, "weights": [int(w) for w in weights],
            "backend": detail, "root": root.strip(), "test_acc": acc,
            "train_loss": float(sum(losses) / len(losses)),
        })
        print("  round %d: backend=%s acc=%.4f root=%s..." % (rnd, detail, acc, root[:16]))

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print("wrote %d rounds to %s" % (rounds, out_dir))


if __name__ == "__main__":
    main()
