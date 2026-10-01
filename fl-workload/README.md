# fl-workload

Plain federated-learning workload driving `dfrl-agg`.

## Why plain FL and not federated DQN

Determinism is the contribution (route B); the learning task is the load. A
supervised workload is *better* for the argument, not merely cheaper:

- The evaluated prior design (OJ-COMS 2024) used
  generic FL — linear regression and image classification — aggregated in
  floating point from a precompiled contract. Running the same task family lets
  C1 demonstrate the failure mode **on the very design being characterized**,
  rather than on a scenario that design never used.
- The parameter count becomes a free variable, which is what the X8 gas surface
  needs. In the channel-access workload `|Θ|` was pinned by the network
  architecture.
- Non-IID heterogeneity is a dial (Dirichlet α) and is *measured*, so R2-8's
  bounded-heterogeneity question gets a number instead of an assumption.

The workload therefore uses supervised learning so that parameter count,
heterogeneity, and aggregation arithmetic can be varied independently.

## Layout

| file | contents |
|---|---|
| `fetch_mnist.py` | fetches MNIST and triple-checks it (size / published MD5 / idx structure), writes `SOURCE.json` |
| `data.py` | dataset loading + Dirichlet non-IID split; synthetic fallback with MNIST's shape |
| `models.py` | logistic regression (7,850 params) and width-tunable MLP (`width_for_params`) |
| `client.py` | local float32 SGD; the weight `s_n` is the sample count, an on-chain integer |
| `aggbridge.py` | bridge to `dfrl-agg`: quantize, aggregate via the Go CLI, cross-check against `pyref` |
| `floatpipe.py` | prior-work-style float aggregation under every strategy (C1 negative control) |
| `framework_divergence.py` | **strongest C1 evidence**: idiomatic numpy/torch one-liners for the same weighted average give 4 distinct state roots; also reports the thread-count negative result |
| `metrics.py` | per-round telemetry: accuracy, loss, state root, quantization error, heterogeneity |
| `manifest.py` | run manifests: seed, config, git commit, library versions, platform |
| `fedcore.py` | mechanics shared by every arm, so arms differ only in the step under study |
| `run.py` | the round loop (single run) |
| `baselines/centralized.py` | no federation; reference for learning quality, compute-matched |
| `baselines/fedavg_offchain.py` | federated + exact integer aggregation via `dfrl-agg` |
| `baselines/priorwork_float.py` | one chain per float strategy, run as independent validators |
| `campaign.py` | multi-seed campaign: mean ± 95% CI, paired t and Wilcoxon |
| `fork_visibility.py` | does the fork ever show up in accuracy? |
| `smoke_test.py` | end-to-end checks |

numpy only for the model — no torch. The float pipeline needs explicit control
over how sums are associated, and a framework choosing its own kernels, thread
counts and fusion would inject exactly the uncontrolled variables the C1
experiment studies. torch is installed and available if a convolutional
workload is ever wanted.

## Running

```
python fetch_mnist.py
python smoke_test.py
python run.py --data mnist --rounds 15 --model logreg --alpha 0.5 --lr 0.1 --float-pipe
python run.py --data mnist --rounds 5 --model mlp --target-params 250000 --backend both
```

`--data` defaults to `synthetic`; `--data mnist` needs no further arguments once
`fetch_mnist.py` has run. The source is recorded in every manifest, so a
synthetic-data number can never be mistaken for an MNIST number.

Every run writes `runs/<run_id>/manifest.json` and `rounds.jsonl`. **All figures
must be generated from those files**, never from a value still in memory
(R1-5b needs ≥10 seeds with mean ± 95% CI and paired tests; that is impossible
to assemble retroactively).

Multi-seed campaign and the fork-visibility experiment:

```
python campaign.py --seeds 10 --rounds 20 --threshold 0.90 --data mnist
python fork_visibility.py --rounds 60 --data mnist
```

`--keep-submissions` persists each client's canonical submission per round. That
is the raw material for the replay audit (C3 / X7): a third party recomputes the
aggregate from those bytes and compares. Off by default for disk cost only.

## What already holds

From `smoke_test.py`, all passing:

- Integer path: same seed → **byte-identical state roots every round**, and the
  Go aggregate is verified against the independent Python reference each round
  (`go+python-verified`).
- Float path on the *same* client updates: **4 distinct state roots within a
  single round** — four validators, four chains. Not all five strategies
  diverge, so the result establishes that reproducibility is unspecified
  rather than that floating point always diverges.
- Quantization error stays inside the two-stage budget: client quantization
  (≤ 2^-(F+1)) plus one aggregate division rounding (≤ 2^-(F+1)) = 2^-F.
  Observed 9.20e-07 against a 9.54e-07 budget at F=20 (worst over the
  10-seed x 20-round campaign, runs/874fd18b0d4e). An earlier note here said
  8.2e-07; that was a stale figure from a smaller run and the paper had copied
  it. Note this is the end-to-end quantize/aggregate/dequantize error (two
  roundings, budget 2^-F), not quantization alone (one rounding, 2^-(F+1)).
- Out-of-range parameters are rejected, never clamped.

## On real MNIST (2026-08-24)

Federated logistic regression, 10 clients, integer aggregation via `dfrl-agg`:

| α | measured heterogeneity (mean TV) | shard sizes | final test accuracy |
|---|---|---|---|
| 100 (near IID) | 0.036 | 5,772 – 6,358 | 0.9071 |
| 0.5 | 0.446 | 2,009 – 9,826 | 0.8956 |
| 0.1 (severe) | 0.695 | 272 – 15,640 | 0.7835 |

90.7% at α=100 is the expected range for multinomial logistic regression on
MNIST, so the pipeline learns correctly rather than merely running. The α column
is the direct answer to R2-8: heterogeneity is measured and its cost quantified,
not assumed to be bounded.

The 250k-parameter MLP reaches 0.9006 in five rounds, every round cross-verified
between Go and Python.

Throughout, every round shows **4 distinct float state roots** on the identical
client updates, against one reproducible integer root.

## Comparison arms, 10 seeds (2026-08-24)

`campaign.py --seeds 10 --rounds 20 --threshold 0.90 --data mnist`, logistic
regression, 10 clients, Dirichlet α=0.5:

| arm | final accuracy (mean ± 95% CI) | rounds to 0.90 |
|---|---|---|
| centralized | 0.9240 ± 0.0007 | 0.0 (10/10 seeds) |
| fed-int | 0.9088 ± 0.0009 | 8.9 (10/10) |
| fed-float | 0.9088 ± 0.0009 | 8.9 (10/10) |

**fed-int vs centralized**: mean difference −0.01528, 95% CI [−0.01611, −0.01445],
paired t p=1.3e-11, Wilcoxon p=0.002. This is the cost of **federation under
non-IID data**, not the cost of the chain — any federated deployment pays it. It
must never be reported as a blockchain overhead.

**fed-int vs fed-float**: the paired differences are **exactly zero on all ten
seeds**, so no statistical test is meaningful (the driver says so rather than
printing a p-value as if it meant something). The two arms differ only in the
aggregation arithmetic, and that difference does not reach the learning metric at
all. Meanwhile the float arm's five chains hold **4 distinct state roots**, with a
mean maximum pairwise parameter distance of 3.2e-07 at the final round.

That pair of lines is the result: identical learning, irreconcilable state.

The chain's own cost is therefore neither of the above. It is gas, latency and
on-chain bytes (Phase 3) plus quantization error (≤ 2^-F, measured at ~8e-07).
Answering R2-14 means giving that decomposition rather than one headline number.

## The fork is invisible in the learning metrics

`fork_visibility.py`, two validators differing only in the association order of
one sum, 60 rounds on MNIST:

| round | distinct state roots | max \|ΔΘ\| | accuracy | spread |
|---|---|---|---|---|
| 0 | 2 | 2.98e-08 | 0.8028 | 0.00000 |
| 12 | 2 | 2.38e-07 | 0.9057 | 0.00000 |
| 36 | 2 | 4.47e-07 | 0.9163 | 0.00000 |
| 59 | 2 | 6.86e-07 | 0.9184 | 0.00000 |

The roots differ from round 0 and never re-converge — at the consensus layer that
is a permanent partition. The parameter distance grows about 23× but stays at the
1e-7 scale, and the test accuracy is **identical to every digit for all 60
rounds**: a 1e-7 perturbation almost never flips an argmax.

So the defect cannot be found by watching learning quality. A deployment
monitoring accuracy sees a healthy system while its validators hold
irreconcilable state. That is a plausible reason the failure mode went unremarked
in prior on-chain aggregation designs, this group's own included, and it is the
strongest argument for making determinism a property of the specification rather
than something checked empirically afterwards.

These measurements do not show that floating-point aggregation harms learning.
They isolate a consensus problem: byte-distinct committed states can coexist
with indistinguishable learning metrics.

## Artifact scope

The checked-in records include the 10-seed, 20-round comparison campaign, the
60-round fork-visibility run, five exported MNIST rounds for contract replay,
and the trimmed-mean robustness illustration. The NumPy-version comparison is
the only result that requires two Python environments; the exact versions and
manual comparison procedure are documented in `requirements.txt`.
