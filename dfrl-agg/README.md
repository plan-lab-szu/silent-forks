# dfrl-agg

Deterministic aggregation primitives for consensus-executed federated learning.
Reference implementation of the deterministic aggregation specification in
the accompanying paper. The consensus-relevant rules are collected in
[`DETERMINISM_SPEC.md`](DETERMINISM_SPEC.md). See the repository-level
[`ARTIFACTS.md`](../ARTIFACTS.md) for claim-to-command mappings.

## Why this package exists

On-chain model aggregation requires every validator to compute a byte-identical
global model during block execution; otherwise the state root diverges and the
chain forks. Existing designs — including this group's own prior work
(OJ-COMS 2024) — call a floating-point ML backend from inside the consensus
execution path. IEEE 754 addition is not associative, so the result depends on
summation order, library version and build configuration, none of which the
consensus protocol specifies.

This package removes floating point from the consensus path entirely: clients
quantize before submitting, and aggregation is exact integer arithmetic.
Integer addition is associative and commutative, so the output does not depend
on anything the protocol leaves unspecified.

## Layout

| file | contents |
|---|---|
| `limits.go` | protocol parameters and the bit budget (spec §2.2) |
| `tensor.go` | quantized tensor type, shape and range validation |
| `quant.go` | client-side float32 ↔ fixed-point conversion (off the consensus path) |
| `fedavg.go` | `divRoundHalfEven` and the weighted average (spec §3.1–3.2) |
| `trimmed.go` | coordinate-wise trimmed mean with address tie-break (spec §3.3, U3) |
| `codec.go` | canonical encoding, non-canonical rejection, state root (spec §4.2) |
| `gen.go` | reproducible input generation (splitmix64), shared with the Python reference |
| `floatref.go` | **negative control only** — float strategies that demonstrate divergence (C1) |
| `vectors/` | golden vectors: inputs → expected state root |
| `pyref/` | independent Python reference and the cross-language check |
| `cmd/dfrl-agg/` | CLI for the off-chain aggregator and the divergence demo |

Zero external dependencies on the consensus path — standard library only, and no
CGO. The same package is intended to back the off-chain aggregator, the geth
precompile and the fused opcode, so that "on-chain equals off-chain" (U6) and
"validators agree byte for byte" (X6) are properties of the build rather than
things to reconcile later.

`floatref.go` is the one deliberate exception to the no-float rule. Nothing on
the consensus path calls into it; its only callers are tests and the `diverge`
subcommand.

## Running

```
go test ./...                          # includes the exhaustive rounding test
go test -run TestGoldenVectors -update # regenerate vectors/vectors.json
python pyref/check_vectors.py          # cross-language byte equality (no numpy)
python pyref/fp_divergence.py          # C1 negative control (needs numpy)
go run ./cmd/dfrl-agg diverge          # same, from the Go side
```

### Randomized differential conformance

The nine curated vectors are a thin suite for a specification claiming
agreement across arbitrary conforming implementations, so a larger randomized
set is generated on demand and checked through the same independent Python
reference:

```
go test -run TestGenerateDifferentialVectors ./...   # writes vectors/differential.json
python pyref/check_vectors.py vectors/differential.json
```

`vectors/differential.json` is ~1.2 MB of derived data and is deliberately NOT
tracked in git: it is fully determined by the fixed master seed in
`differential_test.go`, so the two commands above reproduce it byte for byte.
`DFRL_DIFF_CASES=N` changes the case count (default 2000).

Every generated case is valid by construction, because a case both
implementations reject identically would test error handling rather than
agreement; malformed input is covered by the precompile rejection tests in
`chain/geth`.

The suite's sensitivity is measured rather than assumed. Replacing the Python
reference's ties-to-even rule with ties-away-from-zero -- the subtlest
deviation the rounding rule admits -- is caught by 4 of the 9 curated vectors
and 262 of the 2000 randomized cases. See
`evidence/differential_conformance.log`.

## Artifact status

The checked-in implementation and tests establish the following:

- Rounding rule verified exhaustively against an independent reference over
  `den ∈ [1,64]`, `num ∈ [-4096,4096]`, negatives and ties included.
- Order independence verified: 1000 random permutations of the client set, for
  both `FedAvg` and `TrimmedMean`, produce identical output bytes.
- Cross-language: all 9 curated vectors and 2,000 fixed-seed randomized valid
  cases agree at the commitment-and-length level between Go and an independent
  pure-Python reference, despite Go truncating integer division where Python
  floors.
- C1 negative control reproduces the prototype result on portable inputs:
  6 float strategies → 5 distinct state roots on amd64, while the integer path
  on the same inputs gives one root in both languages.

- Native Linux `amd64` and `arm64` CI runs agree on the integer commitment; the
  architecture-dependent floating-point negative control is retained under
  `evidence/`.
- The companion geth fork integrates the same arithmetic through precompile
  and opcode entry points and tests height-scheduled execution, post-state-root
  propagation, and Engine API rejection.

## What this does not claim

Determinism is execution correctness, not input quality. This package does not
make updates meaningful, detect free riders, or provide privacy; plaintext
tensors remain visible to validators, and the system model assumes an honest
validator majority.

Nor is the claim that floating point always diverges. Some strategies agree and
some do not — the defect is that nothing in the protocol determines which one a
validator runs.
