# Artifact Evaluation Guide

## 1. Scope

The artifact supports four kinds of claims made in the paper:

1. ordinary floating-point implementation choices can produce different
   committed aggregate bytes from identical inputs;
2. a complete integer specification makes independently written
   implementations agree;
3. the specified aggregation is integrated through EVM entry points and a
   mandatory, height-scheduled state transition; and
4. the reported gas, latency, learning, and robustness results are derived from
   retained machine-readable experiment records.

The artifact does not claim to reproduce a live heterogeneous PoS network or a
multi-epoch finality trajectory. The paper treats those as unmeasured scope.

## 2. Environment

The portable core requires Go 1.24 or later and Python 3.12 or later. The
independent Python conformance implementation uses only the Python standard
library. Learning and framework-divergence experiments use the versions pinned
in `fl-workload/requirements.txt`; their recorded environment is also embedded
in each run manifest.

Consensus-client tests use the modified go-ethereum source included at
`chain/geth/`. This location is intentional: its module maps `dfrl-agg` to
`../../dfrl-agg`. Cross-architecture observations additionally require native
`amd64` and `arm64` runners; the retained CI logs allow those results to be
inspected without either machine.

## 3. One-command core check

From the repository root:

```text
python artifact/run_quick.py
```

The runner performs the following checks and stops at the first failure:

- all `dfrl-agg` unit and property tests;
- deterministic generation of 2,000 randomized valid conformance cases;
- Go/Python agreement on the nine curated vectors; and
- Go/Python agreement on all randomized cases.

Add `--with-geth` to include the EVM entry points, mandatory scheduling,
post-state propagation, and Engine API payload verdict:

```text
python artifact/run_quick.py --with-geth
```

## 4. Claim-to-artifact map

| Paper result | Primary artifact and command | Expected observation |
|---|---|---|
| Floating-point implementation axes (Sec. 4.1) | `cd fl-workload` then `python framework_divergence.py`; `cd ../dfrl-agg` then `go test -run TestFloatDivergenceReport -v` | Idiomatic framework expressions and reduction strategies produce multiple commitments from identical inputs |
| Compact three-client examples (Sec. 4.1) | `cd dfrl-agg` then `go test -run TestMinimalCounterexample -v` | Order and precision examples reproduce on both architectures; the FMA example is architecture dependent |
| Client-triggered split (Sec. 4.3) | `cd dfrl-agg` then `go test -run TestAttackerClosure -v` | Changing one client's well-formed value separates the six floating-point implementations into two or three commitment groups |
| Mandatory boundary and payload rejection (Sec. 4.4) | `cd chain/geth` then `go test ./core -run 'TestDFRLAggregation(IsMandatory|ReachesStateRoot)$' -v` and `go test ./eth/catalyst -run TestDFRLDivergentStateRootIsINVALID -v` | The system call runs at the boundary with no transaction; its result reaches the post-state root; a payload carrying the other root is `INVALID` |
| Independent conformance (Sec. 7.1) | `python artifact/run_quick.py` | Nine curated and 2,000 fixed-seed randomized cases agree between Go and Python |
| Cross-process execution (Sec. 7.2) | After starting five isolated geth processes, `cd chain/testbed` then `go run ./driver --rpc 8600,8601,8602,8603,8604 --rounds 100 --keys keys.json --contract ../contracts/DFRLAggregator.json`; retained result: `run100.log` | Event, storage, and direct-call observations agree across all five processes for all 100 rounds |
| Real-data contract path (Sec. 7.3) | `cd chain/geth` then `go test ./core/vm -run TestX4RealMNISTOnChainMatchesOffChain -v` | Five exported MNIST rounds match through compiled Solidity bytecode and the direct aggregation path |
| Gas and latency (Secs. 8.1--8.3) | `chain/geth/core/vm/x8_grid.csv`, `x9_results/`, `fit_x8_gas_surface.py`, and `analyze_x9.py` | Regenerates the fitted gas surface and the precompile/opcode latency summaries |
| Cross-architecture integer agreement (Sec. 8.4) | `.github/workflows/arm64-divergence.yml` and the paired logs in `evidence/` | Integer commitments agree across native `amd64` and `arm64`; the negative floating-point control differs as reported |
| Learning and robustness (Secs. 8.5--8.6) | checked-in records under `fl-workload/runs/`; rerun with `campaign.py`, `fork_visibility.py`, and `robustness_experiment.py` | Reproduces the accuracy, quantization-error, silent-fork-visibility, and trimmed-mean results |

PowerShell users may replace the single quotes around Go test regular
expressions with double quotes.

## 5. Reproducing the learning results

Create the pinned Python environment, fetch and verify MNIST, and run from the
workload directory so new records remain under `fl-workload/runs/`:

```text
cd fl-workload
python -m pip install -r requirements.txt
python fetch_mnist.py
python fork_visibility.py --rounds 60 --data mnist
python campaign.py --seeds 10 --rounds 20 --threshold 0.90 --data mnist
python robustness_experiment.py
```

The 60-round and 10-seed experiments are full reproductions rather than smoke
tests. For a fast pipeline check, use `python smoke_test.py`. The framework
comparison needs the pinned NumPy and PyTorch packages; the NumPy-version
comparison additionally requires the second environment described in
`fl-workload/requirements.txt`.

## 6. Retained evidence and provenance

`evidence/README.md` indexes logs that require a particular architecture,
build tag, or consensus-client setup. Checked-in manifests record seeds,
parameters, platform, and library versions. The five real-data contract inputs
under `fl-workload/runs/x4_export/` retain the canonical client bytes and
expected aggregate for independent replay.

Solidity source and compiled JSON artifacts are both retained. The compiler
provenance is recorded in `chain/tools/SOURCE.json`. The files
`chain/testbed/keys.json` and `chain/testbed/genesis.json` belong only to
isolated development chains.

## 7. Evidence boundaries

- The Go and Python implementations are independent for valid-input arithmetic
  and commitment computation; raw EVM calldata rejection is tested on the Go
  and geth paths rather than duplicated in Python.
- The five-process experiment uses isolated development chains on one
  workstation. It does not test P2P propagation or leader election.
- The real-data contract test deliberately shares the Go aggregation core. It
  is an integration test, not a third independent arithmetic implementation.
- The Engine API test establishes local payload rejection over an identical
  parent. The subsequent PoS finality trajectory is derived from protocol rules
  and is not measured by this artifact.
