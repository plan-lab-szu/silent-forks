# Silent Forks

Research artifact for *Exploiting Floating-Point Nondeterminism: A Security
Threat to On-Chain Federated Learning*.

This repository contains the deterministic aggregation implementation, an
independent conformance oracle, federated-learning workloads, Solidity
contracts, the modified consensus client, retained experiment records, and the
scripts used to reproduce the reported results.

Start with [ARTIFACTS.md](ARTIFACTS.md), which maps each evaluated claim to its
implementation, command, expected observation, and retained evidence.

## Quick check

The portable core requires Go 1.24 or later and Python 3.12 or later. From the
repository root, run:

```text
python artifact/run_quick.py
```

This command runs the Go unit and property tests, regenerates a fixed-seed
randomized conformance corpus, and checks the curated and randomized cases
against the independent Python implementation. NumPy and PyTorch are not
needed for this check.

The repository includes the modified go-ethereum source under `chain/geth`.
To include the consensus and EVM integration tests, run:

```text
python artifact/run_quick.py --with-geth
```

## Repository layout

| Path | Contents |
|---|---|
| `dfrl-agg/` | Deterministic Go implementation, pure-Python reference, generators, and conformance vectors |
| `fl-workload/` | MNIST workload, divergence experiments, campaigns, and retained run records |
| `chain/contracts/` | Solidity sources and compiled contract artifacts |
| `chain/geth/` | Modified go-ethereum client with EVM entry points and mandatory aggregation integration |
| `chain/testbed/` | Isolated-process testbed drivers and retained execution records |
| `evidence/` | Platform-specific and protocol-boundary evidence |
| `artifact/` | Portable artifact runner |

The keys in `chain/testbed/keys.json` are deterministic, public test keys for
isolated development chains. Never use them on a funded or public network.

## Scope

The artifact evaluates numerical divergence, deterministic integer
aggregation, execution integration, and the reported performance and learning
measurements. It does not reproduce a live heterogeneous proof-of-stake network
or measure a multi-epoch finality trajectory. See [ARTIFACTS.md](ARTIFACTS.md)
for the precise evidence boundaries.

## Upstream software

`chain/geth` is derived from go-ethereum and retains its upstream copyright and
license files (`COPYING` and `COPYING.LESSER`). The imported base revision and
the local integration scope are recorded in
[`chain/geth/DFRL_ORIGIN.md`](chain/geth/DFRL_ORIGIN.md).
