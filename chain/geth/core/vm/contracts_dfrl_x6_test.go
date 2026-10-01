// Copyright 2026 The DT-DFRL authors.

package vm_test

import (
	"bytes"
	"testing"

	dfrlagg "dfrl-agg"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/core/vm/runtime"
)

// TestX6MultiValidatorStateRootAgreement is a scoped, in-process version of
// X6 (REWRITE_PLAN.md Phase 3c): "N validators, many rounds, submitted bytes
// 100% identical."
//
// Scope, stated plainly so this is not mistaken for the full X6 experiment:
//   - This runs K INDEPENDENT single-shot EVM executions per round, each
//     against its own fresh, ephemeral StateDB (via core/vm/runtime.Execute),
//     not K long-running node processes connected by Clique consensus. There
//     is no p2p, no block production, no persistent per-validator chain
//     across rounds.
//   - What it DOES faithfully test: given byte-identical calldata (the same
//     client submissions), does an independent EVM execution -- contract
//     creation, precompile dispatch through the real jump table and gas
//     accounting, a KECCAK256 over the result, and an SSTORE of that hash --
//     produce a byte-identical resulting account/storage trie root on K
//     separate invocations? If any layer between "bytes in" and "committed
//     state" introduced nondeterminism (map iteration order in gas
//     accounting, trie construction, hashing), THIS is where it would show up
//     even though the dfrl-agg-level tests already show the arithmetic itself
//     is order-independent.
//   - A full multi-node Clique testbed (real p2p, real block production, real
//     cross-process timing) is Phase 3c work and is explicitly NOT attempted
//     here.
//
// The bytecode under test: STATICCALL the DFRL precompile with the calldata
// forwarded verbatim, KECCAK256 the return data, SSTORE it at slot 0, and also
// return it -- so both the call's direct return value and its effect on
// committed state are checked.
func TestX6MultiValidatorStateRootAgreement(t *testing.T) {
	const validators = 5 // matches REWRITE_PLAN's X6 target of >= 5
	const rounds = 20    // matches REWRITE_PLAN's X6 target of >= 100 divided
	// down for a single-process in-test run; see scope note above.

	code := buildCommitForwarder(vm.DFRLFedAvgAddress)
	cfg := dfrlConfig()

	var mismatches int
	for r := 0; r < rounds; r++ {
		weights, tensors := roundInputs(t, uint64(20260824+r), 10, 2048)
		input := encodeDFRLInput(t, weights, tensors)

		var roots [][32]byte
		var rets [][]byte
		for v := 0; v < validators; v++ {
			ret, state, err := runtime.Execute(code, input, &runtime.Config{
				ChainConfig: cfg,
				GasLimit:    50_000_000,
			})
			if err != nil {
				t.Fatalf("round %d validator %d: EVM execution failed: %v", r, v, err)
			}
			rules := cfg.Rules(nil, false, 0)
			root := state.IntermediateRoot(rules)
			roots = append(roots, root)
			rets = append(rets, ret)
		}

		for v := 1; v < validators; v++ {
			if roots[v] != roots[0] {
				mismatches++
				t.Errorf("round %d: validator %d state root %x != validator 0 root %x",
					r, v, roots[v], roots[0])
			}
			if !bytes.Equal(rets[v], rets[0]) {
				t.Errorf("round %d: validator %d return value differs from validator 0", r, v)
			}
		}
	}
	if mismatches == 0 {
		t.Logf("%d rounds x %d validators: 0 state-root mismatches", rounds, validators)
	}
}

// roundInputs generates one round's client submissions, reusing dfrl-agg's
// portable generator so this test's inputs are reproducible the same way
// fl-workload's are.
func roundInputs(t *testing.T, seed uint64, n, p int) ([]int64, []dfrlagg.Tensor) {
	t.Helper()
	ups := dfrlagg.GenUpdates(seed, n, p, dfrlagg.PaperWeights, 1<<17, dfrlagg.DefaultF)
	weights := make([]int64, n)
	tensors := make([]dfrlagg.Tensor, n)
	for i, u := range ups {
		weights[i] = u.Weight
		tensors[i] = u.Tensor
	}
	return weights, tensors
}

// buildCommitForwarder extends buildStaticcallForwarder: after the STATICCALL
// returns, it hashes the return data and SSTOREs the hash at slot 0 before
// returning it, so the call has an effect on committed state that
// IntermediateRoot can actually see (a bare STATICCALL with no storage write
// would leave global state unchanged regardless of whether determinism holds,
// making a root comparison vacuous).
func buildCommitForwarder(addr [20]byte) []byte {
	var b []byte
	push := func(v byte) { b = append(b, byte(vm.PUSH1), v) }
	op := func(o vm.OpCode) { b = append(b, byte(o)) }

	op(vm.CALLDATASIZE)
	push(0)
	push(0)
	op(vm.CALLDATACOPY)

	push(0)
	push(0)
	op(vm.CALLDATASIZE)
	push(0)
	b = append(b, byte(vm.PUSH20))
	b = append(b, addr[:]...)
	op(vm.GAS)
	op(vm.STATICCALL)
	op(vm.POP)

	// Copy return data to memory, hash it, store the hash at slot 0.
	op(vm.RETURNDATASIZE)
	push(0)
	push(0)
	op(vm.RETURNDATACOPY)
	op(vm.RETURNDATASIZE) // size
	push(0)               // offset
	op(vm.KECCAK256)      // -> hash on stack
	op(vm.DUP1)
	push(0)
	op(vm.SSTORE) // SSTORE(0, hash); DUP1 left a copy of hash on the stack

	// Return the hash: MSTORE it at mem[0] then RETURN 32 bytes from mem[0].
	push(0)
	op(vm.MSTORE)
	push(32)
	push(0)
	op(vm.RETURN)
	return b
}
