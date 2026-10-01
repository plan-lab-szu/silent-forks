// Copyright 2026 The DT-DFRL authors.

package vm_test

import (
	"math/big"
	"testing"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/core/vm/runtime"
	"github.com/holiman/uint256"

	dfrlagg "dfrl-agg"
)

// TestDiagnoseAggregateModelGasBreakdown isolates where aggregateModel's gas
// actually goes, because a prior claim in this project's docs ("O(n^2)
// encodePacked concatenation is ~98% of the cost") was an UNVERIFIED estimate
// that turned out to be wrong once the fix was measured (see HANDBOOK.md
// correction). This test does not assert anything -- it logs a breakdown so
// the real cost driver can be identified from evidence instead of guessed
// again.
func TestDiagnoseAggregateModelGasBreakdown(t *testing.T) {
	art := loadCompiledContract(t, "DFRLAggregatorCommitOnly.json", "DFRLAggregatorCommitOnly")
	cfg := dfrlConfig()

	run := func(n, p int) (aggGas uint64) {
		statedb, _ := state.New(common.Hash{}, state.NewDatabaseForTesting())
		contractAddr := common.HexToAddress("0x00000000000000000000000000000000c0ffee")
		statedb.CreateAccount(contractAddr)
		statedb.SetCode(contractAddr, art.BinRuntime, 0)

		weights := make([]int64, n)
		for i := range weights {
			weights[i] = int64(i + 1)
		}
		_, tensors := genUpdates(t, 555, n, p, weights, 1<<14)
		round := big.NewInt(0)
		for i := 0; i < n; i++ {
			sender := common.BytesToAddress([]byte{byte(0x10 + i)})
			statedb.CreateAccount(sender)
			statedb.AddBalance(sender, uint256.NewInt(1_000_000_000_000_000_000), 0)
			enc, err := dfrlagg.Encode(tensors[i])
			if err != nil {
				t.Fatal(err)
			}
			input, err := art.ABI.Pack("uploadModel", round, uint64(weights[i]), enc)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := runtime.Call(contractAddr, input, &runtime.Config{
				ChainConfig: cfg, Origin: sender, State: statedb, GasLimit: 800_000_000,
			}); err != nil {
				t.Fatalf("uploadModel: %v", err)
			}
		}
		aggInput, err := art.ABI.Pack("aggregateModel", round)
		if err != nil {
			t.Fatal(err)
		}
		caller := common.BytesToAddress([]byte{0x99})
		statedb.CreateAccount(caller)
		const limit = 800_000_000
		_, remaining, err := runtime.Call(contractAddr, aggInput, &runtime.Config{
			ChainConfig: cfg, Origin: caller, State: statedb, GasLimit: limit,
		})
		if err != nil {
			t.Fatalf("aggregateModel: %v", err)
		}
		return limit - remaining
	}

	pcGas := func(n, p int) uint64 {
		weights := make([]int64, n)
		for i := range weights {
			weights[i] = int64(i + 1)
		}
		_, tensors := genUpdates(t, 555, n, p, weights, 1<<14)
		input := encodeDFRLInput(t, weights, tensors)
		return vm.PrecompiledContractsDFRL[vm.DFRLFedAvgAddress].RequiredGas(input)
	}

	t.Log("=== fixed N=1, vary P (isolates per-tensor storage-read cost) ===")
	for _, p := range []int{16, 256, 1024, 4096, 16384} {
		g := run(1, p)
		t.Logf("N=1 P=%-6d aggregateModel=%-10d precompile-estimate=%-8d ratio=%.1fx",
			p, g, pcGas(1, p), float64(g)/float64(pcGas(1, p)))
	}

	t.Log("=== fixed P=4096, vary N (isolates per-entry overhead) ===")
	for _, n := range []int{1, 2, 5, 10} {
		g := run(n, 4096)
		t.Logf("N=%-3d P=4096  aggregateModel=%-10d precompile-estimate=%-8d ratio=%.1fx",
			n, g, pcGas(n, 4096), float64(g)/float64(pcGas(n, 4096)))
	}
}
