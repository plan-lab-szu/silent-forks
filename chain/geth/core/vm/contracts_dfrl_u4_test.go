// Copyright 2026 The DT-DFRL authors.

package vm_test

import (
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"testing"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/vm/runtime"
	"github.com/holiman/uint256"

	dfrlagg "dfrl-agg"
)

// loadCompiledContract is loadDFRLAggregator generalized to an arbitrary
// solc --combined-json output file under contracts/, so
// DFRLAggregator.sol and DFRLAggregatorCommitOnly.sol can share one loader.
func loadCompiledContract(t *testing.T, filename, contractName string) solcArtifact {
	t.Helper()
	path := filepath.Join("..", "..", "..", "contracts", filename)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("compiled contract not found at %s (run solc first, see HANDBOOK.md): %v", path, err)
	}
	return parseSolcCombinedJSON(t, raw, contractName)
}

// TestU4StorageGasComparison measures, on IDENTICAL input, the gas cost of
// storing the full aggregated payload on chain (DFRLAggregator.sol) versus
// storing only a commitment (DFRLAggregatorCommitOnly.sol). This is R1-6c/U4's
// argument turned into a number: raw on-chain storage of a model does not
// scale, and here is by how much, at this specific parameter count.
//
// Scope: one (N, |Theta|) point, one run, on this development machine's gas
// schedule (which includes the provisional, disclosed dfrlGasBase/mu1/mu2
// constants from contracts_dfrl.go). The qualitative finding -- full storage
// costs meaningfully more, and the gap grows with |Theta| because SSTORE is
// charged per 32-byte word while the commitment is a fixed 32 bytes regardless
// of model size -- follows from the EVM's own gas schedule, not from anything
// specific to this benchmark run.
func TestU4StorageGasComparison(t *testing.T) {
	for _, p := range []int{256, 4096} {
		p := p
		t.Run(fmtP(p), func(t *testing.T) {
			u4Case(t, p)
		})
	}
}

func u4Case(t *testing.T, p int) {
	fullArt := loadCompiledContract(t, "DFRLAggregator.json", "DFRLAggregator")
	commitArt := loadCompiledContract(t, "DFRLAggregatorCommitOnly.json", "DFRLAggregatorCommitOnly")
	cfg := dfrlConfig()

	const n = 5
	weights := []int64{3, 5, 7, 11, 13}
	_, tensors := genUpdates(t, 9001, n, p, weights, 1<<14)

	run := func(art solcArtifact) uint64 {
		statedb, _ := state.New(common.Hash{}, state.NewDatabaseForTesting())
		contractAddr := common.HexToAddress("0x00000000000000000000000000000000c0ffee")
		statedb.CreateAccount(contractAddr)
		statedb.SetCode(contractAddr, art.BinRuntime, 0)

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
				ChainConfig: cfg, Origin: sender, State: statedb, GasLimit: 200_000_000,
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

	fullGas := run(fullArt)
	commitGas := run(commitArt)
	t.Logf("p=%d: aggregateModel gas -- full storage=%d, commit-only=%d, storage tax=%d (%.1f%% more)",
		p, fullGas, commitGas, fullGas-commitGas, 100*float64(fullGas-commitGas)/float64(commitGas))

	if fullGas <= commitGas {
		t.Errorf("expected full-storage variant to cost more gas than commit-only; got full=%d commit=%d", fullGas, commitGas)
	}
}

func fmtP(p int) string {
	return fmt.Sprintf("p=%d", p)
}
