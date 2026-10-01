// Copyright 2026 The DT-DFRL authors.

package vm_test

import (
	"encoding/csv"
	"math/big"
	"os"
	"strconv"
	"testing"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/core/vm/runtime"
	"github.com/holiman/uint256"

	dfrlagg "dfrl-agg"
)

// TestX8ContractGasSurface completes X8 (REWRITE_PLAN Phase 3c checklist):
// the precompile-layer (N,|Theta|) gas grid already existed from Phase 3a;
// this is the missing CONTRACT-layer grid, covering both the commit-only and
// full-storage variants across a full N x |Theta| sweep, so the surface can
// be fit and plotted the same way fit_gas_model.py already does for the
// precompile. Unlike the X9 latency comparison, gas is a DETERMINISTIC
// function of the input under a fixed chain config -- there is no
// measurement noise here and therefore no need for repeated trials or a CI;
// one run per grid point is the exact, reproducible answer.
func TestX8ContractGasSurface(t *testing.T) {
	if testing.Short() {
		t.Skip("X8 gas surface sweep skipped in -short mode")
	}
	commitArt := loadCompiledContract(t, "DFRLAggregatorCommitOnly.json", "DFRLAggregatorCommitOnly")
	fullArt := loadCompiledContract(t, "DFRLAggregator.json", "DFRLAggregator")
	cfg := dfrlConfig()

	run := func(art solcArtifact, n, p int) uint64 {
		statedb, _ := state.New(common.Hash{}, state.NewDatabaseForTesting())
		contractAddr := common.HexToAddress("0x00000000000000000000000000000000c0ffee")
		statedb.CreateAccount(contractAddr)
		statedb.SetCode(contractAddr, art.BinRuntime, 0)

		weights := make([]int64, n)
		for i := range weights {
			weights[i] = int64(i + 1)
		}
		_, tensors := genUpdates(t, 8080, n, p, weights, 1<<14)
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
				ChainConfig: cfg, Origin: sender, State: statedb, GasLimit: 1_600_000_000,
			}); err != nil {
				t.Fatalf("uploadModel(n=%d,p=%d): %v", n, p, err)
			}
		}
		aggInput, err := art.ABI.Pack("aggregateModel", round)
		if err != nil {
			t.Fatal(err)
		}
		caller := common.BytesToAddress([]byte{0x99})
		statedb.CreateAccount(caller)
		const limit = 1_600_000_000
		_, remaining, err := runtime.Call(contractAddr, aggInput, &runtime.Config{
			ChainConfig: cfg, Origin: caller, State: statedb, GasLimit: limit,
		})
		if err != nil {
			t.Fatalf("aggregateModel(n=%d,p=%d): %v", n, p, err)
		}
		return limit - remaining
	}

	pcGas := func(n, p int) uint64 {
		weights := make([]int64, n)
		for i := range weights {
			weights[i] = int64(i + 1)
		}
		_, tensors := genUpdates(t, 8080, n, p, weights, 1<<14)
		input := encodeDFRLInput(t, weights, tensors)
		return vm.PrecompiledContractsDFRL[vm.DFRLFedAvgAddress].RequiredGas(input)
	}

	ns := []int{1, 2, 5, 10, 20}
	ps := []int{16, 256, 1024, 4096, 16384}

	f, err := os.Create("x8_grid.csv")
	if err != nil {
		t.Fatalf("create x8_grid.csv: %v", err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	if err := w.Write([]string{"n", "p", "commit_gas", "full_gas", "precompile_estimate"}); err != nil {
		t.Fatal(err)
	}

	for _, n := range ns {
		for _, p := range ps {
			commitGas := run(commitArt, n, p)
			fullGas := run(fullArt, n, p)
			pcEst := pcGas(n, p)
			t.Logf("n=%-3d p=%-6d commit=%-12d full=%-12d precompile_est=%-8d full/commit=%.3fx",
				n, p, commitGas, fullGas, pcEst, float64(fullGas)/float64(commitGas))
			if err := w.Write([]string{
				strconv.Itoa(n), strconv.Itoa(p),
				strconv.FormatUint(commitGas, 10), strconv.FormatUint(fullGas, 10),
				strconv.FormatUint(pcEst, 10),
			}); err != nil {
				t.Fatal(err)
			}
			if fullGas <= commitGas {
				t.Errorf("n=%d p=%d: expected full-storage gas > commit-only gas; got full=%d commit=%d", n, p, fullGas, commitGas)
			}
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		t.Fatalf("flush x8_grid.csv: %v", err)
	}
	t.Logf("wrote %d grid points to x8_grid.csv", len(ns)*len(ps))
}
