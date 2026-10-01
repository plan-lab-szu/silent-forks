// Copyright 2026 The DT-DFRL authors.

package vm_test

import (
	"bytes"
	"math/big"
	"testing"

	dfrlagg "dfrl-agg"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/vm/runtime"
	"github.com/holiman/uint256"
)

// TestDFRLRobustAggregatorContract is TestDFRLAggregatorContract's
// counterpart for Phase 4/U3: deploys DFRLRobustAggregator.sol, has several
// accounts upload updates (two of them adversarial outliers), calls
// aggregateModel (STATICCALLs the RobustAgg precompile at 0x0102), and checks
// the on-chain result against a direct dfrl-agg.TrimmedMean call -- the same
// bit-for-bit standard TestDFRLAggregatorContract holds FedAvg to, extended
// to the second primitive.
func TestDFRLRobustAggregatorContract(t *testing.T) {
	art := loadCompiledContract(t, "DFRLRobustAggregator.json", "DFRLRobustAggregator")
	cfg := dfrlConfig()
	statedb, _ := state.New(common.Hash{}, state.NewDatabaseForTesting())

	contractAddr := common.HexToAddress("0x00000000000000000000000000000000c0ffee")
	statedb.CreateAccount(contractAddr)
	statedb.SetCode(contractAddr, art.BinRuntime, 0)

	const n, p, k = 7, 256, 1
	weights := make([]int64, n)
	for i := range weights {
		weights[i] = int64(i + 1)
	}
	_, tensors := genUpdates(t, 4343, n, p, weights, 1<<14)
	// Two clients go maximally adversarial, mirroring dfrl-agg's
	// TestTrimmedMeanBoundsOutlierInfluence unit-level illustration, now at
	// the real contract/precompile layer instead of a synthetic Go call.
	tensors[1] = tensors[1].Clone()
	tensors[5] = tensors[5].Clone()
	for j := range tensors[1].Data {
		tensors[1].Data[j] = dfrlagg.ParamBound - 1
		tensors[5].Data[j] = -(dfrlagg.ParamBound - 1)
	}

	round := big.NewInt(0)
	senders := make([]common.Address, n)
	for i := 0; i < n; i++ {
		senders[i] = common.BytesToAddress([]byte{byte(0x10 + i)})
		statedb.CreateAccount(senders[i])
		statedb.AddBalance(senders[i], uint256.NewInt(1_000_000_000_000_000_000), 0)

		enc, err := dfrlagg.Encode(tensors[i])
		if err != nil {
			t.Fatal(err)
		}
		input, err := art.ABI.Pack("uploadModel", round, uint64(weights[i]), enc)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := runtime.Call(contractAddr, input, &runtime.Config{
			ChainConfig: cfg, Origin: senders[i], State: statedb, GasLimit: 5_000_000,
		}); err != nil {
			t.Fatalf("uploadModel from sender %d: %v", i, err)
		}
	}

	aggInput, err := art.ABI.Pack("aggregateModel", round, uint32(k))
	if err != nil {
		t.Fatal(err)
	}
	caller := common.BytesToAddress([]byte{0x99})
	statedb.CreateAccount(caller)
	out, _, err := runtime.Call(contractAddr, aggInput, &runtime.Config{
		ChainConfig: cfg, Origin: caller, State: statedb, GasLimit: 20_000_000,
	})
	if err != nil {
		t.Fatalf("aggregateModel: %v", err)
	}
	var onChainRoot [32]byte
	if err := art.ABI.UnpackIntoInterface(&onChainRoot, "aggregateModel", out); err != nil {
		t.Fatal(err)
	}

	dlInput, err := art.ABI.Pack("downloadModel", round)
	if err != nil {
		t.Fatal(err)
	}
	dlOut, _, err := runtime.Call(contractAddr, dlInput, &runtime.Config{
		ChainConfig: cfg, Origin: caller, State: statedb, GasLimit: 5_000_000,
	})
	if err != nil {
		t.Fatalf("downloadModel: %v", err)
	}
	var onChainResult []byte
	if err := art.ABI.UnpackIntoInterface(&onChainResult, "downloadModel", dlOut); err != nil {
		t.Fatal(err)
	}

	ups := mustUpdates(t, weights, tensors)
	want, err := dfrlagg.TrimmedMean(ups, k)
	if err != nil {
		t.Fatal(err)
	}
	wantEnc, err := dfrlagg.Encode(want)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(onChainResult, wantEnc) {
		t.Fatalf("on-chain RobustAgg result differs from direct dfrlagg.TrimmedMean call\n got=%x\nwant=%x", onChainResult, wantEnc)
	}
	t.Logf("contract-mediated trimmed-mean aggregation matches direct dfrl-agg call (n=%d, k=%d, root=%x)", n, k, onChainRoot)

	// Sanity: the adversarial pair really did get trimmed away, not just that
	// the arithmetic matched a reference that itself might not be robust.
	honest := make([]dfrlagg.Update, 0, n-2)
	for i, u := range ups {
		if i != 1 && i != 5 {
			honest = append(honest, u)
		}
	}
	clean, err := dfrlagg.TrimmedMean(honest, 0)
	if err != nil {
		t.Fatal(err)
	}
	fedavg, err := dfrlagg.FedAvg(ups)
	if err != nil {
		t.Fatal(err)
	}
	var devTrim, devAvg int64
	for j := range clean.Data {
		dt := want.Data[j] - clean.Data[j]
		da := fedavg.Data[j] - clean.Data[j]
		if dt < 0 {
			dt = -dt
		}
		if da < 0 {
			da = -da
		}
		devTrim += dt
		devAvg += da
	}
	if devTrim >= devAvg {
		t.Fatalf("on-chain trimmed mean did not bound outlier influence better than FedAvg would have: devTrim=%d devAvg=%d", devTrim, devAvg)
	}
	if devTrim == 0 {
		t.Logf("outlier deviation from clean-honest baseline: trimmed=0 (exact -- outliers landed exactly on the trimmed extremes), fedavg-would-be=%d", devAvg)
	} else {
		t.Logf("outlier deviation from clean-honest baseline: trimmed=%d, fedavg-would-be=%d (%.1fx worse)", devTrim, devAvg, float64(devAvg)/float64(devTrim))
	}
}
