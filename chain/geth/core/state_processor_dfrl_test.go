package core

import (
	"math/big"
	"testing"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/rawdb"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/tracing"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/params"
	"github.com/ethereum/go-ethereum/triedb"
	"github.com/holiman/uint256"
)

// The property under test is the one the liveness argument rests on: at an
// aggregation boundary the aggregation executes as part of block processing,
// with no transaction carrying it and therefore no proposer able to omit it,
// and its state writes land in the block's post-state root.
//
// The contract deployed at params.DFRLAggregatorAddress is deliberately
// trivial: it stores the round index it was called with. That is enough to
// answer the question -- did the call happen, and did it change the state --
// without entangling the test in aggregation arithmetic, which
// dfrl-agg's own tests already cover.
//
//	calldata[0:32] = round
//	SSTORE(0, calldata[0:32])
//
// PUSH0 CALLDATALOAD PUSH0 SSTORE STOP  =>  5f 35 5f 55 00
var dfrlProbeCode = []byte{0x5f, 0x35, 0x5f, 0x55, 0x00}

// dfrlProbeCodeOffByOne stands in for a SECOND, conforming-but-different
// aggregation implementation: given identical calldata it commits a value one
// unit away, the shape a one-quantization-step numeric divergence takes.
//
// PUSH0 CALLDATALOAD PUSH1 1 ADD PUSH0 SSTORE STOP => 5f 35 60 01 01 5f 55 00
var dfrlProbeCodeOffByOne = []byte{0x5f, 0x35, 0x60, 0x01, 0x01, 0x5f, 0x55, 0x00}

func dfrlTestConfig(interval uint64) *params.ChainConfig {
	zero := uint64(0)
	cfg := *params.MergedTestChainConfig
	cfg.DFRLTime = &zero
	cfg.DFRLAggregationInterval = interval
	return &cfg
}

// runPreExecutionAt processes the pre-execution system calls for a block at the
// given height and returns the probe contract's stored word afterwards.
func runPreExecutionAt(t *testing.T, cfg *params.ChainConfig, height uint64, probe []byte) (stored common.Hash, root common.Hash) {
	t.Helper()
	db := rawdb.NewMemoryDatabase()
	tdb := triedb.NewDatabase(db, triedb.HashDefaults)
	sdb, err := state.New(types.EmptyRootHash, state.NewDatabase(tdb, nil))
	if err != nil {
		t.Fatal(err)
	}
	if probe != nil {
		sdb.CreateAccount(params.DFRLAggregatorAddress)
		sdb.SetCode(params.DFRLAggregatorAddress, probe, tracing.CodeChangeGenesis)
		sdb.AddBalance(params.DFRLAggregatorAddress, uint256.NewInt(0), tracing.BalanceChangeUnspecified)
	}

	header := &types.Header{
		Number:     new(big.Int).SetUint64(height),
		Time:       1000,
		Difficulty: common.Big0,
		GasLimit:   30_000_000,
		BaseFee:    big.NewInt(0),
	}
	parent := &types.Header{
		Number:     new(big.Int).SetUint64(height - 1),
		Time:       999,
		Difficulty: common.Big0,
		GasLimit:   30_000_000,
		BaseFee:    big.NewInt(0),
	}
	blockCtx := vm.BlockContext{
		CanTransfer: CanTransfer,
		Transfer:    Transfer,
		GetHash:     func(uint64) common.Hash { return common.Hash{} },
		Coinbase:    common.Address{},
		BlockNumber: new(big.Int).Set(header.Number),
		Time:        header.Time,
		Difficulty:  common.Big0,
		BaseFee:     header.BaseFee,
		GasLimit:    header.GasLimit,
		Random:      &common.Hash{},
	}
	evm := vm.NewEVM(blockCtx, sdb, cfg, vm.Config{})
	defer evm.Release()

	// No beacon root and no transactions: nothing but the scheduled system calls.
	PreExecution(t.Context(), nil, parent, cfg, evm, header.Number, header.Time)

	stored = sdb.GetState(params.DFRLAggregatorAddress, common.Hash{})
	root, err = sdb.Commit(cfg.Rules(new(big.Int).SetUint64(height), true, header.Time), height)
	if err != nil {
		t.Fatal(err)
	}
	return stored, root
}

// TestDFRLAggregationIsMandatory establishes that at a boundary height the
// aggregation call runs with no transaction present, that it does not run off
// the boundary, and that it does not run at all when the interval is unset --
// so the earlier, omittable behavior is preserved for existing networks.
func TestDFRLAggregationIsMandatory(t *testing.T) {
	const k = 32

	t.Run("runs at a boundary with no transactions", func(t *testing.T) {
		stored, _ := runPreExecutionAt(t, dfrlTestConfig(k), 2*k, dfrlProbeCode)
		want := common.BigToHash(big.NewInt(2*k/k - 1)) // round index 1
		if stored != want {
			t.Fatalf("boundary height %d: probe stored %v, want round index %v", 2*k, stored, want)
		}
		t.Logf("height %d: aggregation executed with zero transactions in the block, round index %v",
			2*k, stored.Big())
	})

	t.Run("does not run off a boundary", func(t *testing.T) {
		stored, _ := runPreExecutionAt(t, dfrlTestConfig(k), 2*k+1, dfrlProbeCode)
		if stored != (common.Hash{}) {
			t.Fatalf("non-boundary height %d: probe stored %v, want untouched", 2*k+1, stored)
		}
	})

	t.Run("does not run when the interval is unset", func(t *testing.T) {
		stored, _ := runPreExecutionAt(t, dfrlTestConfig(0), 2*k, dfrlProbeCode)
		if stored != (common.Hash{}) {
			t.Fatalf("interval 0: probe stored %v, want untouched (omittable contract-call behavior)", stored)
		}
	})

	t.Run("does not run before the DFRL fork", func(t *testing.T) {
		cfg := dfrlTestConfig(k)
		future := uint64(1 << 40)
		cfg.DFRLTime = &future
		stored, _ := runPreExecutionAt(t, cfg, 2*k, dfrlProbeCode)
		if stored != (common.Hash{}) {
			t.Fatalf("pre-fork: probe stored %v, want untouched", stored)
		}
	})
}

// TestDFRLAggregationReachesStateRoot is the step the liveness argument needs:
// whatever the mandatory aggregation writes is committed to the block's
// post-state root. Two nodes whose aggregation produces different bytes
// therefore produce different state roots for the same parent and the same
// (empty) transaction list, which is what a receiving validator recomputes and
// compares during payload validation.
func TestDFRLAggregationReachesStateRoot(t *testing.T) {
	const k = 32

	_, rootWith := runPreExecutionAt(t, dfrlTestConfig(k), 2*k, dfrlProbeCode)
	_, rootWithout := runPreExecutionAt(t, dfrlTestConfig(0), 2*k, dfrlProbeCode)
	if rootWith == rootWithout {
		t.Fatal("mandatory aggregation left the post-state root unchanged; the liveness argument needs it to reach the root")
	}
	t.Logf("post-state root with mandatory aggregation    %v", rootWith)
	t.Logf("post-state root without it (same parent, same empty tx list) %v", rootWithout)

	// The step the argument actually needs: two implementations that receive
	// IDENTICAL calldata at the SAME height with the same parent and the same
	// (empty) transaction list, but commit values one unit apart, produce
	// different post-state roots. That is precisely what a receiving validator
	// recomputes and compares, and it is where engine_newPayload returns
	// INVALID. The divergence is injected here by bytecode rather than by float
	// arithmetic -- dfrl-agg's own tests establish the arithmetic -- so that
	// this test isolates the block-processing property.
	_, rootImplA := runPreExecutionAt(t, dfrlTestConfig(k), 2*k, dfrlProbeCode)
	_, rootImplB := runPreExecutionAt(t, dfrlTestConfig(k), 2*k, dfrlProbeCodeOffByOne)
	if rootImplA == rootImplB {
		t.Fatal("two implementations differing by one unit produced the same post-state root")
	}
	t.Logf("same height, same parent, no transactions, results one unit apart:")
	t.Logf("    implementation A post-state root %v", rootImplA)
	t.Logf("    implementation B post-state root %v", rootImplB)
}
