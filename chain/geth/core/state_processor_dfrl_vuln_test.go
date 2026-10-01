//go:build e3lite_vulnerable

package core

import (
	"math/big"
	"os"
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

// TestMandatoryVulnAggPostStateRoot runs the full mechanism the liveness
// argument needs, with no synthetic stand-ins: the height-scheduled system call
// invokes the real MandatoryVulnAgg contract, which STATICCALLs the real
// floating-point aggregation precompile at 0x0103, and the resulting commitment
// is committed to the block's post-state root.
//
// The test itself cannot compare two summation orders, because a single test
// binary contains only the one its build tags selected. It therefore PRINTS the
// post-state root and the stored commitment, and the comparison is made across
// two runs:
//
//	go test -tags e3lite_vulnerable                     -run TestMandatoryVulnAggPostStateRoot -v ./core/
//	go test -tags "e3lite_vulnerable e3lite_reverse"    -run TestMandatoryVulnAggPostStateRoot -v ./core/
//
// Two binaries, one machine, one architecture, byte-identical genesis state and
// byte-identical contract code, no transactions in the block: if the roots
// differ, the only remaining difference between them is the accumulation order
// compiled into the precompile.
//
// Set DFRL_VULN_ROOT_OUT to a path to have the root written there, so a script
// can diff the two runs without parsing test output.
func TestMandatoryVulnAggPostStateRoot(t *testing.T) {
	const k = 32

	runtimeCode := common.FromHex(mandatoryVulnAggRuntime)
	if len(runtimeCode) == 0 {
		t.Fatal("MandatoryVulnAgg runtime bytecode missing")
	}

	db := rawdb.NewMemoryDatabase()
	tdb := triedb.NewDatabase(db, triedb.HashDefaults)
	sdb, err := state.New(types.EmptyRootHash, state.NewDatabase(tdb, nil))
	if err != nil {
		t.Fatal(err)
	}
	// Genesis state: the aggregation target contract, nothing else. Identical
	// in both builds.
	sdb.CreateAccount(params.DFRLAggregatorAddress)
	sdb.SetCode(params.DFRLAggregatorAddress, runtimeCode, tracing.CodeChangeGenesis)
	sdb.AddBalance(params.DFRLAggregatorAddress, uint256.NewInt(0), tracing.BalanceChangeUnspecified)

	zero := uint64(0)
	cfg := *params.MergedTestChainConfig
	cfg.DFRLTime = &zero
	cfg.DFRLAggregationInterval = k

	height := uint64(2 * k)
	header := &types.Header{
		Number: new(big.Int).SetUint64(height), Time: 1000,
		Difficulty: common.Big0, GasLimit: 30_000_000, BaseFee: big.NewInt(0),
	}
	parent := &types.Header{
		Number: new(big.Int).SetUint64(height - 1), Time: 999,
		Difficulty: common.Big0, GasLimit: 30_000_000, BaseFee: big.NewInt(0),
	}
	blockCtx := vm.BlockContext{
		CanTransfer: CanTransfer, Transfer: Transfer,
		GetHash:     func(uint64) common.Hash { return common.Hash{} },
		BlockNumber: new(big.Int).Set(header.Number), Time: header.Time,
		Difficulty: common.Big0, BaseFee: header.BaseFee,
		GasLimit: header.GasLimit, Random: &common.Hash{},
	}
	evm := vm.NewEVM(blockCtx, sdb, &cfg, vm.Config{})
	defer evm.Release()

	PreExecution(t.Context(), nil, parent, &cfg, evm, header.Number, header.Time)

	commitment := sdb.GetState(params.DFRLAggregatorAddress, common.Hash{})
	round := sdb.GetState(params.DFRLAggregatorAddress, common.BigToHash(big.NewInt(1)))
	root, err := sdb.Commit(cfg.Rules(header.Number, true, header.Time), height)
	if err != nil {
		t.Fatal(err)
	}

	if commitment == (common.Hash{}) {
		t.Fatal("aggregation produced no commitment: the system call did not reach the precompile")
	}
	t.Logf("accumulation order compiled in: %s", vm.DFRLVulnOrderName())
	t.Logf("height %d, empty transaction list", height)
	t.Logf("  aggregate commitment  %v", commitment)
	t.Logf("  round index stored    %v", round.Big())
	t.Logf("  POST-STATE ROOT       %v", root)

	if out := os.Getenv("DFRL_VULN_ROOT_OUT"); out != "" {
		line := vm.DFRLVulnOrderName() + " " + root.Hex() + " " + commitment.Hex() + "\n"
		if err := os.WriteFile(out, []byte(line), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", out)
	}
}
