//go:build e3lite_vulnerable

package catalyst

import (
	"math/big"
	"testing"

	"github.com/ethereum/go-ethereum/beacon/engine"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/params"
)

// TestDFRLOriginalExchangeDebug runs the same check with the original 3-client contract.
func TestDFRLOriginalExchangeDebug(t *testing.T) {
	genesis := dfrlExchangeGenesis()
	debugExchangeGenesis(t, genesis)
}

// TestDFRLNaturalExchangeDebug reads back the commitment the contract stored
// after the system call, to diagnose why fwd and rev produce identical state
// roots even though the precompile output differs.
func TestDFRLNaturalExchangeDebug(t *testing.T) {
	genesis := dfrlNatGenesis()
	debugExchangeGenesis(t, genesis)
}

func debugExchangeGenesis(t *testing.T, genesis *core.Genesis) {
	n, ethservice := startEthService(t, genesis, nil)
	defer n.Close()
	api := newConsensusAPIWithoutHeartbeat(ethservice)

	parent := ethservice.BlockChain().CurrentBlock()
	t.Logf("compiled accumulation order: %s", vm.DFRLVulnOrderName())
	t.Logf("genesis hash %v", genesis.ToBlock().Hash())
	t.Logf("building over parent %d (%v)", parent.Number, parent.Hash())

	own := getNewEnvelope(t, api, parent, nil, &dfrlExchangeBeaconRoot)
	t.Logf("own payload: number %d state root %v block hash %v",
		own.ExecutionPayload.Number, own.ExecutionPayload.StateRoot,
		own.ExecutionPayload.BlockHash)

	// Import and set as head.
	st := submitPayload(t, api, own)
	t.Logf("submit result: %v", st.Status)

	fcu := engine.ForkchoiceStateV1{
		HeadBlockHash: own.ExecutionPayload.BlockHash,
	}
	fcuResp, err := api.ForkchoiceUpdatedV3(t.Context(), fcu, nil)
	if err != nil {
		t.Fatalf("forkchoice update: %v", err)
	}
	t.Logf("forkchoice update: %v", fcuResp.PayloadStatus.Status)

	// Read the commitment from state. StateAt needs a header.
	block := ethservice.BlockChain().GetBlockByNumber(own.ExecutionPayload.Number)
	if block == nil {
		t.Fatal("block not found after import + forkchoice")
	}
	stateDB, err := ethservice.BlockChain().StateAt(block.Header())
	if err != nil {
		t.Fatalf("cannot open state: %v", err)
	}

	// slot 0 = lastCommitment, slot 1 = lastRound
	commitment := stateDB.GetState(params.DFRLAggregatorAddress, common.Hash{})
	round := stateDB.GetState(params.DFRLAggregatorAddress, common.BigToHash(big.NewInt(1)))
	codeSize := stateDB.GetCodeSize(params.DFRLAggregatorAddress)

	t.Logf("DFRLAggregatorAddress: %v", params.DFRLAggregatorAddress)
	t.Logf("  code size: %d bytes", codeSize)
	t.Logf("  slot 0 (lastCommitment): %v", commitment)
	t.Logf("  slot 1 (lastRound):      %v", round)

	if commitment == (common.Hash{}) {
		t.Error("ZERO COMMITMENT — the system call did not reach the precompile or reverted")
	}

	// Also check what precompiles are active.
	rules := genesis.Config.Rules(big.NewInt(1), true, 10000)
	t.Logf("rules.IsDFRL = %v", rules.IsDFRL)
	activeAddrs := vm.ActivePrecompiles(rules)
	t.Logf("active precompiles (%d):", len(activeAddrs))
	for _, addr := range activeAddrs {
		t.Logf("  %v", addr)
	}

	// Check if 0x0103 is in the set.
	found := false
	for _, addr := range activeAddrs {
		if addr == vm.DFRLVulnerableFloatAggAddress {
			found = true
			break
		}
	}
	t.Logf("0x0103 in active set: %v", found)
}
