//go:build e3lite_vulnerable

// Copyright 2026 The DT-DFRL authors.

package catalyst

import (
	"os"
	"testing"

	"github.com/ethereum/go-ethereum/beacon/engine"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/params"
)

// Two conforming nodes, each building its own payload, exchanging the raw
// bytes, and rejecting the peer's.
//
// TestDFRLDivergentStateRootIsINVALID already shows that a payload carrying a
// state root this node did not compute is rejected, but it gets that root by
// substitution: the root is a constant measured in package core, written into
// an otherwise locally built payload. That is enough for "the Engine API
// rejects a divergent root" and not enough for "two conforming nodes reject
// each other", which is the step the manuscript's level chain still needs.
//
// The gap matters because the two claims fail differently. Substitution proves
// the validator's re-execution check works. It does not prove that two
// independently built clients, over byte-identical inputs, actually produce
// payloads the other refuses -- the aggregate divergence might never reach the
// state root a payload commits to, or might be normalised away before the
// header is sealed.
//
// This test closes it without any substitution. Each process builds a payload
// through the ordinary payload-building path, writes the envelope verbatim, and
// judges the envelope the other process wrote. Nothing is edited between the
// two, and no header field is recomputed by the test.
//
// Two nodes, two processes, because the accumulation order is a build tag and
// not a runtime flag (see core/vm/contracts_dfrl_vulnerable_order_fwd.go for
// why that choice is deliberate). The protocol is:
//
//	go test -tags e3lite_vulnerable -run TestDFRLTwoNodePayloadExchange ./eth/catalyst/
//	  with DFRL_PAYLOAD_OUT=fwd.json
//	go test -tags "e3lite_vulnerable e3lite_reverse" -run ... ./eth/catalyst/
//	  with DFRL_PAYLOAD_OUT=rev.json DFRL_PAYLOAD_IN=fwd.json
//	go test -tags e3lite_vulnerable -run ... ./eth/catalyst/
//	  with DFRL_PAYLOAD_IN=rev.json
//
// Run 2 and run 3 each assert that the peer's payload is INVALID, giving the
// rejection in both directions.
//
// Scope. The submissions being aggregated are a constant in the deployed
// contract (chain/contracts/MandatoryVulnAgg.sol), not values clients sent
// through a submission entry point. For this experiment that is a feature: the
// two nodes provably consume the same input because they run the same contract
// bytes over the same genesis, so "did they receive the same submissions" is
// not a question. It does mean this test says nothing about the client
// submission path, round membership locking, or late submissions.

// dfrlExchangeAggRuntime is the deployed runtime bytecode of
// chain/contracts/MandatoryVulnAgg.sol, the same constant package core's
// TestMandatoryVulnAggPostStateRoot uses (core/state_processor_dfrl_vuln_data_test.go).
// Duplicated rather than exported so that both builds provably execute the same
// contract bytes, leaving the compiled accumulation order as the only
// difference between the two processes.
const dfrlExchangeAggRuntime = "0x608060405234801561000f575f5ffd5b5060043610610034575f3560e01c806327ff01e91461011457806382bc07e61461012e575b5f6040518060600160405280603481526020016101656034913990505f5f6101036001600160a01b03168360405161006c9190610137565b5f60405180830381855afa9150503d805f81146100a4576040519150601f19603f3d011682016040523d82523d5f602084013e6100a9565b606091505b5091509150816100f15760405162461bcd60e51b815260206004820152600f60248201526e1d9d5b1b881859d9c819985a5b1959608a1b604482015260640160405180910390fd5b80516020808301919091205f5536106101125761010e365f61014d565b6001555b005b61011c5f5481565b60405190815260200160405180910390f35b61011c60015481565b5f82518060208501845e5f920191825250919050565b5f6020828403121561015d575f5ffd5b503591905056fe000000030000000000001d1700000001bc13f80000000000000007d9000000013c3640000000000000001dcc000000013bfc6800a26469706673582212200bda393f1e27a17b0d02cbde45efd728535f4c9d537304fc8a29e1c8d8b4d98564736f6c63430008240033"

// dfrlExchangeGenesis is identical in both builds: the aggregation target
// contract, the system contracts the post-Shanghai forks call on every block,
// one funded account, and mandatory aggregation at every height.
func dfrlExchangeGenesis() *core.Genesis {
	config := *params.MergedTestChainConfig
	zero := uint64(0)
	config.DFRLTime = &zero
	// Every block h>0 is an aggregation boundary, so the first PoS block built
	// over genesis already carries an aggregation. Avoids a long pre-chain
	// whose blocks would have to match across the two processes as well.
	config.DFRLAggregationInterval = 1

	return &core.Genesis{
		Config: &config,
		Alloc: types.GenesisAlloc{
			testAddr: {Balance: testBalance},
			params.DFRLAggregatorAddress: {
				Code:    common.FromHex(dfrlExchangeAggRuntime),
				Balance: common.Big0,
			},
			params.BeaconRootsAddress:          {Nonce: 1, Code: common.FromHex("0x3373fffffffffffffffffffffffffffffffffffffffe14604457602036146024575f5ffd5b620180005f350680545f35146037575f5ffd5b6201800001545f5260205ff35b6201800042064281555f359062018000015500"), Balance: common.Big0},
			params.HistoryStorageAddress:       {Nonce: 1, Code: params.HistoryStorageCode, Balance: common.Big0},
			params.WithdrawalQueueAddress:      {Nonce: 1, Code: params.WithdrawalQueueCode, Balance: common.Big0},
			params.ConsolidationQueueAddress:   {Nonce: 1, Code: params.ConsolidationQueueCode, Balance: common.Big0},
			params.BuilderDepositAddress:       {Nonce: 1, Code: params.BuilderDepositCode, Balance: common.Big0},
			params.BuilderExitAddress:          {Nonce: 1, Code: params.BuilderExitCode, Balance: common.Big0},
			params.DeterministicFactoryAddress: {Nonce: 1, Code: params.DeterministicFactoryCode, Balance: common.Big0},
		},
		ExtraData:  []byte("dfrl exchange genesis"),
		Timestamp:  9000,
		Difficulty: common.Big0,
	}
}

func TestDFRLTwoNodePayloadExchange(t *testing.T) {
	outPath := os.Getenv("DFRL_PAYLOAD_OUT")
	inPath := os.Getenv("DFRL_PAYLOAD_IN")
	if outPath == "" && inPath == "" {
		t.Skip("set DFRL_PAYLOAD_OUT and/or DFRL_PAYLOAD_IN; see this file's header for the three-run protocol")
	}

	genesis := dfrlExchangeGenesis()
	n, ethservice := startEthService(t, genesis, nil)
	defer n.Close()
	api := newConsensusAPIWithoutHeartbeat(ethservice)

	parent := ethservice.BlockChain().CurrentBlock()
	t.Logf("compiled accumulation order: %s", vm.DFRLVulnOrderName())
	t.Logf("genesis hash %v; building over parent %d (%v)", genesis.ToBlock().Hash(), parent.Number, parent.Hash())

	// Build our own payload through the ordinary path. No field is touched
	// afterwards.
	own := getNewEnvelope(t, api, parent, []*types.Withdrawal{}, &dfrlExchangeBeaconRoot)
	t.Logf("own payload: number %d state root %v block hash %v",
		own.ExecutionPayload.Number, own.ExecutionPayload.StateRoot, own.ExecutionPayload.BlockHash)

	// Control: this node accepts what it built. If this fails the experiment
	// says nothing about the peer's payload.
	if st := submitPayload(t, api, own); st.Status != engine.VALID {
		t.Fatalf("control: own payload should be VALID, got %v (%v)", st.Status, errString(st.ValidationError))
	}
	t.Log("control: own payload is VALID on its own builder")

	if outPath != "" {
		writeEnvelope(t, outPath, own)
	}
	if inPath == "" {
		return
	}
	judgePeerEnvelope(t, api, own, readEnvelope(t, inPath), false)
}
