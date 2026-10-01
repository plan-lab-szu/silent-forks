//go:build e3lite_vulnerable

// Copyright 2026 The DT-DFRL authors.

package catalyst

import (
	"math/big"
	"os"
	"testing"

	"github.com/ethereum/go-ethereum/beacon/engine"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/params"
)

// Natural-input variant of TestDFRLTwoNodeTxPathDiverges.
//
// Same rationale as api_dfrl_exchange_natural_test.go: close the narrative gap
// between scope-grid commitment divergence and chain-level payload rejection
// using unperturbed training data.
//
// The witness is the same 10-client × 16-coordinate window [112, 128) from
// round 0, delivered through VulnerableWrapper's agg(bytes) entry point as a
// transaction. The contract hashes the precompile output and writes to storage,
// turning the divergent aggregate into a divergent receipt root.
//
//	go test -tags e3lite_vulnerable          ... DFRL_NATV_OUT=natv_fwd.json
//	go test -tags "e3lite_vulnerable e3lite_reverse" ... DFRL_NATV_OUT=natv_rev.json DFRL_NATV_IN=natv_fwd.json
//	go test -tags e3lite_vulnerable          ... DFRL_NATV_IN=natv_rev.json

const dfrlNatTxvWitnessInput = "0x0000000a0000000000001d1700000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc1000000000000000007d900000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000001dcc00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc1000000000000000018ed00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000001e6900000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000000c5400000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000000000000000266200000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79d000bcca02003bca20003cd0b600bc240000bc4cf0003bc10000000000000000147a00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c795400bc7a2400bcca16003bc9d8003cd02600bc242800bc46e8003bc07000000000000000130600000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000000000000000161800000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000"

func dfrlNatTxvGenesis(art dfrlDetArtifact) *core.Genesis {
	config := *params.MergedTestChainConfig
	zero := uint64(0)
	config.DFRLTime = &zero

	return &core.Genesis{
		Config: &config,
		Alloc: types.GenesisAlloc{
			testAddr:                           {Balance: testBalance},
			dfrlTxvWrapperAddr:                 {Code: art.BinRuntime, Balance: common.Big0},
			params.BeaconRootsAddress:          {Nonce: 1, Code: common.FromHex("0x3373fffffffffffffffffffffffffffffffffffffffe14604457602036146024575f5ffd5b620180005f350680545f35146037575f5ffd5b6201800001545f5260205ff35b6201800042064281555f359062018000015500"), Balance: common.Big0},
			params.HistoryStorageAddress:       {Nonce: 1, Code: params.HistoryStorageCode, Balance: common.Big0},
			params.WithdrawalQueueAddress:      {Nonce: 1, Code: params.WithdrawalQueueCode, Balance: common.Big0},
			params.ConsolidationQueueAddress:   {Nonce: 1, Code: params.ConsolidationQueueCode, Balance: common.Big0},
			params.BuilderDepositAddress:       {Nonce: 1, Code: params.BuilderDepositCode, Balance: common.Big0},
			params.BuilderExitAddress:          {Nonce: 1, Code: params.BuilderExitCode, Balance: common.Big0},
			params.DeterministicFactoryAddress: {Nonce: 1, Code: params.DeterministicFactoryCode, Balance: common.Big0},
		},
		ExtraData:  []byte("dfrl natural tx-path vulnerable genesis"),
		Timestamp:  9000,
		GasLimit:   30_000_000,
		Difficulty: common.Big0,
	}
}

func TestDFRLTwoNodeNaturalTxPathDiverges(t *testing.T) {
	outPath := os.Getenv("DFRL_NATV_OUT")
	inPath := os.Getenv("DFRL_NATV_IN")
	if outPath == "" && inPath == "" {
		t.Skip("set DFRL_NATV_OUT and/or DFRL_NATV_IN; see this file's header")
	}

	art := loadSolcArtifact(t, "VulnerableWrapper.json", "VulnerableWrapper")
	genesis := dfrlNatTxvGenesis(art)
	n, ethservice := startEthService(t, genesis, nil)
	defer n.Close()
	api := newConsensusAPIWithoutHeartbeat(ethservice)

	data, err := art.ABI.Pack("agg", common.FromHex(dfrlNatTxvWitnessInput))
	if err != nil {
		t.Fatalf("pack agg: %v", err)
	}
	tx := types.MustSignNewTx(testKey, types.LatestSigner(genesis.Config), &types.LegacyTx{
		Nonce: 0, To: &dfrlTxvWrapperAddr, Gas: 5_000_000,
		GasPrice: big.NewInt(params.InitialBaseFee * 2), Data: data,
	})
	for i, err := range ethservice.TxPool().Add([]*types.Transaction{tx}, true) {
		if err != nil {
			t.Fatalf("tx %d into pool: %v", i, err)
		}
	}

	parent := ethservice.BlockChain().CurrentBlock()
	t.Logf("compiled accumulation order: %s", vm.DFRLVulnOrderName())
	t.Logf("genesis hash %v; natural-input 10-client 16-coord agg() via tx through %v",
		genesis.ToBlock().Hash(), dfrlTxvWrapperAddr)

	own := getNewEnvelope(t, api, parent, []*types.Withdrawal{}, &dfrlExchangeBeaconRoot)
	if got := len(own.ExecutionPayload.Transactions); got != 1 {
		t.Fatalf("payload carries %d transactions, expected the one agg() call", got)
	}
	t.Logf("own payload: number %d 1 tx state root %v block hash %v",
		own.ExecutionPayload.Number, own.ExecutionPayload.StateRoot,
		own.ExecutionPayload.BlockHash)

	if st := submitPayload(t, api, own); st.Status != engine.VALID {
		t.Fatalf("control: own payload should be VALID, got %v (%v)", st.Status, errString(st.ValidationError))
	}
	t.Log("control: own payload is VALID on its own builder")

	if outPath != "" {
		writeEnvelope(t, outPath, own)
	}
	if inPath != "" {
		peer := readEnvelope(t, inPath)
		judgePeerEnvelope(t, api, own, peer, false)
	}
}
