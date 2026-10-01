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

// The single-variable half of the divergence/defence pair.
//
// TestDFRLTwoNodePayloadExchange drives the float aggregation through the
// height-scheduled system call, with the submissions baked into the deployed
// contract. TestDFRLTwoNodeDeterministicPathAgrees drives the specified
// aggregation through a contract entry point, with the submissions arriving as
// transaction calldata. Both results hold, but comparing them means comparing
// across two differences at once -- the arithmetic AND the delivery -- so
// neither can be reported as "only the arithmetic changed".
//
// This test supplies the missing cell: the float aggregation, delivered the
// same way the specified one is. Submissions arrive as transaction calldata,
// the contract writes the precompile's output to storage (which is what turns a
// divergent result into a divergent state root -- return data alone is not
// consensus state), and the two builds exchange payloads through the same
// harness. Against the deterministic test, the only thing that differs is the
// specification: encoding, representation, and which precompile runs.
//
//	go test -tags e3lite_vulnerable          ... DFRL_TXV_OUT=txv_fwd.json
//	go test -tags "e3lite_vulnerable e3lite_reverse" ... DFRL_TXV_OUT=txv_rev.json DFRL_TXV_IN=txv_fwd.json
//	go test -tags e3lite_vulnerable          ... DFRL_TXV_IN=txv_rev.json
//
// The mandatory-system-call test is not superseded by this one. It carries a
// property this test does not: aggregation scheduled by height, in a block with
// no transactions, which no proposer can decline to run. Keep both.
//
// One asymmetry remains and must be reported. VulnerableWrapper takes the whole
// batch in a single `agg(bytes)` call, while DFRLAggregatorCommitOnly has a
// per-client `uploadModel` followed by `aggregateModel`, so this test sends one
// transaction where the deterministic test sends four. That is a difference in
// the two contracts' interfaces, not in the arithmetic or the delivery
// mechanism, and it is not what makes either result come out the way it does.

// dfrlTxvWrapperAddr hosts VulnerableWrapper.
var dfrlTxvWrapperAddr = common.HexToAddress("0x00000000000000000000000000000000BADF10")

// dfrlTxvWitnessInput is the vulnerable precompile's raw input for the same
// three client submissions the other two tests use: numClients uint32, then per
// client weight uint64, paramCount uint32, and theta as big-endian float32
// words. Byte-identical to the constant compiled into MandatoryVulnAgg.sol, so
// all three experiments aggregate the same values.
const dfrlTxvWitnessInput = "0x000000030000000000001d1700000001bc13f80000000000000007d9000000013c3640000000000000001dcc000000013bfc6800"

func dfrlTxvGenesis(art dfrlDetArtifact) *core.Genesis {
	config := *params.MergedTestChainConfig
	zero := uint64(0)
	config.DFRLTime = &zero
	// No DFRLAggregationInterval: the aggregation here is driven by a
	// transaction, so that this test differs from the deterministic one only
	// in the specification and not in how the call is scheduled.

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
		ExtraData:  []byte("dfrl tx-path vulnerable genesis"),
		Timestamp:  9000,
		GasLimit:   30_000_000,
		Difficulty: common.Big0,
	}
}

func TestDFRLTwoNodeTxPathDiverges(t *testing.T) {
	outPath := os.Getenv("DFRL_TXV_OUT")
	inPath := os.Getenv("DFRL_TXV_IN")
	if outPath == "" && inPath == "" {
		t.Skip("set DFRL_TXV_OUT and/or DFRL_TXV_IN; see this file's header for the three-run protocol")
	}

	art := loadSolcArtifact(t, "VulnerableWrapper.json", "VulnerableWrapper")
	genesis := dfrlTxvGenesis(art)
	n, ethservice := startEthService(t, genesis, nil)
	defer n.Close()
	api := newConsensusAPIWithoutHeartbeat(ethservice)

	data, err := art.ABI.Pack("agg", common.FromHex(dfrlTxvWitnessInput))
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
	t.Logf("genesis hash %v; one agg() call through %v carrying the three submissions as calldata",
		genesis.ToBlock().Hash(), dfrlTxvWrapperAddr)

	own := getNewEnvelope(t, api, parent, []*types.Withdrawal{}, &dfrlExchangeBeaconRoot)
	if got := len(own.ExecutionPayload.Transactions); got != 1 {
		t.Fatalf("payload carries %d transactions, expected the one agg() call; "+
			"without it the block contains no aggregation at all", got)
	}
	t.Logf("own payload: number %d 1 tx state root %v block hash %v",
		own.ExecutionPayload.Number, own.ExecutionPayload.StateRoot, own.ExecutionPayload.BlockHash)

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
