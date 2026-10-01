//go:build e3lite_vulnerable

// Copyright 2026 The DT-DFRL authors.

package catalyst

import (
	"encoding/binary"
	"math"
	"math/big"
	"os"
	"testing"

	dfrlagg "dfrl-agg"

	"github.com/ethereum/go-ethereum/beacon/engine"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/params"
)

// Natural-input defence control: same 10-client 16-coordinate window as the
// natural exchange tests, run through the deterministic integer path.
//
// Closes the same gap as api_dfrl_defence_test.go but with unperturbed input.
// Both orderings should produce identical roots and mutual VALID because the
// integer arithmetic is associative and the specification fixes everything.
//
//	go test -tags e3lite_vulnerable          ... DFRL_NATD_OUT=natd_fwd.json
//	go test -tags "e3lite_vulnerable e3lite_reverse" ... DFRL_NATD_OUT=natd_rev.json DFRL_NATD_IN=natd_fwd.json
//	go test -tags e3lite_vulnerable          ... DFRL_NATD_IN=natd_rev.json

// dfrlNatDetWitness is parsed from dfrlNatTxvWitnessInput at init time.
var dfrlNatDetWitness []struct {
	weight uint64
	theta  []float32
}

func init() {
	raw := common.FromHex(dfrlNatTxvWitnessInput)
	n := binary.BigEndian.Uint32(raw)
	off := 4
	for i := uint32(0); i < n; i++ {
		w := binary.BigEndian.Uint64(raw[off:])
		off += 8
		p := int(binary.BigEndian.Uint32(raw[off:]))
		off += 4
		theta := make([]float32, p)
		for j := 0; j < p; j++ {
			theta[j] = math.Float32frombits(binary.BigEndian.Uint32(raw[off:]))
			off += 4
		}
		dfrlNatDetWitness = append(dfrlNatDetWitness, struct {
			weight uint64
			theta  []float32
		}{w, theta})
	}
}

func dfrlNatDetGenesis(art dfrlDetArtifact) *core.Genesis {
	config := *params.MergedTestChainConfig
	zero := uint64(0)
	config.DFRLTime = &zero

	return &core.Genesis{
		Config: &config,
		Alloc: types.GenesisAlloc{
			testAddr:                           {Balance: testBalance},
			dfrlDetAggregatorAddr:              {Code: art.BinRuntime, Balance: common.Big0},
			params.BeaconRootsAddress:          {Nonce: 1, Code: common.FromHex("0x3373fffffffffffffffffffffffffffffffffffffffe14604457602036146024575f5ffd5b620180005f350680545f35146037575f5ffd5b6201800001545f5260205ff35b6201800042064281555f359062018000015500"), Balance: common.Big0},
			params.HistoryStorageAddress:       {Nonce: 1, Code: params.HistoryStorageCode, Balance: common.Big0},
			params.WithdrawalQueueAddress:      {Nonce: 1, Code: params.WithdrawalQueueCode, Balance: common.Big0},
			params.ConsolidationQueueAddress:   {Nonce: 1, Code: params.ConsolidationQueueCode, Balance: common.Big0},
			params.BuilderDepositAddress:       {Nonce: 1, Code: params.BuilderDepositCode, Balance: common.Big0},
			params.BuilderExitAddress:          {Nonce: 1, Code: params.BuilderExitCode, Balance: common.Big0},
			params.DeterministicFactoryAddress: {Nonce: 1, Code: params.DeterministicFactoryCode, Balance: common.Big0},
		},
		ExtraData:  []byte("dfrl natural deterministic genesis"),
		Timestamp:  9000,
		GasLimit:   30_000_000,
		Difficulty: common.Big0,
	}
}

func dfrlNatDetTxs(t *testing.T, art dfrlDetArtifact, config *params.ChainConfig) []*types.Transaction {
	t.Helper()
	signer := types.LatestSigner(config)
	round := big.NewInt(0)
	txs := make([]*types.Transaction, 0, len(dfrlNatDetWitness)+1)

	for i, w := range dfrlNatDetWitness {
		tensor, err := dfrlagg.Quantize(w.theta, []uint32{uint32(len(w.theta))}, dfrlagg.DefaultF)
		if err != nil {
			t.Fatalf("quantize client %d: %v", i, err)
		}
		enc, err := dfrlagg.Encode(tensor)
		if err != nil {
			t.Fatalf("encode client %d: %v", i, err)
		}
		data, err := art.ABI.Pack("uploadModel", round, w.weight, enc)
		if err != nil {
			t.Fatalf("pack uploadModel %d: %v", i, err)
		}
		txs = append(txs, types.MustSignNewTx(testKey, signer, &types.LegacyTx{
			Nonce: uint64(i), To: &dfrlDetAggregatorAddr, Gas: 1_000_000,
			GasPrice: big.NewInt(params.InitialBaseFee * 2), Data: data,
		}))
	}

	data, err := art.ABI.Pack("aggregateModel", round)
	if err != nil {
		t.Fatalf("pack aggregateModel: %v", err)
	}
	txs = append(txs, types.MustSignNewTx(testKey, signer, &types.LegacyTx{
		Nonce: uint64(len(dfrlNatDetWitness)), To: &dfrlDetAggregatorAddr, Gas: 5_000_000,
		GasPrice: big.NewInt(params.InitialBaseFee * 2), Data: data,
	}))
	return txs
}

func TestDFRLTwoNodeNaturalDeterministicPathAgrees(t *testing.T) {
	outPath := os.Getenv("DFRL_NATD_OUT")
	inPath := os.Getenv("DFRL_NATD_IN")
	if outPath == "" && inPath == "" {
		t.Skip("set DFRL_NATD_OUT and/or DFRL_NATD_IN; see this file's header")
	}

	art := loadCommitOnlyArtifact(t)
	genesis := dfrlNatDetGenesis(art)
	n, ethservice := startEthService(t, genesis, nil)
	defer n.Close()
	api := newConsensusAPIWithoutHeartbeat(ethservice)

	txs := dfrlNatDetTxs(t, art, genesis.Config)
	for i, err := range ethservice.TxPool().Add(txs, true) {
		if err != nil {
			t.Fatalf("tx %d into pool: %v", i, err)
		}
	}

	parent := ethservice.BlockChain().CurrentBlock()
	t.Logf("compiled accumulation order: %s (unused on the specified path)", vm.DFRLVulnOrderName())
	t.Logf("genesis hash %v; %d natural submissions + 1 aggregation through %v",
		genesis.ToBlock().Hash(), len(dfrlNatDetWitness), dfrlDetAggregatorAddr)

	own := getNewEnvelope(t, api, parent, []*types.Withdrawal{}, &dfrlExchangeBeaconRoot)
	if got := len(own.ExecutionPayload.Transactions); got != len(txs) {
		t.Fatalf("payload carries %d transactions, expected all %d", got, len(txs))
	}
	t.Logf("own payload: number %d %d txs state root %v block hash %v",
		own.ExecutionPayload.Number, len(own.ExecutionPayload.Transactions),
		own.ExecutionPayload.StateRoot, own.ExecutionPayload.BlockHash)

	if st := submitPayload(t, api, own); st.Status != engine.VALID {
		t.Fatalf("control: own payload should be VALID, got %v (%v)", st.Status, errString(st.ValidationError))
	}
	t.Log("control: own payload is VALID on its own builder")

	if outPath != "" {
		writeEnvelope(t, outPath, own)
	}
	if inPath != "" {
		peer := readEnvelope(t, inPath)
		judgePeerEnvelope(t, api, own, peer, true)
	}
}
