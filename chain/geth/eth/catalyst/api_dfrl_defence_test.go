//go:build e3lite_vulnerable

// Copyright 2026 The DT-DFRL authors.

package catalyst

import (
	"bytes"
	"encoding/json"
	"math"
	"math/big"
	"os"
	"path/filepath"
	"testing"

	dfrlagg "dfrl-agg"

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/beacon/engine"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/params"
)

// The defence control for TestDFRLTwoNodePayloadExchange.
//
// That test leaves a gap an attentive reader will find immediately: it shows
// the unspecified float path breaks byte agreement, and says nothing about
// whether the specification fixes it. Without this control, "the specification
// eliminates the divergence" is an inference from the specification's text
// rather than a measurement.
//
// The control holds the two binaries fixed. They are the same two builds that
// disagreed -- one compiled with the forward accumulation order, one with the
// reverse -- exchanged through the same harness, over the same three client
// submissions from real federated-MNIST round 0. What changes is that the
// aggregation follows the specification: submissions arrive canonically encoded
// through the contract's own entry point, and the arithmetic runs on the
// deterministic integer precompile at 0x0101 instead of the float one at
// 0x0103. The expectation inverts: identical post-state roots, and each node
// accepting the other's payload.
//
//	go test -tags e3lite_vulnerable          ... DFRL_DET_OUT=det_fwd.json
//	go test -tags "e3lite_vulnerable e3lite_reverse" ... DFRL_DET_OUT=det_rev.json DFRL_DET_IN=det_fwd.json
//	go test -tags e3lite_vulnerable          ... DFRL_DET_IN=det_rev.json
//
// What this control does and does not establish. It establishes that the one
// implementation freedom that broke the unspecified path -- the accumulation
// order compiled into the aggregation kernel -- has no effect once the
// arithmetic is specified, because integer addition is associative and the
// specification fixes the representation, rounding and encoding. It does not
// establish immunity to every implementation difference; cross-language and
// cross-architecture agreement of the integer path is a separate claim with
// its own evidence (the Go/Python conformance vectors and the arm64 CI).
//
// Two changes separate this control from the divergence experiment, not one,
// and both are the specification rather than experimental convenience: the
// arithmetic moves to the specified integer path, and the submissions arrive
// canonically encoded through a submission entry point instead of as a float
// constant baked into the aggregation target. The second change also closes a
// scope limitation of the divergence experiment, which never exercised a
// submission path at all.

// dfrlDetAggregatorAddr hosts DFRLAggregatorCommitOnly. Any address does; this
// one matches the address core/vm's contract test uses.
var dfrlDetAggregatorAddr = common.HexToAddress("0x00000000000000000000000000000000c0ffee")

// dfrlDetWitness is the same three client submissions the divergence
// experiment aggregates, taken from MandatoryVulnAgg.sol's hardcoded input:
// real federated-MNIST round-0 weights with one coordinate each. Here they are
// carried as the float32 values they are, and quantized below exactly as a
// conforming client would before submitting.
var dfrlDetWitness = []struct {
	weight uint64
	bits   uint32
}{
	{7447, 0xbc13f800},
	{2009, 0x3c364000},
	{7628, 0x3bfc6800},
}

type dfrlDetArtifact struct {
	ABI        abi.ABI
	BinRuntime []byte
}

// loadCommitOnlyArtifact reads the precompiled contract rather than shelling
// out to solc, for the same reason core/vm's contract test does: the suite must
// not depend on solc being on PATH, and both builds must provably execute the
// same contract bytes.
func loadCommitOnlyArtifact(t *testing.T) dfrlDetArtifact {
	return loadSolcArtifact(t, "DFRLAggregatorCommitOnly.json", "DFRLAggregatorCommitOnly")
}

// loadSolcArtifact reads one contract out of a solc --combined-json file.
func loadSolcArtifact(t *testing.T, file, contract string) dfrlDetArtifact {
	t.Helper()
	path := filepath.Join("..", "..", "..", "contracts", file)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("compiled contract not found at %s: %v", path, err)
	}
	var combined struct {
		Contracts map[string]struct {
			ABI        json.RawMessage `json:"abi"`
			BinRuntime string          `json:"bin-runtime"`
		} `json:"contracts"`
	}
	if err := json.Unmarshal(raw, &combined); err != nil {
		t.Fatal(err)
	}
	for name, c := range combined.Contracts {
		if filepath.Ext(name) != "" && len(c.BinRuntime) > 0 &&
			len(name) >= len(contract) && name[len(name)-len(contract):] == contract {
			parsed, err := abi.JSON(bytes.NewReader(c.ABI))
			if err != nil {
				t.Fatal(err)
			}
			return dfrlDetArtifact{ABI: parsed, BinRuntime: common.FromHex(c.BinRuntime)}
		}
	}
	t.Fatalf("%s not found in %s", contract, file)
	return dfrlDetArtifact{}
}

// dfrlDetGenesis is identical in both builds: the deterministic aggregation
// contract, the system contracts the post-Shanghai forks need, and a funded
// account to submit from. No mandatory aggregation interval is set: here the
// aggregation is driven by a transaction through the contract's entry point,
// which is the path the specification describes for clients.
func dfrlDetGenesis(art dfrlDetArtifact) *core.Genesis {
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
		ExtraData:  []byte("dfrl deterministic genesis"),
		Timestamp:  9000,
		GasLimit:   30_000_000,
		Difficulty: common.Big0,
	}
}

// dfrlDetTxs builds the submission and aggregation calls: one uploadModel per
// client with its canonically encoded quantized tensor, then aggregateModel.
// Every field is fixed, so both processes construct byte-identical transactions;
// the exchange asserts that rather than assuming it.
func dfrlDetTxs(t *testing.T, art dfrlDetArtifact, config *params.ChainConfig) []*types.Transaction {
	t.Helper()
	signer := types.LatestSigner(config)
	round := big.NewInt(0)
	txs := make([]*types.Transaction, 0, len(dfrlDetWitness)+1)

	for i, w := range dfrlDetWitness {
		theta := []float32{math.Float32frombits(w.bits)}
		tensor, err := dfrlagg.Quantize(theta, []uint32{1}, dfrlagg.DefaultF)
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
		Nonce: uint64(len(dfrlDetWitness)), To: &dfrlDetAggregatorAddr, Gas: 5_000_000,
		GasPrice: big.NewInt(params.InitialBaseFee * 2), Data: data,
	}))
	return txs
}

func TestDFRLTwoNodeDeterministicPathAgrees(t *testing.T) {
	outPath := os.Getenv("DFRL_DET_OUT")
	inPath := os.Getenv("DFRL_DET_IN")
	if outPath == "" && inPath == "" {
		t.Skip("set DFRL_DET_OUT and/or DFRL_DET_IN; see this file's header for the three-run protocol")
	}

	art := loadCommitOnlyArtifact(t)
	genesis := dfrlDetGenesis(art)
	n, ethservice := startEthService(t, genesis, nil)
	defer n.Close()
	api := newConsensusAPIWithoutHeartbeat(ethservice)

	txs := dfrlDetTxs(t, art, genesis.Config)
	for i, err := range ethservice.TxPool().Add(txs, true) {
		if err != nil {
			t.Fatalf("tx %d into pool: %v", i, err)
		}
	}

	parent := ethservice.BlockChain().CurrentBlock()
	t.Logf("compiled accumulation order: %s (unused on the specified path)", vm.DFRLVulnOrderName())
	t.Logf("genesis hash %v; %d submissions + 1 aggregation call through %v",
		genesis.ToBlock().Hash(), len(dfrlDetWitness), dfrlDetAggregatorAddr)

	own := getNewEnvelope(t, api, parent, []*types.Withdrawal{}, &dfrlExchangeBeaconRoot)
	if got := len(own.ExecutionPayload.Transactions); got != len(txs) {
		t.Fatalf("payload carries %d transactions, expected all %d; "+
			"an excluded call would leave the aggregation out of the block", got, len(txs))
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
	if inPath == "" {
		return
	}
	judgePeerEnvelope(t, api, own, readEnvelope(t, inPath), true)
}
