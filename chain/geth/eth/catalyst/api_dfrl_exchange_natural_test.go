//go:build e3lite_vulnerable

// Copyright 2026 The DT-DFRL authors.

package catalyst

import (
	"os"
	"testing"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/params"
)

// Natural-input variant of TestDFRLTwoNodePayloadExchange.
//
// This is the closing experiment for the "no attacker needed" narrative. The
// original exchange test uses a three-client witness perturbed by one
// quantization step (the attacker-closure case from §3.3). This test uses an
// unperturbed 16-coordinate window from real federated-MNIST round 0 with all
// 10 original training clients and weights. No client-side perturbation.
//
// Window: coordinates [112, 128) of 7850. At this window, 8 of 16 coordinates
// show cross-client diversity (clients 6 and 7 differ from the majority), and
// seq-fwd vs seq-rev produce 6 differing aggregate coordinates and different
// SHA3-256 commitments. All six conforming strategies produce 5 distinct
// commitments -- the same count the scope-grid survey reports across 195
// model-seed-round cells.
//
// The hardcoded contract is MandatoryVulnAgg_Natural.sol, compiled with
// solc 0.8.36. It calls the vulnerable float precompile at 0x0103 with a
// 764-byte input encoding 10 clients × 16 float32 parameters.
//
// Protocol: same three-run exchange as the original test.
//
//	go test -tags e3lite_vulnerable          ... DFRL_NAT_OUT=nat_fwd.json
//	go test -tags "e3lite_vulnerable e3lite_reverse" ... DFRL_NAT_OUT=nat_rev.json DFRL_NAT_IN=nat_fwd.json
//	go test -tags e3lite_vulnerable          ... DFRL_NAT_IN=nat_rev.json

// dfrlNatAggRuntime is the deployed runtime bytecode of
// chain/contracts/MandatoryVulnAgg_Natural.sol (10 clients, 16 coordinates,
// unperturbed round-0 training data, window [112,128)).
const dfrlNatAggRuntime = "0x608060405234801561000f575f5ffd5b5060043610610038575f3560e01c806327ff01e91461013357806382bc07e61461015157610039565b5b5f6040518061032001604052806102fc81526020016103306102fc913990505f5f61010373ffffffffffffffffffffffffffffffffffffffff168360405161008191906101e6565b5f60405180830381855afa9150503d805f81146100b9576040519150601f19603f3d011682016040523d82523d5f602084013e6100be565b606091505b509150915081610103576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016100fa90610256565b60405180910390fd5b80805190602001205f8190555060205f36905010610131575f3681019061012a91906102ab565b6001819055505b005b61013b61016f565b60405161014891906102ee565b60405180910390f35b610159610174565b6040516101669190610316565b60405180910390f35b5f5481565b60015481565b5f81519050919050565b5f81905092915050565b5f5b838110156101ab578082015181840152602081019050610190565b5f8484015250505050565b5f6101c08261017a565b6101ca8185610184565b93506101da81856020860161018e565b80840191505092915050565b5f6101f182846101b6565b915081905092915050565b5f82825260208201905092915050565b7f76756c6e20616767206661696c656400000000000000000000000000000000005f82015250565b5f610240600f836101fc565b915061024b8261020c565b602082019050919050565b5f6020820190508181035f83015261026d81610234565b9050919050565b5f5ffd5b5f819050919050565b61028a81610278565b8114610294575f5ffd5b50565b5f813590506102a581610281565b92915050565b5f602082840312156102c0576102bf610274565b5b5f6102cd84828501610297565b91505092915050565b5f819050919050565b6102e8816102d6565b82525050565b5f6020820190506103015f8301846102df565b92915050565b61031081610278565b82525050565b5f6020820190506103295f830184610307565b9291505056fe0000000a0000000000001d1700000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc1000000000000000007d900000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000001dcc00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc1000000000000000018ed00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000001e6900000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000000c5400000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000000000000000266200000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79d000bcca02003bca20003cd0b600bc240000bc4cf0003bc10000000000000000147a00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c795400bc7a2400bcca16003bc9d8003cd02600bc242800bc46e8003bc07000000000000000130600000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000000000000000161800000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000a2646970667358221220f9b6e51373d498153dda90400e4b75923724563f8c1098805ea2b3b09be9fc9064736f6c63430008240033"

func dfrlNatGenesis() *core.Genesis {
	config := *params.MergedTestChainConfig
	zero := uint64(0)
	config.DFRLTime = &zero
	config.DFRLAggregationInterval = 1

	return &core.Genesis{
		Config: &config,
		Alloc: types.GenesisAlloc{
			testAddr: {Balance: testBalance},
			params.DFRLAggregatorAddress: {
				Code:    common.FromHex(dfrlNatAggRuntime),
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
		ExtraData:  []byte("dfrl natural witness genesis"),
		Timestamp:  9000,
		Difficulty: common.Big0,
	}
}

func TestDFRLTwoNodePayloadExchangeNatural(t *testing.T) {
	outPath := os.Getenv("DFRL_NAT_OUT")
	inPath := os.Getenv("DFRL_NAT_IN")
	if outPath == "" && inPath == "" {
		t.Skip("set DFRL_NAT_OUT and/or DFRL_NAT_IN; see this file's header")
	}

	genesis := dfrlNatGenesis()
	n, ethservice := startEthService(t, genesis, nil)
	defer n.Close()
	api := newConsensusAPIWithoutHeartbeat(ethservice)

	parent := ethservice.BlockChain().CurrentBlock()
	t.Logf("compiled accumulation order: %s", vm.DFRLVulnOrderName())
	t.Logf("genesis hash %v; natural-input 10-client 16-coord mandatory aggregation",
		genesis.ToBlock().Hash())
	t.Logf("building over parent %d (%v)", parent.Number, parent.Hash())

	own := getNewEnvelope(t, api, parent, []*types.Withdrawal{}, &dfrlExchangeBeaconRoot)
	t.Logf("own payload: number %d state root %v block hash %v",
		own.ExecutionPayload.Number, own.ExecutionPayload.StateRoot,
		own.ExecutionPayload.BlockHash)

	if outPath != "" {
		writeEnvelope(t, outPath, own)
	}
	if inPath != "" {
		peer := readEnvelope(t, inPath)
		judgePeerEnvelope(t, api, own, peer, false)
	}
}
