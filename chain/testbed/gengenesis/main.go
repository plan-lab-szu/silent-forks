// Command gengenesis builds genesis.json for the DFRL testbed by reusing
// go-ethereum's own ChainConfig/Genesis types and JSON marshaling -- rather
// than hand-authoring the config, which has enough interacting fields
// (block-based forks, time-based forks, merge fields) that a hand-written
// JSON is a real way to introduce an inconsistency this project's own red
// lines warn against.
//
// Base config: params.AllDevChainProtocolChanges (same base --dev mode uses),
// with OsakaTime and DFRLTime both pinned to 0 -- mirrors dfrlConfig() in
// core/vm/contracts_dfrl_test.go exactly, so the testbed and the unit tests
// share one definition of "DFRL active".
package main

import (
	"encoding/json"
	"fmt"
	"math/big"
	"os"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/params"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: gengenesis <keys.json>")
		os.Exit(2)
	}
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	var keys []struct {
		Address string `json:"address"`
		PrivKey string `json:"privkey"`
	}
	if err := json.Unmarshal(raw, &keys); err != nil {
		panic(err)
	}

	zero := uint64(0)
	cfg := *params.AllDevChainProtocolChanges
	cfg.OsakaTime = &zero
	cfg.DFRLTime = &zero

	// Every fork this genesis activates (Shanghai..Bogota, all pinned to time 0)
	// expects its own system contract to already have code at genesis --
	// withdrawal/consolidation queues, beacon roots, history storage, etc.
	// core.DeveloperGenesisBlock installs these via SystemContractAllocs();
	// without them, block building fails with "empty system contract" the
	// first time a post-Shanghai fork tries to call into one. Reuse the same
	// helper rather than hand-listing addresses, for the same reason the rest
	// of this config is copied from AllDevChainProtocolChanges instead of
	// hand-authored: one canonical source, not two that can drift apart.
	alloc := core.SystemContractAllocs()
	alloc[common.BytesToAddress([]byte{0x01, 0x01})] = types.Account{Balance: big.NewInt(1)} // DFRL FedAvg
	alloc[common.BytesToAddress([]byte{0x01, 0x02})] = types.Account{Balance: big.NewInt(1)} // DFRL RobustAgg
	funded := new(big.Int)
	funded.SetString("100000000000000000000", 10) // 100 ETH-equivalent, plenty for testbed gas
	for _, k := range keys {
		alloc[common.HexToAddress(k.Address)] = types.Account{Balance: funded}
	}

	genesis := &core.Genesis{
		Config:     &cfg,
		GasLimit:   200_000_000, // generous: the U4 test showed aggregateModel can need >100M gas before the O(n^2) fix
		BaseFee:    big.NewInt(params.InitialBaseFee),
		Difficulty: big.NewInt(0),
		Timestamp:  0,
		Alloc:      alloc,
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(genesis); err != nil {
		panic(err)
	}
}
