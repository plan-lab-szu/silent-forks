// Copyright 2026 The DT-DFRL authors.

package vm_test

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"testing"

	dfrlagg "dfrl-agg"
	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/vm/runtime"
	"github.com/holiman/uint256"
)

// contractArtifact loads solc's --combined-json abi,bin,bin-runtime output.
// The contract itself lives in ../../contracts/DFRLAggregator.sol -- outside
// this Go module, since it is the paper's contract, not part of the geth
// fork. Compiled once by hand (see HANDBOOK.md for the solc command); this
// test reads the resulting JSON rather than shelling out to solc so the test
// suite does not depend on solc being on PATH.
type solcArtifact struct {
	ABI        abi.ABI
	Bin        []byte
	BinRuntime []byte
}

func loadDFRLAggregator(t *testing.T) solcArtifact {
	t.Helper()
	path := filepath.Join("..", "..", "..", "contracts", "DFRLAggregator.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("compiled contract not found at %s (run solc first, see HANDBOOK.md): %v", path, err)
	}
	return parseSolcCombinedJSON(t, raw, "DFRLAggregator")
}

// parseSolcCombinedJSON parses `solc --combined-json abi,bin,bin-runtime`
// output and selects the entry named contractName (the combined-json map key
// is "path/File.sol:ContractName"). A name is required -- not "whichever
// entry comes first" -- because DFRLAggregator.sol and
// DFRLAggregatorCommitOnly.sol both import DFRLInputBuilder.sol, so their
// compiled JSON holds TWO contracts each, and Go's randomized map iteration
// order previously picked whichever one happened to come first (this
// surfaced as a flaky "method not found" once the library import was added --
// see HANDBOOK.md's gas-diagnosis correction for how that was caught).
func parseSolcCombinedJSON(t *testing.T, raw []byte, contractName string) solcArtifact {
	t.Helper()
	var doc struct {
		Contracts map[string]struct {
			Bin        string          `json:"bin"`
			BinRuntime string          `json:"bin-runtime"`
			ABI        json.RawMessage `json:"abi"`
		} `json:"contracts"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("parse combined-json: %v", err)
	}
	var c struct {
		Bin        string
		BinRuntime string
		ABI        json.RawMessage
	}
	suffix := ":" + contractName
	found := false
	for k, v := range doc.Contracts {
		if strings.HasSuffix(k, suffix) {
			c.Bin, c.BinRuntime, c.ABI = v.Bin, v.BinRuntime, v.ABI
			found = true
			break
		}
	}
	if !found {
		names := make([]string, 0, len(doc.Contracts))
		for k := range doc.Contracts {
			names = append(names, k)
		}
		t.Fatalf("contract %q not found in combined-json; available: %v", contractName, names)
	}
	parsedABI, err := abi.JSON(bytes.NewReader(c.ABI))
	if err != nil {
		t.Fatalf("parse ABI: %v", err)
	}
	bin, err := hex.DecodeString(c.Bin)
	if err != nil {
		t.Fatalf("decode bin: %v", err)
	}
	binRuntime, err := hex.DecodeString(c.BinRuntime)
	if err != nil {
		t.Fatalf("decode bin-runtime: %v", err)
	}
	return solcArtifact{ABI: parsedABI, Bin: bin, BinRuntime: binRuntime}
}

// TestDFRLAggregatorContract deploys DFRLAggregator.sol, has several accounts
// upload quantized updates, calls aggregateModel (which STATICCALLs the DFRL
// precompile), and checks the on-chain result against a direct dfrl-agg call.
// This is the first real evidence that the precompile works from an ACTUAL
// SOLIDITY CONTRACT, not just hand-assembled test bytecode.
func TestDFRLAggregatorContract(t *testing.T) {
	art := loadDFRLAggregator(t)
	cfg := dfrlConfig()
	statedb, _ := state.New(common.Hash{}, state.NewDatabaseForTesting())

	contractAddr := common.HexToAddress("0x00000000000000000000000000000000c0ffee")
	statedb.CreateAccount(contractAddr)
	statedb.SetCode(contractAddr, art.BinRuntime, 0)

	const n, p = 5, 256
	weights := []int64{3, 5, 7, 11, 13}
	_, tensors := genUpdates(t, 4242, n, p, weights, 1<<14)

	round := big.NewInt(0)
	senders := make([]common.Address, n)
	for i := 0; i < n; i++ {
		senders[i] = common.BytesToAddress([]byte{byte(0x10 + i)})
		statedb.CreateAccount(senders[i])
		statedb.AddBalance(senders[i], uint256.NewInt(1_000_000_000_000_000_000), 0)

		enc, err := dfrlagg.Encode(tensors[i])
		if err != nil {
			t.Fatal(err)
		}
		input, err := art.ABI.Pack("uploadModel", round, uint64(weights[i]), enc)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := runtime.Call(contractAddr, input, &runtime.Config{
			ChainConfig: cfg, Origin: senders[i], State: statedb, GasLimit: 5_000_000,
		}); err != nil {
			t.Fatalf("uploadModel from sender %d: %v", i, err)
		}
	}

	aggInput, err := art.ABI.Pack("aggregateModel", round)
	if err != nil {
		t.Fatal(err)
	}
	caller := common.BytesToAddress([]byte{0x99})
	statedb.CreateAccount(caller)
	out, _, err := runtime.Call(contractAddr, aggInput, &runtime.Config{
		ChainConfig: cfg, Origin: caller, State: statedb, GasLimit: 20_000_000,
	})
	if err != nil {
		t.Fatalf("aggregateModel: %v", err)
	}
	var onChainRoot [32]byte
	if err := art.ABI.UnpackIntoInterface(&onChainRoot, "aggregateModel", out); err != nil {
		t.Fatal(err)
	}

	dlInput, err := art.ABI.Pack("downloadModel", round)
	if err != nil {
		t.Fatal(err)
	}
	dlOut, _, err := runtime.Call(contractAddr, dlInput, &runtime.Config{
		ChainConfig: cfg, Origin: caller, State: statedb, GasLimit: 5_000_000,
	})
	if err != nil {
		t.Fatalf("downloadModel: %v", err)
	}
	var onChainResult []byte
	if err := art.ABI.UnpackIntoInterface(&onChainResult, "downloadModel", dlOut); err != nil {
		t.Fatal(err)
	}

	ups := make([]dfrlagg.Update, n)
	for i := 0; i < n; i++ {
		var a dfrlagg.Addr
		a[19] = byte(i + 1)
		ups[i] = dfrlagg.Update{Addr: a, Weight: weights[i], Tensor: tensors[i]}
	}
	want, err := dfrlagg.FedAvg(ups)
	if err != nil {
		t.Fatal(err)
	}
	wantEnc, err := dfrlagg.Encode(want)
	if err != nil {
		t.Fatal(err)
	}
	wantRoot, err := dfrlagg.StateRoot(want)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(onChainResult, wantEnc) {
		t.Fatalf("on-chain result differs from direct dfrl-agg call\n got=%x\nwant=%x", onChainResult, wantEnc)
	}
	if onChainRoot != wantRoot {
		t.Logf("note: on-chain commitment is keccak256(canonical bytes) (%x); dfrl-agg's own StateRoot is SHA3-256 of the same bytes (%x) -- different hash functions, same input, both fine for their own purposes (see codec.go's StateRoot doc comment)", onChainRoot, wantRoot)
	}
	t.Logf("contract-mediated aggregation matches direct dfrl-agg call")
}
