// Command driver runs the real X6 experiment: identical signed transactions
// broadcast to N independently-running geth processes (no p2p between them --
// each is its own isolated --dev chain; see ARTIFACTS.md for the scope),
// and their resulting state roots compared after each round.
//
//	cd chain/testbed
//	go run ./driver --rpc 8600,8601,8602,8603,8604 --rounds 100 --keys keys.json --contract ../contracts/DFRLAggregator.json
//
// What this proves that the in-process X6-lite test (core/vm's
// contracts_dfrl_x6_test.go) does not: real node processes, real JSON-RPC,
// real mempool/block-building pipeline, real on-disk state commitment -- not
// just an in-memory EVM call. What it does NOT prove: consensus over WHICH
// transactions get included when nodes disagree (there is no p2p, so there is
// no "who proposes block N" question here) -- that question is orthogonal to
// whether the state transition function itself is deterministic, which is
// exactly what route B claims and exactly what this measures.
package main

import (
	"bytes"
	"crypto/ecdsa"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"os"
	"strings"
	"time"

	dfrlagg "dfrl-agg"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
)

const chainID = 1337

type keyEntry struct {
	Address string `json:"address"`
	PrivKey string `json:"privkey"`
}

type node struct {
	url   string
	nonce map[common.Address]uint64
}

func (n *node) rpc(method string, params ...interface{}) (json.RawMessage, error) {
	body, _ := json.Marshal(map[string]interface{}{
		"jsonrpc": "2.0", "id": 1, "method": method, "params": params,
	})
	resp, err := http.Post(n.url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if out.Error != nil {
		return nil, fmt.Errorf("%s: %s", method, out.Error.Message)
	}
	return out.Result, nil
}

func (n *node) sendRaw(raw []byte) (common.Hash, error) {
	var hashHex string
	res, err := n.rpc("eth_sendRawTransaction", "0x"+hex.EncodeToString(raw))
	if err != nil {
		return common.Hash{}, err
	}
	if err := json.Unmarshal(res, &hashHex); err != nil {
		return common.Hash{}, err
	}
	return common.HexToHash(hashHex), nil
}

type receipt struct {
	BlockNumber string `json:"blockNumber"`
	Status      string `json:"status"`
	Logs        []struct {
		Data   string   `json:"data"`
		Topics []string `json:"topics"`
	} `json:"logs"`
	ContractAddress string `json:"contractAddress"`
}

func (n *node) waitReceipt(hash common.Hash, timeout time.Duration) (*receipt, error) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		res, err := n.rpc("eth_getTransactionReceipt", hash.Hex())
		if err == nil && len(res) > 4 { // "null" is 4 bytes
			var r receipt
			if err := json.Unmarshal(res, &r); err != nil {
				return nil, err
			}
			return &r, nil
		}
		time.Sleep(200 * time.Millisecond)
	}
	return nil, fmt.Errorf("timed out waiting for receipt %s", hash.Hex())
}

func (n *node) blockStateRoot(blockNumHex string) (common.Hash, error) {
	res, err := n.rpc("eth_getBlockByNumber", blockNumHex, false)
	if err != nil {
		return common.Hash{}, err
	}
	var b struct {
		StateRoot string `json:"stateRoot"`
	}
	if err := json.Unmarshal(res, &b); err != nil {
		return common.Hash{}, err
	}
	return common.HexToHash(b.StateRoot), nil
}

// callCommittedRoot reads DFRLAggregator.committedRoot(round) via eth_call
// against the node's CURRENT chain tip -- an independent confirmation of the
// commitment value, taken a different way (state read, not event-log
// decoding) than the value extracted from the transaction receipt.
func (n *node) callCommittedRoot(contract common.Address, selector []byte, round uint64) (common.Hash, error) {
	data := packUint256Call(selector, round)
	callObj := map[string]string{
		"to":   contract.Hex(),
		"data": "0x" + hex.EncodeToString(data),
	}
	res, err := n.rpc("eth_call", callObj, "latest")
	if err != nil {
		return common.Hash{}, err
	}
	var hexStr string
	if err := json.Unmarshal(res, &hexStr); err != nil {
		return common.Hash{}, err
	}
	return common.HexToHash(hexStr), nil
}

func mustDecodeKey(hexKey string) *ecdsa.PrivateKey {
	k, err := crypto.HexToECDSA(hexKey)
	if err != nil {
		log.Fatal(err)
	}
	return k
}

// buildTx builds and signs one legacy transaction. Deliberately legacy
// (type 0), not a dynamic-fee tx: it is the simplest tx type this codebase
// still supports and keeps the "same raw bytes to every node" property
// trivially easy to reason about.
func buildTx(key *ecdsa.PrivateKey, nonce uint64, to *common.Address, gas uint64, data []byte) []byte {
	tx := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: big.NewInt(2_000_000_000), // 2 gwei, well above the dev miner's minimum of 1
		Gas:      gas,
		To:       to,
		Value:    big.NewInt(0),
		Data:     data,
	})
	signer := types.NewEIP155Signer(big.NewInt(chainID))
	signed, err := types.SignTx(tx, signer, key)
	if err != nil {
		log.Fatal(err)
	}
	raw, err := signed.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	return raw
}

func loadKeys(path string) []keyEntry {
	raw, err := readFile(path)
	if err != nil {
		log.Fatal(err)
	}
	var keys []keyEntry
	if err := json.Unmarshal(raw, &keys); err != nil {
		log.Fatal(err)
	}
	return keys
}

type contractArtifact struct {
	Bin        []byte
	BinRuntime []byte
	ABIRaw     json.RawMessage
	Selectors  map[string][]byte // function name -> 4-byte selector
}

func loadContract(path string) contractArtifact {
	raw, err := readFile(path)
	if err != nil {
		log.Fatal(err)
	}
	var doc struct {
		Contracts map[string]struct {
			Bin        string          `json:"bin"`
			BinRuntime string          `json:"bin-runtime"`
			ABI        json.RawMessage `json:"abi"`
		} `json:"contracts"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		log.Fatal(err)
	}
	var c struct {
		Bin, BinRuntime string
		ABI             json.RawMessage
	}
	for _, v := range doc.Contracts {
		c.Bin, c.BinRuntime, c.ABI = v.Bin, v.BinRuntime, v.ABI
		break
	}
	bin, _ := hex.DecodeString(c.Bin)
	binRuntime, _ := hex.DecodeString(c.BinRuntime)

	var abiEntries []struct {
		Type   string `json:"type"`
		Name   string `json:"name"`
		Inputs []struct {
			Type string `json:"type"`
		} `json:"inputs"`
	}
	if err := json.Unmarshal(c.ABI, &abiEntries); err != nil {
		log.Fatal(err)
	}
	selectors := map[string][]byte{}
	for _, e := range abiEntries {
		if e.Type != "function" {
			continue
		}
		types := make([]string, len(e.Inputs))
		for i, in := range e.Inputs {
			types[i] = in.Type
		}
		sig := fmt.Sprintf("%s(%s)", e.Name, strings.Join(types, ","))
		selectors[e.Name] = crypto.Keccak256([]byte(sig))[:4]
	}
	return contractArtifact{Bin: bin, BinRuntime: binRuntime, ABIRaw: c.ABI, Selectors: selectors}
}

// packUploadModel / packAggregateModel / packDownloadModel hand-encode the
// three calls this driver needs. A full ABI encoder (accounts/abi) is
// available in this module, but these three signatures are simple enough
// (uint256, uint64, bytes / uint256 alone) that hand-packing avoids pulling
// the encoder's reflection-based API into a throwaway driver -- worth doing
// differently from the Go test files, which DO use accounts/abi, because
// those tests need Unpack (decoding return values), which is genuinely
// tedious to hand-roll; this driver only needs Pack (encoding calls) and gets
// its results from event logs instead.
func packUint256Call(selector []byte, round uint64) []byte {
	buf := make([]byte, 4+32)
	copy(buf, selector)
	new(big.Int).SetUint64(round).FillBytes(buf[4+32-8:])
	return buf
}

func packUploadModel(selector []byte, round, weight uint64, tensor []byte) []byte {
	// uploadModel(uint256 round, uint64 weight, bytes tensor)
	head := make([]byte, 4+32+32+32) // selector, round, weight, offset-to-bytes
	copy(head, selector)
	new(big.Int).SetUint64(round).FillBytes(head[4 : 4+32])
	new(big.Int).SetUint64(weight).FillBytes(head[4+32 : 4+64])
	new(big.Int).SetUint64(96).FillBytes(head[4+64 : 4+96]) // offset: 3 head words after selector
	lenWord := make([]byte, 32)
	new(big.Int).SetUint64(uint64(len(tensor))).FillBytes(lenWord)
	padded := padTo32(tensor)
	out := append([]byte{}, head...)
	out = append(out, lenWord...)
	out = append(out, padded...)
	return out
}

func padTo32(b []byte) []byte {
	rem := len(b) % 32
	if rem == 0 {
		return b
	}
	return append(append([]byte{}, b...), make([]byte, 32-rem)...)
}

func readFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

func main() {
	rpcPorts := flag.String("rpc", "8600,8601,8602,8603,8604", "comma-separated HTTP ports, one per independent node")
	rounds := flag.Int("rounds", 20, "number of aggregation rounds")
	keysPath := flag.String("keys", "keys.json", "path to keys.json")
	contractPath := flag.String("contract", "../contracts/DFRLAggregator.json", "path to solc combined-json output")
	nClients := flag.Int("clients", 5, "number of client (upload) accounts, must be <= len(keys)-1")
	nParams := flag.Int("params", 32, "parameters per client tensor (kept small: RPC gas cap is 50,000,000)")
	flag.Parse()

	keys := loadKeys(*keysPath)
	if len(keys) < *nClients+1 {
		log.Fatalf("need at least %d keys, got %d", *nClients+1, len(keys))
	}
	art := loadContract(*contractPath)

	var nodes []*node
	for _, p := range strings.Split(*rpcPorts, ",") {
		nodes = append(nodes, &node{url: "http://127.0.0.1:" + strings.TrimSpace(p), nonce: map[common.Address]uint64{}})
	}
	log.Printf("driving %d independent nodes, %d rounds, %d clients x %d params", len(nodes), *rounds, *nClients, *nParams)

	deployerKey := mustDecodeKey(keys[*nClients].PrivKey)

	// Deploy: identical raw tx to every node. The resulting contract address
	// is deterministic (keccak256(rlp(sender,nonce))) and depends only on the
	// sender+nonce, not on any node-specific state, so it must match across
	// all five once mined.
	deployRaw := buildTx(deployerKey, 0, nil, 3_000_000, art.Bin)
	var contractAddr common.Address
	for i, n := range nodes {
		hash, err := n.sendRaw(deployRaw)
		if err != nil {
			log.Fatalf("node %d deploy: %v", i, err)
		}
		r, err := n.waitReceipt(hash, 15*time.Second)
		if err != nil {
			log.Fatalf("node %d deploy receipt: %v", i, err)
		}
		if r.Status != "0x1" {
			log.Fatalf("node %d deploy reverted", i)
		}
		addr := common.HexToAddress(r.ContractAddress)
		if i == 0 {
			contractAddr = addr
		} else if addr != contractAddr {
			log.Fatalf("DETERMINISM VIOLATION at deploy: node %d contract address %s != node 0's %s", i, addr.Hex(), contractAddr.Hex())
		}
	}
	log.Printf("deployed at %s on all %d nodes, identical address", contractAddr.Hex(), len(nodes))

	clientKeys := make([]*ecdsa.PrivateKey, *nClients)
	clientAddrs := make([]common.Address, *nClients)
	for i := 0; i < *nClients; i++ {
		clientKeys[i] = mustDecodeKey(keys[i].PrivKey)
		clientAddrs[i] = crypto.PubkeyToAddress(clientKeys[i].PublicKey)
	}

	clientNonce := make([]uint64, *nClients)
	aggregatorNonce := uint64(1) // 0 was the deployment
	mismatches := 0

	for round := 0; round < *rounds; round++ {
		weights := make([]int64, *nClients)
		for i := range weights {
			weights[i] = int64(i + 1)
		}
		ups := dfrlagg.GenUpdates(uint64(20260824+round), *nClients, *nParams, weights, 1<<10, dfrlagg.DefaultF)

		var uploadHashesPerNode [][]common.Hash
		for _, n := range nodes {
			var hashes []common.Hash
			for i := 0; i < *nClients; i++ {
				enc, err := dfrlagg.Encode(ups[i].Tensor)
				if err != nil {
					log.Fatal(err)
				}
				data := packUploadModel(art.Selectors["uploadModel"], uint64(round), uint64(ups[i].Weight), enc)
				raw := buildTx(clientKeys[i], clientNonce[i], &contractAddr, 3_000_000, data)
				h, err := n.sendRaw(raw)
				if err != nil {
					log.Fatalf("round %d upload %d: %v", round, i, err)
				}
				hashes = append(hashes, h)
			}
			uploadHashesPerNode = append(uploadHashesPerNode, hashes)
		}
		for i := range clientNonce {
			clientNonce[i]++
		}
		for ni, n := range nodes {
			for _, h := range uploadHashesPerNode[ni] {
				if _, err := n.waitReceipt(h, 15*time.Second); err != nil {
					log.Fatalf("round %d node %d upload receipt: %v", round, ni, err)
				}
			}
		}

		aggData := packUint256Call(art.Selectors["aggregateModel"], uint64(round))
		aggRaw := buildTx(deployerKey, aggregatorNonce, &contractAddr, 15_000_000, aggData) // node's RPC gas cap (--rpc.gascap) is 16,777,216 by default
		aggregatorNonce++

		var roots []common.Hash     // from each node's own transaction receipt log
		var callRoots []common.Hash // from an independent eth_call read of contract storage
		var stateRoots []common.Hash
		var blockNums []string
		for ni, n := range nodes {
			h, err := n.sendRaw(aggRaw)
			if err != nil {
				log.Fatalf("round %d node %d aggregate send: %v", round, ni, err)
			}
			r, err := n.waitReceipt(h, 15*time.Second)
			if err != nil {
				log.Fatalf("round %d node %d aggregate receipt: %v", round, ni, err)
			}
			if r.Status != "0x1" {
				log.Fatalf("round %d node %d aggregateModel reverted", round, ni)
			}
			if len(r.Logs) == 0 {
				log.Fatalf("round %d node %d: no ModelAggregated log", round, ni)
			}
			// ModelAggregated(round indexed, root, resultLen, nSubmitters) --
			// root is the first NON-indexed field, so it is in Data[0:32].
			dataHex := strings.TrimPrefix(r.Logs[0].Data, "0x")
			dataBytes, _ := hex.DecodeString(dataHex)
			emittedRoot := common.BytesToHash(dataBytes[0:32])
			roots = append(roots, emittedRoot)
			blockNums = append(blockNums, r.BlockNumber)

			cr, err := n.callCommittedRoot(contractAddr, art.Selectors["committedRoot"], uint64(round))
			if err != nil {
				log.Fatalf("round %d node %d committedRoot call: %v", round, ni, err)
			}
			callRoots = append(callRoots, cr)

			sr, err := n.blockStateRoot(r.BlockNumber)
			if err != nil {
				log.Fatalf("round %d node %d state root: %v", round, ni, err)
			}
			stateRoots = append(stateRoots, sr)
		}

		// rootsOK / callOK are the real X6 claim: the DFRL contract's own
		// commitment, produced by five independently-executing node
		// processes, is byte-identical -- checked two ways (event log and an
		// independent eth_call state read against each node's own chain tip).
		//
		// stateOK (whole-BLOCK state root) is NOT part of that claim and is
		// expected to differ: this genesis activates Cancun/Prague/Osaka/
		// Bogota from block 0, and every one of those forks writes its own
		// system contract on EVERY block, including empty ones (e.g. EIP-4788's
		// beacon-root ring buffer, keyed by block timestamp). These five nodes
		// have no p2p between them and self-mine on independent wall-clock
		// timers, so they produce different NUMBERS of empty blocks between
		// rounds purely from OS scheduling jitter -- which changes the global
		// state root through machinery that has nothing to do with DFRL. It is
		// logged for visibility, never as a pass/fail signal.
		rootsOK, callOK, stateOK := true, true, true
		for i := 1; i < len(roots); i++ {
			if roots[i] != roots[0] {
				rootsOK = false
			}
			if callRoots[i] != callRoots[0] {
				callOK = false
			}
			if stateRoots[i] != stateRoots[0] {
				stateOK = false
			}
		}
		if roots[0] != callRoots[0] {
			log.Fatalf("round %d: receipt-log root %s != eth_call committedRoot %s on node 0 -- internal inconsistency, not a cross-node one",
				round, roots[0].Hex(), callRoots[0].Hex())
		}

		want, err := dfrlagg.FedAvg(ups)
		if err != nil {
			log.Fatal(err)
		}
		wantEnc, err := dfrlagg.Encode(want)
		if err != nil {
			log.Fatal(err)
		}
		// Keccak256, matching DFRLAggregator.sol's `keccak256(result)` --
		// NOT dfrlagg.StateRoot, which is SHA3-256 (FIPS-202) and would
		// disagree with the contract's commitment on hash function alone even
		// if the underlying bytes are identical (see codec.go's StateRoot doc
		// comment on this exact distinction).
		wantRoot := crypto.Keccak256Hash(wantEnc)
		wantOK := roots[0] == wantRoot

		status := "OK"
		if !rootsOK || !callOK || !wantOK {
			status = "MISMATCH"
			mismatches++
		}
		log.Printf("round %3d block=%s commitRoot=%s roots_match=%v call_match=%v vs_direct_dfrlagg=%v (whole_block_state_match=%v, expected false -- see comment) [%s]",
			round, blockNums[0], roots[0].Hex()[:10], rootsOK, callOK, wantOK, stateOK, status)
	}

	log.Printf("=== %d/%d rounds mismatched ===", mismatches, *rounds)
	if mismatches > 0 {
		log.Fatal("DETERMINISM VIOLATION DETECTED")
	}
}
