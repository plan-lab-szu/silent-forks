// Command e3lite is the E3-lite divergence experiment: deploy
// VulnerableWrapper (chain/contracts/VulnerableWrapper.sol) to two
// independently-running geth nodes -- one compiled linux/arm64, one compiled
// linux/amd64, both from the exact same -tags e3lite_vulnerable source tree
// -- send them the IDENTICAL signed transaction calling agg() on the C1
// minimal counterexample witness (dfrl-agg/minimal_counterexample_test.go's
// seq-fwd vs fused-mul case: N=3, P=1, weights=[2,3,1],
// theta=[-0.1,0.1,2^-20]), and report whether the two architectures commit
// the same state.
//
// This is deliberately NOT a rerun of testbed/driver (which drives the REAL
// DFRLAggregator across N homogeneous-architecture nodes to demonstrate
// determinism). This is its negative-control mirror: same JSON-RPC-driven
// pattern, opposite claim, against the vulnerable precompile and
// heterogeneous architectures instead.
//
//	go run ./testbed/e3lite --rpc-arm64 8700 --rpc-amd64 8701 \
//	    --keys testbed/keys.json --contract chain/contracts/VulnerableWrapper.json
package main

import (
	"bytes"
	"crypto/ecdsa"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/big"
	"net/http"
	"os"
	"strings"
	"time"

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
	name string
	url  string
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

func (n *node) callLastRoot(contract common.Address, selector []byte) (common.Hash, error) {
	callObj := map[string]string{
		"to":   contract.Hex(),
		"data": "0x" + hex.EncodeToString(selector),
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

func buildTx(key *ecdsa.PrivateKey, nonce uint64, to *common.Address, gas uint64, data []byte) []byte {
	tx := types.NewTx(&types.LegacyTx{
		Nonce:    nonce,
		GasPrice: big.NewInt(2_000_000_000),
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
	raw, err := os.ReadFile(path)
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
	Bin       []byte
	Selectors map[string][]byte
}

func loadContract(path string) contractArtifact {
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	var doc struct {
		Contracts map[string]struct {
			Bin string          `json:"bin"`
			ABI json.RawMessage `json:"abi"`
		} `json:"contracts"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		log.Fatal(err)
	}
	var c struct {
		Bin string
		ABI json.RawMessage
	}
	for _, v := range doc.Contracts {
		c.Bin, c.ABI = v.Bin, v.ABI
		break
	}
	bin, _ := hex.DecodeString(c.Bin)

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
	return contractArtifact{Bin: bin, Selectors: selectors}
}

// vulnInput builds the vulnerable precompile's raw ABI (see
// contracts_dfrl_vulnerable_e3lite.go's decodeDFRLVulnInput doc comment):
// numClients uint32, then per client (weight uint64, paramCount uint32,
// theta as big-endian float32 words).
func vulnInput(weights []uint64, theta [][]float32) []byte {
	n := len(weights)
	buf := make([]byte, 4)
	binary.BigEndian.PutUint32(buf, uint32(n))
	for i := 0; i < n; i++ {
		e := make([]byte, 12)
		binary.BigEndian.PutUint64(e, weights[i])
		binary.BigEndian.PutUint32(e[8:], uint32(len(theta[i])))
		buf = append(buf, e...)
		for _, v := range theta[i] {
			b := make([]byte, 4)
			binary.BigEndian.PutUint32(b, math.Float32bits(v))
			buf = append(buf, b...)
		}
	}
	return buf
}

// packAggCall ABI-encodes a call to agg(bytes) -- a single dynamic
// parameter, so the encoding is selector + offset(32) + length + padded data.
func packAggCall(selector []byte, data []byte) []byte {
	head := make([]byte, 4+32)
	copy(head, selector)
	new(big.Int).SetUint64(32).FillBytes(head[4 : 4+32])
	lenWord := make([]byte, 32)
	new(big.Int).SetUint64(uint64(len(data))).FillBytes(lenWord)
	out := append([]byte{}, head...)
	out = append(out, lenWord...)
	out = append(out, padTo32(data)...)
	return out
}

func padTo32(b []byte) []byte {
	rem := len(b) % 32
	if rem == 0 {
		return b
	}
	return append(append([]byte{}, b...), make([]byte, 32-rem)...)
}

func main() {
	rpcArm64 := flag.String("rpc-arm64", "8700", "HTTP port of the arm64 node")
	rpcAmd64 := flag.String("rpc-amd64", "8701", "HTTP port of the amd64 node")
	keysPath := flag.String("keys", "keys.json", "path to keys.json")
	contractPath := flag.String("contract", "../contracts/VulnerableWrapper.json", "path to solc combined-json output")
	flag.Parse()

	keys := loadKeys(*keysPath)
	art := loadContract(*contractPath)

	nodes := []*node{
		{name: "arm64", url: "http://127.0.0.1:" + *rpcArm64},
		{name: "amd64", url: "http://127.0.0.1:" + *rpcAmd64},
	}

	deployerKey := mustDecodeKey(keys[0].PrivKey)

	deployRaw := buildTx(deployerKey, 0, nil, 3_000_000, art.Bin)
	var contractAddr common.Address
	for i, n := range nodes {
		hash, err := n.sendRaw(deployRaw)
		if err != nil {
			log.Fatalf("%s deploy: %v", n.name, err)
		}
		r, err := n.waitReceipt(hash, 20*time.Second)
		if err != nil {
			log.Fatalf("%s deploy receipt: %v", n.name, err)
		}
		if r.Status != "0x1" {
			log.Fatalf("%s deploy reverted", n.name)
		}
		addr := common.HexToAddress(r.ContractAddress)
		if i == 0 {
			contractAddr = addr
		} else if addr != contractAddr {
			log.Fatalf("contract address differs: %s=%s vs node0=%s", n.name, addr.Hex(), contractAddr.Hex())
		}
	}
	log.Printf("deployed VulnerableWrapper at %s on both nodes", contractAddr.Hex())

	// The C1 minimal counterexample witness: dfrl-agg/minimal_counterexample_test.go's
	// smallest reported divergent input for seq-fwd vs fused-mul.
	weights := []uint64{2, 3, 1}
	theta := [][]float32{{-0.1}, {0.1}, {float32(math.Pow(2, -20))}}
	input := vulnInput(weights, theta)
	callData := packAggCall(art.Selectors["agg"], input)

	var roots []common.Hash
	var stateRoots []common.Hash
	var outputs [][]byte
	for _, n := range nodes {
		raw := buildTx(deployerKey, 1, &contractAddr, 3_000_000, callData)
		h, err := n.sendRaw(raw)
		if err != nil {
			log.Fatalf("%s agg send: %v", n.name, err)
		}
		r, err := n.waitReceipt(h, 20*time.Second)
		if err != nil {
			log.Fatalf("%s agg receipt: %v", n.name, err)
		}
		if r.Status != "0x1" {
			log.Fatalf("%s agg() reverted -- precompile call failed, check fork gating / build tag", n.name)
		}
		if len(r.Logs) == 0 {
			log.Fatalf("%s: no Aggregated log", n.name)
		}
		dataHex := strings.TrimPrefix(r.Logs[0].Data, "0x")
		dataBytes, _ := hex.DecodeString(dataHex)
		root := common.BytesToHash(dataBytes[0:32])
		roots = append(roots, root)
		outputs = append(outputs, dataBytes)

		cr, err := n.callLastRoot(contractAddr, art.Selectors["lastRoot"])
		if err != nil {
			log.Fatalf("%s lastRoot call: %v", n.name, err)
		}
		if cr != root {
			log.Fatalf("%s: receipt-log root %s != eth_call lastRoot %s -- internal inconsistency", n.name, root.Hex(), cr.Hex())
		}

		sr, err := n.blockStateRoot(r.BlockNumber)
		if err != nil {
			log.Fatalf("%s state root: %v", n.name, err)
		}
		stateRoots = append(stateRoots, sr)

		log.Printf("%-6s block=%s committedRoot=%s stateRoot=%s", n.name, r.BlockNumber, root.Hex(), sr.Hex())
	}

	if roots[0] == roots[1] {
		log.Printf("=== RESULT: arm64 and amd64 committed the SAME root (%s) for this witness on this run ===", roots[0].Hex())
		log.Printf("(not a contradiction of C1: whether fused-mul contracts is architecture- AND toolchain-dependent; " +
			"see contracts_dfrl_vulnerable_e3lite_test.go for the Go-level cross-check that established divergence for this exact witness)")
	} else {
		log.Printf("=== RESULT: DIVERGENCE. arm64 committedRoot=%s amd64 committedRoot=%s ===", roots[0].Hex(), roots[1].Hex())
		log.Printf("Two independently-running geth processes, identical signed transaction, identical genesis, " +
			"differing only in CPU architecture, committed DIFFERENT state for the same block. " +
			"A validator receiving the other's block via engine_newPayload would compute a different state root " +
			"than the one the block header claims, and would reject it as INVALID.")
	}
	log.Printf("raw outputs: arm64=%x amd64=%x", outputs[0], outputs[1])
}
