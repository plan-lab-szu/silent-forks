// Copyright 2026 The DT-DFRL authors.

package vm_test

import (
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"testing"

	dfrlagg "dfrl-agg"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/vm/runtime"
	"github.com/holiman/uint256"
)

// TestX4RealMNISTOnChainMatchesOffChain completes X4 (REWRITE_PLAN Phase 3c
// checklist): the byte-identical on-chain-vs-off-chain result was already
// verified in TestDFRLAggregatorContract, but only on SYNTHETIC random
// tensors. This test replays REAL local updates from actual MNIST training
// (exported by fl-workload/export_x4_rounds.py, which runs the same fed-int
// arm the paper's headline accuracy numbers already come from) through the
// ACTUAL DFRLAggregator.sol contract, and checks the result against
// expected.bin -- the SAME dfrl-agg Go backend's output the Python side
// already used to advance its training loop.
//
// Two things this adds beyond the synthetic-tensor test:
//  1. A real model's quantized update is not guaranteed to exercise the same
//     code paths as a random tensor (e.g. value-range edges near quantize()'s
//     rejection boundary); this closes that gap.
//  2. |Theta|=7,850 (LogReg's real parameter count) sits between the X8 gas
//     grid's 4,096 and 16,384 points -- an out-of-sample check of the fitted
//     gas model at a value the paper's own workload actually uses, not just
//     at the grid nodes the fit was trained on.
//
// Run fl-workload/export_x4_rounds.py first if runs/x4_export is missing;
// this test skips (not fails) when the export directory isn't there, since
// it depends on a Python step outside this Go module.
func TestX4RealMNISTOnChainMatchesOffChain(t *testing.T) {
	exportDir := filepath.Join("..", "..", "..", "..", "fl-workload", "runs", "x4_export")
	manifestPath := filepath.Join(exportDir, "manifest.json")
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Skipf("no X4 export found at %s (run: python fl-workload/export_x4_rounds.py): %v", manifestPath, err)
	}
	var manifest struct {
		Clients   int   `json:"clients"`
		Rounds    int   `json:"rounds"`
		Dims      []int `json:"dims"`
		F         int   `json:"f"`
		RoundMeta []struct {
			Round   int    `json:"round"`
			Weights []int  `json:"weights"`
			Root    string `json:"root"`
			TestAcc float64 `json:"test_acc"`
		} `json:"round_meta"`
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatalf("parse manifest.json: %v", err)
	}
	t.Logf("real MNIST export: |Theta|=%v clients=%d rounds=%d", manifest.Dims, manifest.Clients, manifest.Rounds)

	art := loadDFRLAggregator(t)
	cfg := dfrlConfig()
	clientFileRe := regexp.MustCompile(`^client(\d+)_w(\d+)\.bin$`)

	var predictedP float64
	if len(manifest.Dims) == 1 {
		predictedP = float64(manifest.Dims[0])
	}

	for _, rm := range manifest.RoundMeta {
		rm := rm
		t.Run(fmt.Sprintf("round%d", rm.Round), func(t *testing.T) {
			roundDir := filepath.Join(exportDir, fmt.Sprintf("round%04d", rm.Round))
			entries, err := os.ReadDir(roundDir)
			if err != nil {
				t.Fatalf("read %s: %v", roundDir, err)
			}
			type client struct {
				idx    int
				weight uint64
				name   string
			}
			var clients []client
			for _, e := range entries {
				m := clientFileRe.FindStringSubmatch(e.Name())
				if m == nil {
					continue
				}
				idx, _ := strconv.Atoi(m[1])
				w, _ := strconv.ParseUint(m[2], 10, 64)
				clients = append(clients, client{idx: idx, weight: w, name: e.Name()})
			}
			sort.Slice(clients, func(i, j int) bool { return clients[i].idx < clients[j].idx })
			if len(clients) != manifest.Clients {
				t.Fatalf("found %d client files in %s, manifest says %d", len(clients), roundDir, manifest.Clients)
			}

			statedb, _ := state.New(common.Hash{}, state.NewDatabaseForTesting())
			contractAddr := common.HexToAddress("0x00000000000000000000000000000000c0ffee")
			statedb.CreateAccount(contractAddr)
			statedb.SetCode(contractAddr, art.BinRuntime, 0)
			round := big.NewInt(0)

			var ups []dfrlagg.Update
			for _, c := range clients {
				b, err := os.ReadFile(filepath.Join(roundDir, c.name))
				if err != nil {
					t.Fatal(err)
				}
				tensor, err := dfrlagg.Decode(b)
				if err != nil {
					t.Fatalf("decode %s: %v", c.name, err)
				}
				var a dfrlagg.Addr
				a[19] = byte(c.idx + 1)
				ups = append(ups, dfrlagg.Update{Addr: a, Weight: int64(c.weight), Tensor: tensor})

				sender := common.BytesToAddress([]byte{byte(0x10 + c.idx)})
				statedb.CreateAccount(sender)
				statedb.AddBalance(sender, uint256.NewInt(1_000_000_000_000_000_000), 0)
				input, err := art.ABI.Pack("uploadModel", round, c.weight, b)
				if err != nil {
					t.Fatal(err)
				}
				if _, _, err := runtime.Call(contractAddr, input, &runtime.Config{
					ChainConfig: cfg, Origin: sender, State: statedb, GasLimit: 1_600_000_000,
				}); err != nil {
					t.Fatalf("uploadModel(client=%d): %v", c.idx, err)
				}
			}

			aggInput, err := art.ABI.Pack("aggregateModel", round)
			if err != nil {
				t.Fatal(err)
			}
			caller := common.BytesToAddress([]byte{0x99})
			statedb.CreateAccount(caller)
			const limit = 1_600_000_000
			_, remaining, err := runtime.Call(contractAddr, aggInput, &runtime.Config{
				ChainConfig: cfg, Origin: caller, State: statedb, GasLimit: limit,
			})
			if err != nil {
				t.Fatalf("aggregateModel: %v", err)
			}
			gasUsed := limit - remaining

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

			expected, err := os.ReadFile(filepath.Join(roundDir, "expected.bin"))
			if err != nil {
				t.Fatal(err)
			}
			if string(onChainResult) != string(expected) {
				t.Fatalf("round %d: on-chain result differs from real-MNIST dfrl-agg backend output\n got=%x\nwant=%x",
					rm.Round, onChainResult, expected)
			}

			want, err := dfrlagg.FedAvg(ups)
			if err != nil {
				t.Fatal(err)
			}
			wantEnc, err := dfrlagg.Encode(want)
			if err != nil {
				t.Fatal(err)
			}
			if string(onChainResult) != string(wantEnc) {
				t.Fatalf("round %d: on-chain result differs from direct dfrlagg.FedAvg (in-process) call", rm.Round)
			}

			t.Logf("round %d: on-chain result matches real-MNIST dfrl-agg output bit-for-bit (test_acc=%.4f, |Theta|=%v, N=%d) -- aggregateModel gas=%d",
				rm.Round, rm.TestAcc, manifest.Dims, len(clients), gasUsed)

			if predictedP > 0 {
				predictedGas := 132343.07 + 5252.20469*predictedP + 678.41045*float64(len(clients))*predictedP
				relErr := (float64(gasUsed) - predictedGas) / predictedGas
				t.Logf("  X8 fitted gas model (full-storage) predicts %.0f at N=%d,|Theta|=%.0f -- actual/predicted rel diff %+.1f%% (out-of-sample check, |Theta| not in fit grid)",
					predictedGas, len(clients), predictedP, 100*relErr)
			}
		})
	}
}
