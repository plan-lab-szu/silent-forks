// Copyright 2026 The DT-DFRL authors.

package vm_test

import (
	"encoding/binary"
	"fmt"
	"os"
	"testing"

	dfrlagg "dfrl-agg"
	"github.com/ethereum/go-ethereum/core/vm"
)

// TestFedAvgGasBenchmark measures wall-clock time for pc.Run(FedAvg) across an
// (N, |Theta|) grid and prints a CSV row per point. It is a Test, not a
// Benchmark, specifically so the (n, p) grid is walked exactly once per `go
// test` invocation -- each cell calls testing.Benchmark() internally to get a
// stable per-op timing, rather than letting `go test -bench` replay the whole
// grid under an outer b.N.
//
//	go test ./core/vm -run TestFedAvgGasBenchmark -v > gas_bench.txt
//
// This is a MEASUREMENT, not a calibration: it times Run() on whatever
// machine runs `go test`, which as of this writing is a single Windows
// development workstation, not a validator node from a real testbed (Phase
// 3c does not exist yet). The dfrlGasPerParam / dfrlGasPerNParam constants in
// contracts_dfrl.go are a provisional affine fit to this data, disclosed as
// such per R2-10's requirement that gasPerParam not be an unexplained
// constant. Re-run and refit once Phase 3c's testbed nodes exist; until then,
// treat the resulting gas schedule as "internally consistent and measured",
// not "validator-representative".
func TestFedAvgGasBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("gas benchmark grid skipped in -short mode")
	}
	pc := vm.PrecompiledContractsDFRL[vm.DFRLFedAvgAddress]
	ns := []int{2, 4, 8, 16, 32}
	ps := []int{16, 128, 1024, 8192, 65536, 250000}

	fmt.Println("n,params,input_bytes,ns_per_op")
	for _, n := range ns {
		for _, p := range ps {
			weights := make([]int64, n)
			for i := range weights {
				weights[i] = int64(i + 1)
			}
			ups := dfrlagg.GenUpdates(uint64(n*100003+p), n, p, weights, 1<<17, dfrlagg.DefaultF)
			buf := make([]byte, 4)
			binary.BigEndian.PutUint32(buf, uint32(n))
			for _, u := range ups {
				enc, err := dfrlagg.Encode(u.Tensor)
				if err != nil {
					t.Fatal(err)
				}
				entry := make([]byte, 12+len(enc))
				binary.BigEndian.PutUint64(entry, uint64(u.Weight))
				binary.BigEndian.PutUint32(entry[8:], uint32(len(enc)))
				copy(entry[12:], enc)
				buf = append(buf, entry...)
			}

			name := fmt.Sprintf("n=%d/p=%d", n, p)
			result := testing.Benchmark(func(bb *testing.B) {
				for i := 0; i < bb.N; i++ {
					if _, err := pc.Run(buf); err != nil {
						bb.Fatal(err)
					}
				}
			})
			t.Logf("%s: %s", name, result.String())
			nsPerOp := result.NsPerOp()
			fmt.Printf("%d,%d,%d,%d\n", n, p, len(buf), nsPerOp)
			_ = name
		}
	}
	if os.Getenv("DFRL_BENCH_QUIET") == "" {
		fmt.Println("# fit: ns_per_op ~= C_base_ns + mu1_ns*p + mu2_ns*n*p")
		fmt.Println("# convert to gas with an explicit ns-per-gas constant; do not skip disclosing that constant")
	}
}
