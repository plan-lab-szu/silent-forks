// Copyright 2026 The DT-DFRL authors.

package vm_test

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	dfrlagg "dfrl-agg"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/core/vm/runtime"
)

// buildFedAvgOpcodeMinimal is a SPEED-COMPARISON-ONLY variant of
// buildFedAvgOpcodeCaller: it RETURNs directly from the opcode's output
// buffer with a caller-known exact length, instead of MCOPYing the whole
// declared output capacity into a separate return region first.
// buildFedAvgOpcodeCaller's extra MCOPY exists so its generic-purpose caller
// doesn't need to know the exact output length up front; that convenience
// copy is NOT part of what the fused-opcode design is claimed to save
// (REWRITE_PLAN Phase 3b: eliminating the CALL frame), so including it in a
// precompile-vs-opcode latency comparison would be measuring the wrong thing.
// This variant and buildStaticcallForwarder do matched amounts of
// post-dispatch work (one CALLDATACOPY in, one direct RETURN out, no other
// memory copies), so the difference between them isolates dispatch overhead.
func buildFedAvgOpcodeMinimal(n uint64, exactOutLen uint64) []byte {
	var b []byte
	push := func(v uint64) {
		if v == 0 {
			b = append(b, byte(vm.PUSH1), 0)
			return
		}
		var buf [8]byte
		for i := 0; i < 8; i++ {
			buf[i] = byte(v >> (8 * (7 - i)))
		}
		start := 0
		for start < 7 && buf[start] == 0 {
			start++
		}
		b = append(b, byte(vm.PUSH1)+byte(7-start))
		b = append(b, buf[start:]...)
	}
	op := func(o vm.OpCode) { b = append(b, byte(o)) }
	const outOffset = 0x2000

	op(vm.CALLDATASIZE)
	push(0)
	push(0)
	op(vm.CALLDATACOPY)

	// FEDAVG(inOffset=0, inSize=calldatasize, n, outOffset, outSize=exactOutLen)
	push(exactOutLen)
	push(outOffset)
	push(n)
	op(vm.CALLDATASIZE)
	push(0)
	op(vm.FEDAVG)
	op(vm.POP) // discard success flag; this test assumes valid input

	push(exactOutLen)
	push(outOffset)
	op(vm.RETURN)
	return b
}

// TestX9PrecompileVsOpcodeLatency is the STATISTICALLY DISCIPLINED redo of
// X9's empirical half (REWRITE_PLAN Phase 3c): precompile vs fused opcode,
// same aggregation, three dispatch paths measured on identical input:
//
//  1. pc.Run(input) directly -- the aggregation cost alone, no EVM at all.
//  2. EVM executing a contract that STATICCALLs the precompile -- (1) plus
//     account/precompile dispatch and CALL's gas-stipend accounting.
//  3. EVM executing a contract that runs the FEDAVG opcode inline -- (1) plus
//     opcode dispatch only, no CALL frame.
//
// An EARLIER attempt at this used one single testing.Benchmark call per path
// per (n,p) point and got a sign that FLIPPED across points (opcode ~10%
// faster, then ~8% slower, then ~46% slower) -- not a usable signal, and
// exactly the "single run, no seeds, no CI" methodology gap R1-5b flagged
// elsewhere. This version fixes that the way fl-workload/campaign.py already
// does for the learning-side numbers: many repeated trials, mean +/- 95% CI,
// and an explicit statistical test, with an honest null result reported if
// the CI straddles zero rather than a forced direction.
//
// Design choices worth recording:
//   - Trials are INTERLEAVED in randomized order within each round (direct /
//     precompile / opcode all measured once per round, order shuffled per
//     round) rather than run as three back-to-back blocks. This is the
//     mitigation for the suspected noise source (GC pauses, OS scheduling
//     bursts on a shared, non-dedicated Windows workstation): if a burst of
//     background load hits during round k, interleaving means it has an
//     equal chance of landing on any of the three paths, instead of biasing
//     whichever path happened to run in that time window under a
//     block design.
//   - Per-round timing uses a manually calibrated fixed iteration count
//     (not testing.Benchmark's internal auto-calibration) so every round
//     does comparable wall-clock work; only elapsed/iterations is recorded.
//   - This test only PRODUCES the raw samples (JSON) and a same-process
//     rough mean for a quick eyeball. The authoritative CI and paired
//     significance test are computed by analyze_x9.py, mirroring
//     campaign.py's mean_ci/paired so this experiment and the FL-side
//     numbers are held to the same statistical standard.
func TestX9PrecompileVsOpcodeLatency(t *testing.T) {
	if testing.Short() {
		t.Skip("X9 latency comparison skipped in -short mode")
	}
	// 75, not 25: the first 25-round attempt gave a nominally significant
	// result at a DIFFERENT payload point on each of 3 independent reruns
	// (see HANDBOOK.md's X9 correction record) -- classic false-positive
	// churn from an underpowered sample, not a real, replicable effect.
	// Pooling those 3 runs to 75 rounds gave a stable answer that no longer
	// flips between reruns; 75 is kept as the new floor so a future rerun
	// does not silently regress to the underpowered sample size.
	const rounds = 75
	for _, dims := range [][2]int{{8, 1024}, {2, 16}, {16, 8192}} {
		n, p := dims[0], dims[1]
		t.Run(fmt.Sprintf("n=%d/p=%d", n, p), func(t *testing.T) {
			x9Case(t, n, p, rounds)
		})
	}
}

// timeIterations runs fn exactly iters times back-to-back inside a single
// timed window and returns nanoseconds per call. A single time.Now()/Since
// pair around the whole batch (rather than timing each call individually)
// keeps per-call timer-read overhead from polluting fast paths.
func timeIterations(t *testing.T, iters int, fn func()) float64 {
	t.Helper()
	start := time.Now()
	for i := 0; i < iters; i++ {
		fn()
	}
	elapsed := time.Since(start)
	return float64(elapsed.Nanoseconds()) / float64(iters)
}

// calibrateIterations picks an iteration count so one call to
// timeIterations takes roughly targetPerRound. It times a batch of
// calibrationBatch calls (not a single call) because a single call to a fast
// path (e.g. direct pc.Run on a tiny tensor) can complete in less than the
// OS timer's effective resolution, making a one-call measurement read as
// zero and collapsing the requested iteration count to a useless minimum.
// A floor of minIters guards the same failure mode after scaling: even if
// the batch measurement is itself noisy, per-round timing always sums
// enough calls to be comfortably above timer resolution.
func calibrateIterations(t *testing.T, fn func(), targetPerRound time.Duration) int {
	t.Helper()
	const calibrationBatch = 200
	const minIters = 200
	const maxIters = 20000
	start := time.Now()
	for i := 0; i < calibrationBatch; i++ {
		fn()
	}
	batch := time.Since(start)
	if batch <= 0 {
		return maxIters
	}
	one := batch / calibrationBatch
	iters := int(targetPerRound / one)
	if iters < minIters {
		iters = minIters
	}
	if iters > maxIters {
		iters = maxIters
	}
	return iters
}

type x9Sample struct {
	NDim   int       `json:"n"`
	PDim   int       `json:"p"`
	Rounds int       `json:"rounds"`
	Direct []float64 `json:"direct_ns"`
	ViaPre []float64 `json:"via_precompile_ns"`
	ViaOp  []float64 `json:"via_opcode_ns"`
}

func x9Case(t *testing.T, n, p, rounds int) {
	weights := make([]int64, n)
	for i := range weights {
		weights[i] = int64(i + 1)
	}
	_, tensors := genUpdates(t, 2026, n, p, weights, 1<<14)
	input := encodeDFRLInput(t, weights, tensors)

	pc := vm.PrecompiledContractsDFRL[vm.DFRLFedAvgAddress]
	directFn := func() {
		if _, err := pc.Run(input); err != nil {
			t.Fatal(err)
		}
	}

	staticcallCode := buildStaticcallForwarder(vm.DFRLFedAvgAddress)
	cfg := dfrlConfig()
	precompileFn := func() {
		if _, _, err := runtime.Execute(staticcallCode, input, &runtime.Config{
			ChainConfig: cfg, GasLimit: 50_000_000,
		}); err != nil {
			t.Fatal(err)
		}
	}

	want, err := dfrlagg.FedAvg(mustUpdates(t, weights, tensors))
	if err != nil {
		t.Fatal(err)
	}
	wantEnc, err := dfrlagg.Encode(want)
	if err != nil {
		t.Fatal(err)
	}
	opcodeCode := buildFedAvgOpcodeMinimal(uint64(n), uint64(len(wantEnc)))
	opcodeFn := func() {
		if _, _, err := runtime.Execute(opcodeCode, input, &runtime.Config{
			ChainConfig: cfg, GasLimit: 50_000_000,
		}); err != nil {
			t.Fatal(err)
		}
	}

	// Warm up each path a few times before calibration/measurement so JIT-less
	// but allocator/GC steady-state effects (first-touch page faults, etc.)
	// don't land inside round 1's sample specifically.
	for i := 0; i < 3; i++ {
		directFn()
		precompileFn()
		opcodeFn()
	}

	const targetPerRound = 20 * time.Millisecond
	itersDirect := calibrateIterations(t, directFn, targetPerRound)
	itersPre := calibrateIterations(t, precompileFn, targetPerRound)
	itersOp := calibrateIterations(t, opcodeFn, targetPerRound)

	sample := x9Sample{NDim: n, PDim: p, Rounds: rounds}
	rng := rand.New(rand.NewSource(int64(1000*n + p)))
	for r := 0; r < rounds; r++ {
		order := rng.Perm(3)
		var dNs, pNs, oNs float64
		for _, idx := range order {
			switch idx {
			case 0:
				dNs = timeIterations(t, itersDirect, directFn)
			case 1:
				pNs = timeIterations(t, itersPre, precompileFn)
			case 2:
				oNs = timeIterations(t, itersOp, opcodeFn)
			}
		}
		sample.Direct = append(sample.Direct, dNs)
		sample.ViaPre = append(sample.ViaPre, pNs)
		sample.ViaOp = append(sample.ViaOp, oNs)
	}

	mean := func(xs []float64) float64 {
		s := 0.0
		for _, x := range xs {
			s += x
		}
		return s / float64(len(xs))
	}
	dMean, pMean, oMean := mean(sample.Direct), mean(sample.ViaPre), mean(sample.ViaOp)
	t.Logf("  (rough, same-process means over %d interleaved rounds -- see analyze_x9.py for the real CI/test)", rounds)
	t.Logf("  direct pc.Run:                %10.0f ns/op", dMean)
	t.Logf("  EVM + STATICCALL(precompile): %10.0f ns/op  (dispatch overhead: %.0f ns)", pMean, pMean-dMean)
	t.Logf("  EVM + FEDAVG opcode:          %10.0f ns/op  (dispatch overhead: %.0f ns)", oMean, oMean-dMean)

	resultsDir := filepath.Join("x9_results")
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		t.Fatalf("mkdir x9_results: %v", err)
	}
	outPath := filepath.Join(resultsDir, fmt.Sprintf("n%d_p%d.json", n, p))
	f, err := os.Create(outPath)
	if err != nil {
		t.Fatalf("create %s: %v", outPath, err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(sample); err != nil {
		t.Fatalf("write %s: %v", outPath, err)
	}
	t.Logf("  wrote %d raw samples/path to %s", rounds, outPath)
}
