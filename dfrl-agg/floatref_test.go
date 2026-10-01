package dfrlagg

import (
	"encoding/hex"
	"runtime"
	"testing"
)

// TestFloatDivergenceReport deliberately makes no assertions about WHICH
// strategies diverge. Whether a given pair differs depends on the architecture
// (arm64 may contract a multiply-add into an FMA where amd64 does not) and on
// the toolchain, and this project was rejected once for asserting an unverified
// technical fact. So the test records the table for the CI log and asserts only
// what the specification actually guarantees: that the integer path is
// invariant. The C1 figure is produced from these logs across runners, not
// from a hard-coded expectation.
func TestFloatDivergenceReport(t *testing.T) {
	const n, p = 10, 50000
	theta := make([][]float32, n)
	weights := make([]float32, n)
	for i := 0; i < n; i++ {
		theta[i] = GenFloat32(20260823+uint64(i)*0x1000, p)
		weights[i] = float32(PaperWeights[i%len(PaperWeights)])
	}
	base := FloatAvg(theta, weights, SeqFwd)
	roots := map[string][]FloatStrategy{}
	t.Logf("platform %s/%s, N=%d P=%d", runtime.GOOS, runtime.GOARCH, n, p)
	for _, st := range AllFloatStrategies {
		v := FloatAvg(theta, weights, st)
		differs := 0
		for j := range v {
			if v[j] != base[j] {
				differs++
			}
		}
		r := FloatRoot(v)
		h := hex.EncodeToString(r[:])
		roots[h] = append(roots[h], st)
		t.Logf("  %-12s root=%s differs=%d", st, h[:16], differs)
	}
	t.Logf("%d strategies -> %d distinct state roots", len(AllFloatStrategies), len(roots))
	if len(roots) < 2 {
		t.Logf("NOTE: no divergence observed on this platform; the C1 claim rests on " +
			"the cross-library and cross-strategy evidence, so this is data, not a failure")
	}
}

// TestIntegerPathInvariantToStrategyChoice is the positive counterpart: there is
// no strategy choice to make. The same inputs quantized and aggregated as
// integers give one root, and permuting the clients does not change it.
func TestIntegerPathInvariantToStrategyChoice(t *testing.T) {
	const n, p = 10, 20000
	ups := make([]Update, n)
	for i := 0; i < n; i++ {
		var a Addr
		a[19], a[18] = byte(i+1), byte((i+1)>>8)
		tq, err := Quantize(GenFloat32(20260823+uint64(i)*0x1000, p), []uint32{p}, DefaultF)
		if err != nil {
			t.Fatal(err)
		}
		ups[i] = Update{Addr: a, Weight: PaperWeights[i%len(PaperWeights)], Tensor: tq}
	}
	want, err := StateRoot(mustFedAvg(t, ups))
	if err != nil {
		t.Fatal(err)
	}
	r := NewSplitMix64(2026)
	for i := 0; i < 200; i++ {
		got, err := StateRoot(mustFedAvg(t, shuffle(ups, r)))
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("permutation %d changed the state root", i)
		}
	}
	t.Logf("integer path root (invariant over 200 permutations): %s", hex.EncodeToString(want[:])[:16])
}
