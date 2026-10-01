//go:build e3lite_vulnerable

package vm

import (
	"encoding/binary"
	"encoding/hex"
	"math"
	"runtime"
	"testing"

	"crypto/sha3"
)

func encodeDFRLVulnInput(weights []uint64, theta [][]float32) []byte {
	n := len(weights)
	p := len(theta[0])
	buf := make([]byte, 0, 4+n*(12+p*4))
	hdr := make([]byte, 4)
	binary.BigEndian.PutUint32(hdr, uint32(n))
	buf = append(buf, hdr...)
	for i := 0; i < n; i++ {
		e := make([]byte, 12)
		binary.BigEndian.PutUint64(e, weights[i])
		binary.BigEndian.PutUint32(e[8:], uint32(p))
		buf = append(buf, e...)
		for _, v := range theta[i] {
			b := make([]byte, 4)
			binary.BigEndian.PutUint32(b, math.Float32bits(v))
			buf = append(buf, b...)
		}
	}
	return buf
}

func decodeF32s(out []byte) []float32 {
	v := make([]float32, len(out)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.BigEndian.Uint32(out[i*4:]))
	}
	return v
}

// TestVulnerablePrecompileIsReachableAndComputesWeightedAverage is a sanity
// check, not a divergence claim: on ANY architecture, the precompile must
// compute a mathematically correct weighted average within float32 rounding
// tolerance. This is the thing every honest implementation agrees on, so it
// is safe to assert.
func TestVulnerablePrecompileIsReachableAndComputesWeightedAverage(t *testing.T) {
	c := &vulnerableFloatAggPrecompile{}
	input := encodeDFRLVulnInput([]uint64{1, 1, 2}, [][]float32{{10}, {20}, {30}})
	out, err := c.Run(input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	got := decodeF32s(out)
	// (1*10 + 1*20 + 2*30)/4 = 22.5, exact in float32, no rounding ambiguity
	// possible regardless of accumulation order or FMA contraction.
	want := float32(22.5)
	if got[0] != want {
		t.Fatalf("got %v, want %v", got[0], want)
	}
	if gas := c.RequiredGas(input); gas == 0 || gas < dfrlGasBase {
		t.Fatalf("RequiredGas returned implausible value %d", gas)
	}
}

// TestVulnerablePrecompileRejectsMalformedInput mirrors the discipline
// contracts_dfrl_test.go already applies to the real precompiles: a
// consensus-reachable decoder must never panic on adversarial input.
func TestVulnerablePrecompileRejectsMalformedInput(t *testing.T) {
	c := &vulnerableFloatAggPrecompile{}
	cases := map[string][]byte{
		"empty":            {},
		"zero clients":     {0, 0, 0, 0},
		"truncated header": {0, 0, 0, 1, 0, 0},
		"truncated theta":  append([]byte{0, 0, 0, 1}, encodeDFRLVulnInput([]uint64{1}, [][]float32{{1, 2}})[4:10]...),
		"shape mismatch":   encodeDFRLVulnInputRaw(t),
	}
	for name, in := range cases {
		if _, err := c.Run(in); err == nil {
			t.Errorf("%s: expected error, got nil", name)
		}
	}
}

func encodeDFRLVulnInputRaw(t *testing.T) []byte {
	t.Helper()
	a := encodeDFRLVulnInput([]uint64{1}, [][]float32{{1, 2}})
	b := encodeDFRLVulnInput([]uint64{1}, [][]float32{{1}})[4:] // second client, different param count
	out := make([]byte, 4)
	binary.BigEndian.PutUint32(out, 2)
	out = append(out, a[4:]...)
	out = append(out, b...)
	return out
}

// TestVulnerablePrecompileReproducesC1Witness feeds the precompile the exact
// minimal counterexample dfrl-agg/minimal_counterexample_test.go found for
// seq-fwd vs fused-mul on this same class of machine (N=3, P=1,
// weights=[2,3,1], theta=[-0.1, 0.1, 2^-20]) and reports what this EVM-level
// code path produces. This does NOT assert a specific outcome -- whether the
// unguarded product contracts into an FMA is compiler/architecture-dependent
// by the same reasoning floatref.go documents -- it exists to show the
// witness is reachable through a real precompile call, not just through a Go
// test calling dfrl-agg directly, and to give CI (both amd64 and arm64
// runners) a log line to diff. See arm64-divergence.yml for the equivalent
// check against floatref.go itself.
func TestVulnerablePrecompileReproducesC1Witness(t *testing.T) {
	c := &vulnerableFloatAggPrecompile{}
	weights := []uint64{2, 3, 1}
	theta := [][]float32{{-0.1}, {0.1}, {float32(math.Pow(2, -20))}}
	input := encodeDFRLVulnInput(weights, theta)
	out, err := c.Run(input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	v := decodeF32s(out)

	// This precompile's real ABI is big-endian (matching fedAvgPrecompile /
	// robustAggPrecompile, contracts_dfrl.go), but dfrl-agg's FloatRoot
	// (floatref.go) hashes little-endian bytes. Re-hash little-endian here
	// purely so this root is directly diffable against the roots
	// minimal_counterexample_test.go already recorded for the identical
	// input on the identical machine class -- otherwise an endianness
	// mismatch would look like a computational discrepancy it is not. The
	// precompile's actual on-chain output stays big-endian; only this
	// cross-reference hash is little-endian.
	leRoot := floatRootLittleEndian(v)

	t.Logf("platform %s/%s", runtime.GOOS, runtime.GOARCH)
	t.Logf("vulnerable precompile output on the C1 minimal witness: %v", v)
	t.Logf("little-endian cross-reference root: %s", hex.EncodeToString(leRoot[:])[:16])
	t.Logf("dfrl-agg/minimal_counterexample_test.go recorded, same witness, same machine (2026-08-27): " +
		"seq-fwd=a41cf86a130a4deb, fused-mul=61a16cfce505507d (arm64; amd64 recorded no divergence for this pair)")
	beRoot := sha3Sum256(out)
	t.Logf("this file's own big-endian on-chain output root (NOT directly comparable to the above): %s",
		hex.EncodeToString(beRoot[:])[:16])
}

func floatRootLittleEndian(v []float32) [32]byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		u := math.Float32bits(f)
		b[4*i+0] = byte(u)
		b[4*i+1] = byte(u >> 8)
		b[4*i+2] = byte(u >> 16)
		b[4*i+3] = byte(u >> 24)
	}
	return sha3Sum256(b)
}

func sha3Sum256(b []byte) [32]byte {
	return sha3.Sum256(b)
}
