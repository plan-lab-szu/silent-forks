//go:build e3lite_vulnerable

package vm

import (
	"encoding/binary"
	"math"
	"testing"
)

// encodeVulnInput builds the raw-float32 ABI the vulnerable precompile expects:
//
//	numClients uint32 || { weight uint64, paramCount uint32, theta float32... }
func encodeVulnInput(weights []float32, theta [][]float32) []byte {
	out := make([]byte, 4)
	binary.BigEndian.PutUint32(out, uint32(len(weights)))
	for i, w := range weights {
		var w8 [8]byte
		binary.BigEndian.PutUint64(w8[:], uint64(w))
		out = append(out, w8[:]...)
		var pc [4]byte
		binary.BigEndian.PutUint32(pc[:], uint32(len(theta[i])))
		out = append(out, pc[:]...)
		for _, v := range theta[i] {
			var b [4]byte
			binary.BigEndian.PutUint32(b[:], math.Float32bits(v))
			out = append(out, b[:]...)
		}
	}
	return out
}

// TestVulnerableOrderIsCompiledIn records which accumulation order this binary
// carries, and asserts the precompile actually runs. Two binaries built with
// and without the e3lite_reverse tag must report different names; that is the
// premise the Engine API experiment rests on, so it is worth failing loudly if
// a build ever selects neither file or both.
func TestVulnerableOrderIsCompiledIn(t *testing.T) {
	if dfrlVulnOrderName != "seq-fwd" && dfrlVulnOrderName != "seq-rev" {
		t.Fatalf("unexpected compiled-in order %q", dfrlVulnOrderName)
	}
	t.Logf("this binary aggregates in %s order", dfrlVulnOrderName)

	// The attacker-closure witness from the diagnosis section: honest values
	// fixed, one client's own value moved by a single quantization step. The
	// point here is only that the precompile evaluates it; whether the two
	// orders differ on it is measured by the cross-binary comparison below.
	weights := []float32{7447, 2009, 7628}
	theta := [][]float32{{-0.00903129578}, {0.0111236572}, {0.00770282745}}

	p := &vulnerableFloatAggPrecompile{}
	out, err := p.Run(encodeVulnInput(weights, theta))
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("expected one float32 of output, got %d bytes", len(out))
	}
	t.Logf("aggregate = 0x%x (%v)", out, math.Float32frombits(binary.BigEndian.Uint32(out)))
}
