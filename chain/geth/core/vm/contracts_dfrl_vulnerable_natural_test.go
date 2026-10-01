//go:build e3lite_vulnerable

package vm

import (
	"encoding/binary"
	"encoding/hex"
	"math"
	"testing"
)

// TestVulnerablePrecompileNaturalInput calls the vulnerable precompile with the
// 10-client 16-coordinate natural witness and prints the raw output. Run with
// both build tags to compare:
//
//	go test -tags e3lite_vulnerable -run TestVulnerablePrecompileNaturalInput -v ./core/vm/
//	go test -tags "e3lite_vulnerable e3lite_reverse" -run TestVulnerablePrecompileNaturalInput -v ./core/vm/
func TestVulnerablePrecompileNaturalInput(t *testing.T) {
	// Same input as MandatoryVulnAgg_Natural.sol and dfrlNatTxvWitnessInput.
	inputHex := "0000000a0000000000001d1700000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc1000000000000000007d900000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000001dcc00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc1000000000000000018ed00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000001e6900000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000000c5400000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000000000000000266200000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79d000bcca02003bca20003cd0b600bc240000bc4cf0003bc10000000000000000147a00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c795400bc7a2400bcca16003bc9d8003cd02600bc242800bc46e8003bc07000000000000000130600000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000000000000000161800000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000"
	input, err := hex.DecodeString(inputHex)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("compiled order: %s", dfrlVulnOrderName)
	t.Logf("input: %d bytes (%d hex chars)", len(input), len(inputHex))

	// Decode to check the entries.
	entries, err := decodeDFRLVulnInput(input)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	t.Logf("decoded %d entries, %d params each", len(entries), len(entries[0].theta))
	for i, e := range entries {
		t.Logf("  entry %d: weight=%.0f theta[0]=%08x(%.10g) theta[9]=%08x(%.10g)",
			i, e.weight,
			math.Float32bits(e.theta[0]), e.theta[0],
			math.Float32bits(e.theta[9]), e.theta[9])
	}

	// Run the precompile.
	p := &vulnerableFloatAggPrecompile{}
	out, err := p.Run(input)
	if err != nil {
		t.Fatalf("precompile Run: %v", err)
	}
	t.Logf("output: %d bytes", len(out))

	// Print each output float32.
	for j := 0; j < len(out)/4; j++ {
		bits := binary.BigEndian.Uint32(out[j*4:])
		val := math.Float32frombits(bits)
		t.Logf("  out[%d] = %08x (%.10g)", j, bits, val)
	}
	t.Logf("output hex: %s", hex.EncodeToString(out))

	// Also manually compute to compare.
	params := len(entries[0].theta)
	acc := make([]float32, params)
	var wsum float32
	t.Logf("\n--- manual %s accumulation ---", dfrlVulnOrderName)
	if dfrlVulnOrderName == "seq-rev" {
		for i := len(entries) - 1; i >= 0; i-- {
			e := entries[i]
			wsum += e.weight
			for j := 0; j < params; j++ {
				acc[j] += e.weight * e.theta[j]
			}
		}
	} else {
		for _, e := range entries {
			wsum += e.weight
			for j := 0; j < params; j++ {
				acc[j] += e.weight * e.theta[j]
			}
		}
	}
	for j := 0; j < params; j++ {
		v := acc[j] / wsum
		bits := math.Float32bits(v)
		t.Logf("  manual[%d] = %08x (%.10g)", j, bits, v)
	}
}
