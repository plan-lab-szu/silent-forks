//go:build e3lite_vulnerable && !e3lite_reverse

// Copyright 2026 The DT-DFRL authors.
//
// FORWARD accumulation order for the e3lite vulnerable aggregation. This is
// the default variant: a binary built with `-tags e3lite_vulnerable` gets
// this file, and one built with `-tags "e3lite_vulnerable e3lite_reverse"`
// gets the reverse variant instead.
//
// Splitting the order across build tags rather than a runtime flag is
// deliberate. The claim the Engine API experiment makes is that two
// INDEPENDENTLY BUILT execution clients disagree on a payload; a flag one
// process could be started with differently would let a reader object that
// the divergence was configured rather than compiled. Two binaries, two
// conforming summation orders, no switch.

package vm

// dfrlVulnAccumulate sums the weighted contributions in client submission
// order, the naive loop. See contracts_dfrl_vulnerable_e3lite.go's Run doc
// comment for why the product is left unwrapped.
func dfrlVulnAccumulate(entries []dfrlVulnEntry, params int) ([]float32, float32) {
	acc := make([]float32, params)
	var wsum float32
	for _, e := range entries {
		wsum += e.weight
		for j := 0; j < params; j++ {
			acc[j] += e.weight * e.theta[j]
		}
	}
	return acc, wsum
}

// dfrlVulnOrderName identifies the compiled-in order, so a running node can be
// asked which variant it is without inspecting the binary.
const dfrlVulnOrderName = "seq-fwd"
