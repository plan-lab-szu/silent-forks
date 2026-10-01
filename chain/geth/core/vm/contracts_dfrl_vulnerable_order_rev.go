//go:build e3lite_vulnerable && e3lite_reverse

// Copyright 2026 The DT-DFRL authors.
//
// REVERSE accumulation order for the e3lite vulnerable aggregation. Selected
// by `-tags "e3lite_vulnerable e3lite_reverse"`; see the forward variant in
// contracts_dfrl_vulnerable_order_fwd.go for why the choice is a build tag.
//
// Reverse order is chosen over FMA contraction for this experiment because it
// is architecture-INDEPENDENT: the diagnosis section's compact witnesses show
// the seq-rev divergence reproducing identically on amd64 and arm64, whereas
// the FMA witness appears only on arm64. That lets the Engine API experiment
// run two differing clients on one machine, isolating the consensus behavior
// under test from any question about which host executed which node.
//
// It is also not a contrived order. An implementation that iterates a hash map
// of submissions visits clients in an order the language does not fix -- Go
// randomizes map iteration per process -- so reverse order is one of the
// orders a single unmodified implementation can itself produce between runs.

package vm

// dfrlVulnAccumulate sums the weighted contributions in reverse client
// submission order. Mathematically the same weighted average; in float32, not
// necessarily the same bytes.
func dfrlVulnAccumulate(entries []dfrlVulnEntry, params int) ([]float32, float32) {
	acc := make([]float32, params)
	var wsum float32
	for i := len(entries) - 1; i >= 0; i-- {
		e := entries[i]
		wsum += e.weight
		for j := 0; j < params; j++ {
			acc[j] += e.weight * e.theta[j]
		}
	}
	return acc, wsum
}

// dfrlVulnOrderName identifies the compiled-in order, so a running node can be
// asked which variant it is without inspecting the binary.
const dfrlVulnOrderName = "seq-rev"
