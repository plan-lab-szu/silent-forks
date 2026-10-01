package dfrlagg

import (
	"fmt"
	"sort"
	"testing"
)

// Why the agreeing coordinates matter.
//
// The nine client-triggered trials in the manuscript are built at coordinates
// where all six strategies agree on the unmodified input, because a client can
// only be said to cause divergence where there was none. The scope survey shows
// such coordinates are about a quarter of the model at width 1 and vanish by
// width 16. This test asks what those coordinates ARE: if they agree only
// because the ten client scalars there are identical or near-identical -- MNIST
// border pixels never receive gradient, so their updates stay at zero -- then
// the regime the client attack inhabits is the degenerate part of the model,
// and that must be said plainly rather than left for a reader to discover.
//
// Exploratory: prints a comparison, asserts only internal consistency.
func TestNatureOfAgreeingCoordinates(t *testing.T) {
	for _, r := range []int{0, 2, 4} {
		theta, weights := loadRealRound(t, r)
		p := len(theta[0])

		var agreeSpread, splitSpread []float64
		agreeAllZero, agreeIdentical := 0, 0

		for off := 0; off < p; off++ {
			col := make([]float32, len(theta))
			for i := range theta {
				col[i] = theta[i][off]
			}
			lo, hi := col[0], col[0]
			allZero, identical := true, true
			for _, v := range col {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
				if v != 0 {
					allZero = false
				}
				if v != col[0] {
					identical = false
				}
			}
			spread := float64(hi) - float64(lo)

			n, _ := distinctRoots(sliceTheta(theta, off, 1), weights)
			if n == 1 {
				agreeSpread = append(agreeSpread, spread)
				if allZero {
					agreeAllZero++
				}
				if identical {
					agreeIdentical++
				}
			} else {
				splitSpread = append(splitSpread, spread)
			}
		}

		if len(agreeSpread)+len(splitSpread) != p {
			t.Fatalf("round %d: classified %d+%d coordinates, expected %d",
				r, len(agreeSpread), len(splitSpread), p)
		}

		t.Logf("round %d: %d agreeing / %d splitting coordinates (P=%d)",
			r, len(agreeSpread), len(splitSpread), p)
		t.Logf("    of the agreeing ones: %d have all ten client scalars exactly zero, "+
			"%d have all ten identical (%.1f%% of agreeing)",
			agreeAllZero, agreeIdentical,
			100*float64(agreeIdentical)/float64(len(agreeSpread)))
		t.Logf("    client-scalar spread (max-min) at agreeing  coordinates: %s", quartiles(agreeSpread))
		t.Logf("    client-scalar spread (max-min) at splitting coordinates: %s", quartiles(splitSpread))
	}
}

func quartiles(xs []float64) string {
	if len(xs) == 0 {
		return "n=0"
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	q := func(f float64) float64 {
		return s[int(f*float64(len(s)-1))]
	}
	zeros, mean := 0, 0.0
	for _, v := range s {
		if v == 0 {
			zeros++
		}
		mean += v
	}
	mean /= float64(len(s))
	return fmt.Sprintf("n=%d  zero-spread=%d (%.1f%%)  median=%.3g  q75=%.3g  max=%.3g  mean=%.3g",
		len(s), zeros, 100*float64(zeros)/float64(len(s)),
		q(0.50), q(0.75), s[len(s)-1], mean)
}
