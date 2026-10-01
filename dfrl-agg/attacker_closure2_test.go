package dfrlagg

import (
	"math"
	"testing"
)

// TestAttackerClosure closes the threat model experimentally: every honest
// client's update is held FIXED at its real, exported value and a single client
// -- holding no validator privileges, choosing only its own submission -- picks
// a different value for itself.
//
// THE BASELINE MUST BE NON-DEGENERATE, and this is the part that is easy to get
// wrong. A coordinate at which all clients happen to hold the same value has a
// weighted average equal to that value under every summation order, so it
// "agrees" trivially and proves nothing; in this workload 397 of the 7,850
// coordinates are of exactly that kind (dead MNIST border pixels no client's
// gradient moves). This test therefore admits a window only when all N clients
// hold N DISTINCT values there and every conforming strategy still agrees: the
// deployment is then healthy by any criterion an operator could observe, and a
// subsequent split is attributable to the attacker's choice alone.
//
// PERTURBATIONS ARE MEASURED IN QUANTIZATION STEPS, NOT IN ULPs. A float32 ULP
// at the magnitude of a real update is around 3e-10, far below the protocol's
// own resolution of 2^-F = 9.5e-7 at F=20, so a witness stated in ULPs would be
// erased by any deployment that quantizes on submission and could be dismissed
// on that ground. One step is the smallest change the protocol can express at
// all, so d=1 means the attacker moved its own update by the least amount that
// is representable.
//
// Two quantities are measured:
//
//	minimal d      -- how far from its OWN honest value the attacker must move
//	                  before conforming validators disagree.
//	split density  -- what fraction of the legal values it could submit nearby
//	                  cause a split. A high fraction means the attacker does not
//	                  need to search for a witness.
func TestAttackerClosure(t *testing.T) {
	for _, round := range []int{0, 2, 4} {
		r := round
		t.Run("round"+string(rune('0'+r)), func(t *testing.T) { attackerClosureRound(t, r) })
	}
}

func attackerClosureRound(t *testing.T, round int) {
	theta, weights := loadRealRound(t, round)
	n, p := len(theta), len(theta[0])
	step := float32(1) / float32(int64(1)<<DefaultF) // 2^-F, the protocol's resolution

	const maxWindows = 3
	found := 0
	for off := 0; off < p && found < maxWindows; off++ {
		base := sliceTheta(theta, off, 1)

		// Reject degenerate windows: require every client to hold a distinct value.
		vals := map[float32]bool{}
		for _, row := range base {
			vals[row[0]] = true
		}
		if len(vals) != n {
			continue
		}
		// Require the honest population to agree unaided.
		if k, _ := distinctRoots(base, weights); k != 1 {
			continue
		}
		found++

		slot := found % n // rotate, so no result depends on one client's slot
		honest := base[slot][0]

		minD, witness, ok := minimalQuantStepAttack(base, weights, slot, honest, step)
		diverging, legal := splitDensity(base, weights, slot, honest, step, 2000)

		t.Logf("window off=%-5d honest population: %d/%d distinct values, 1 state root (healthy)",
			off, len(vals), n)
		t.Logf("    attacker = client %d, its honest value %.9g", slot, honest)
		if ok {
			k, byRoot := distinctRoots(withAttackerValue(base, slot, witness), weights)
			t.Logf("    minimal attack: d=%+d step(s) -> %.9g  =>  %d distinct roots, groups %v",
				minD, witness, k, groupNames(byRoot))
		} else {
			t.Logf("    minimal attack: none found within budget")
		}
		t.Logf("    split density: %d/%d legal nearby submissions split the validators (%.1f%%)",
			diverging, legal, 100*float64(diverging)/float64(legal))
	}
	if found == 0 {
		t.Skip("no non-degenerate honest-agreement window in this round")
	}
}

// withAttackerValue copies base and substitutes the attacker's own coordinate.
func withAttackerValue(base [][]float32, slot int, v float32) [][]float32 {
	out := make([][]float32, len(base))
	for i, row := range base {
		out[i] = append([]float32(nil), row...)
	}
	out[slot][0] = v
	return out
}

// legalUpdate reports whether the attacker could actually submit v: it must
// quantize at the protocol F without tripping the range bound of Section 2.2.
func legalUpdate(v float32) bool {
	if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
		return false
	}
	_, err := Quantize([]float32{v}, []uint32{1}, DefaultF)
	return err == nil
}

// minimalQuantStepAttack walks outward from the attacker's own honest value in
// whole quantization steps and returns the smallest signed number of steps that
// makes conforming strategies disagree.
func minimalQuantStepAttack(base [][]float32, weights []float32, slot int, honest, step float32) (int, float32, bool) {
	const budget = 4096
	for d := 1; d <= budget; d++ {
		for _, sign := range []int{+1, -1} {
			cand := honest + float32(sign*d)*step
			if !legalUpdate(cand) {
				continue
			}
			if k, _ := distinctRoots(withAttackerValue(base, slot, cand), weights); k > 1 {
				return sign * d, cand, true
			}
		}
	}
	return 0, 0, false
}

// splitDensity sweeps the attacker's legal choices symmetrically around its own
// honest value, in whole quantization steps, so every candidate stays the same
// order of magnitude as a real update and cannot be dismissed as an anomalous
// submission.
func splitDensity(base [][]float32, weights []float32, slot int, honest, step float32, halfWidth int) (diverging, legal int) {
	for d := -halfWidth; d <= halfWidth; d++ {
		if d == 0 {
			continue
		}
		cand := honest + float32(d)*step
		if !legalUpdate(cand) {
			continue
		}
		legal++
		if k, _ := distinctRoots(withAttackerValue(base, slot, cand), weights); k > 1 {
			diverging++
		}
	}
	return diverging, legal
}

func groupNames(byRoot map[[32]byte][]FloatStrategy) [][]FloatStrategy {
	out := make([][]FloatStrategy, 0, len(byRoot))
	for _, v := range byRoot {
		out = append(out, v)
	}
	return out
}
