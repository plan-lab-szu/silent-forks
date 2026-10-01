package dfrlagg

import "testing"

// refDivRoundHalfEven is an independent reference: it enumerates the candidate
// quotients and picks the closest, breaking ties toward the even one. It shares
// no branch structure with the implementation, so agreement is evidence rather
// than tautology.
func refDivRoundHalfEven(num, den int64) int64 {
	t := num / den
	best := t
	bestDist := int64(-1)
	for _, q := range []int64{t - 1, t, t + 1} {
		d := num - q*den
		if d < 0 {
			d = -d
		}
		switch {
		case bestDist < 0 || d < bestDist:
			best, bestDist = q, d
		case d == bestDist:
			// Tie in distance: prefer the even quotient.
			if q%2 == 0 {
				best = q
			}
		}
	}
	return best
}

func TestDivRoundHalfEvenExhaustive(t *testing.T) {
	// §6 checklist: exhaustive over a small range, negatives and ties included.
	for den := int64(1); den <= 64; den++ {
		for num := int64(-4096); num <= 4096; num++ {
			got := divRoundHalfEven(num, den)
			want := refDivRoundHalfEven(num, den)
			if got != want {
				t.Fatalf("divRoundHalfEven(%d, %d) = %d, want %d", num, den, got, want)
			}
		}
	}
}

func TestDivRoundHalfEvenTies(t *testing.T) {
	cases := []struct{ num, den, want int64 }{
		{1, 2, 0},   // 0.5 -> 0 (even)
		{3, 2, 2},   // 1.5 -> 2 (even)
		{5, 2, 2},   // 2.5 -> 2 (even)
		{7, 2, 4},   // 3.5 -> 4 (even)
		{-1, 2, 0},  // -0.5 -> 0
		{-3, 2, -2}, // -1.5 -> -2
		{-5, 2, -2}, // -2.5 -> -2
		{4, 3, 1},
		{5, 3, 2},
		{-5, 3, -2},
		{0, 7, 0},
	}
	for _, c := range cases {
		if got := divRoundHalfEven(c.num, c.den); got != c.want {
			t.Errorf("divRoundHalfEven(%d,%d)=%d want %d", c.num, c.den, got, c.want)
		}
	}
}

// TestDivRoundHalfEvenNoOverflow exercises the precondition documented in
// §3.2: 2*|num mod den| must not overflow. SumBound (S < 2^29) is what buys us
// the headroom, so test at that boundary with the largest accumulator
// Proposition 1 permits.
func TestDivRoundHalfEvenNoOverflow(t *testing.T) {
	den := SumBound - 1 // largest legal divisor
	maxAcc := int64(ClientBound-1) * (WeightBound - 1) * (ParamBound - 1)
	for _, num := range []int64{maxAcc, -maxAcc, den / 2, -(den / 2), den*3 + den/2} {
		got := divRoundHalfEven(num, den)
		want := refDivRoundHalfEven(num, den)
		if got != want {
			t.Fatalf("divRoundHalfEven(%d,%d)=%d want %d", num, den, got, want)
		}
	}
}

// TestRoundingSymmetricSweepHasNoNetMagnitudeDrift contrasts the directional
// shrinkage of truncation with round-half-to-even over one symmetric residual
// sweep. It does not assume that residuals from a training run are independent
// or claim a stochastic cumulative-error rate.
func TestRoundingSymmetricSweepHasNoNetMagnitudeDrift(t *testing.T) {
	const den = 7
	const lo, hi = int64(-100000), int64(100000)
	var driftRound, driftTrunc int64
	for num := lo; num <= hi; num++ {
		driftRound += abs64(divRoundHalfEven(num, den)*den) - abs64(num)
		driftTrunc += abs64((num/den)*den) - abs64(num)
	}
	// Half-even: residuals cancel, so the total drift stays below one divisor
	// over 200001 dividends.
	if driftRound > den || driftRound < -den {
		t.Errorf("round-half-even magnitude drift = %d, expected within +/-%d", driftRound, den)
	}
	// Truncation: expected drift is -(den-1)/2 per dividend, i.e. hundreds of
	// thousands over the same range. Assert the order of magnitude so the test
	// fails if someone "simplifies" the rounding back to truncation.
	if driftTrunc > -(hi-lo)/2 {
		t.Errorf("truncation magnitude drift = %d, expected strongly negative", driftTrunc)
	}
	t.Logf("magnitude drift over %d dividends: half-even=%d truncate=%d", hi-lo+1, driftRound, driftTrunc)
}

func abs64(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}
