package dfrlagg

import (
	"encoding/hex"
	"math"
	"os"
	"runtime"
	"strconv"
	"testing"
)

// TestMinimalCounterexample searches for the smallest, most human-readable
// input that makes two conforming FloatAvg strategies diverge — a reduced
// witness for the C1 claim, complementary to the N=10,P=50000 demonstration
// in TestFloatDivergenceReport. That test shows the failure mode is real at
// realistic scale; this one shows it is not an artifact of scale: it
// survives shrinking to a handful of clients and a handful of parameters,
// with plain constants a reader can check by hand.
//
// Like TestFloatDivergenceReport, this makes no assertion about which
// specific example is found — divergence is architecture- and
// toolchain-dependent (see floatref.go's FusedMul doc comment) — and records
// what it found for the paper to draw from, across whatever platform runs
// it.
//
// Search space: N clients in {3,4,5}, P params in {1,2,3,4}, weights drawn
// from small positive integers (matching the real protocol's contribution
// scores s_{n,k}, which are non-negative counts), theta values drawn from a
// curated set of textbook float32 rounding-boundary constants. Search runs
// ascending in N, then ascending in P, and stops at the first cell that
// yields a divergent example.
//
// COVERAGE IS NOT UNIFORM ACROSS CELLS, and any claim drawn from this test
// must respect that. A cell holds (3*12^P)^N assignments, which ranges from
// 46,656 at (N=3,P=1) to 9.3e23 at (N=5,P=4). Each cell is therefore
// searched up to searchBudget() assignments and NO further. Only cells whose
// full size is <= that budget are exhaustive; the test logs, per cell,
// exactly how many of how many were tried and marks each EXHAUSTIVE or
// PARTIAL. A negative result ("no divergence found") is evidence of absence
// ONLY for the cells marked EXHAUSTIVE. Cells marked PARTIAL are covered by
// a structured prefix of the enumeration, not a random sample, so they
// support no absence claim at all.
//
// Set DFRL_SEARCH_BUDGET to raise the per-cell budget for a deeper
// evidence-gathering run.
func TestMinimalCounterexample(t *testing.T) {
	// Curated float32 constants a reader can recognize on sight: exact
	// integers, and the classic single-precision rounding-boundary values
	// (machine epsilon 2^-23 and its half, plus a couple of neighbours).
	// 0.1 is included because it is the standard "decimal that isn't exactly
	// representable" example and often participates in short divergent
	// witnesses.
	thetaCandidates := []float32{
		1, -1, 2, -2,
		float32(math.Pow(2, -20)),
		float32(math.Pow(2, -22)),
		float32(math.Pow(2, -23)),
		float32(math.Pow(2, -24)),
		float32(-math.Pow(2, -23)),
		float32(-math.Pow(2, -24)),
		0.1, -0.1,
	}
	weightCandidates := []float32{1, 2, 3}

	pairs := []struct {
		a, b FloatStrategy
	}{
		{SeqFwd, FusedMul},
		{SeqFwd, Widened},
		{SeqFwd, SeqRev},
	}

	type witness struct {
		n, p    int
		theta   [][]float32
		weights []float32
		a, b    FloatStrategy
		ra, rb  [32]byte
		maxDiff float64
	}

	budget := searchBudget()
	t.Logf("platform %s/%s, per-cell budget %d assignments",
		runtime.GOOS, runtime.GOARCH, budget)

	for _, pr := range pairs {
		found := (*witness)(nil)
		exhaustive := []string{}
		partial := []string{}
	searchPR:
		for n := 3; n <= 5; n++ {
			for p := 1; p <= 4; p++ {
				size := gridSize(n, p, len(thetaCandidates), len(weightCandidates))
				tried := 0
				forEachAssignment(n, p, thetaCandidates, weightCandidates,
					func(theta [][]float32, weights []float32) bool {
						tried++
						va := FloatAvg(theta, weights, pr.a)
						vb := FloatAvg(theta, weights, pr.b)
						if !equalF32(va, vb) {
							found = &witness{
								n: n, p: p,
								theta:   copyTheta(theta),
								weights: append([]float32(nil), weights...),
								a:       pr.a, b: pr.b,
								ra:      FloatRoot(va),
								rb:      FloatRoot(vb),
								maxDiff: maxAbsDiff64(va, vb),
							}
							return false
						}
						return tried < budget
					})
				cell := "N=" + strconv.Itoa(n) + ",P=" + strconv.Itoa(p)
				covered := float64(tried) >= size
				if found != nil {
					t.Logf("%s vs %s: %s searched %d assignments before hitting a witness",
						pr.a, pr.b, cell, tried)
					break searchPR
				}
				if covered {
					exhaustive = append(exhaustive, cell+" ("+strconv.Itoa(tried)+")")
				} else {
					partial = append(partial, cell+" ("+strconv.Itoa(tried)+"/"+
						strconv.FormatFloat(size, 'g', 4, 64)+")")
				}
			}
		}
		if found == nil {
			t.Logf("%s vs %s: NO DIVERGENCE FOUND", pr.a, pr.b)
			t.Logf("  EXHAUSTIVE cells (absence is established here): %v", exhaustive)
			t.Logf("  PARTIAL cells (absence is NOT established here): %v", partial)
			continue
		}
		ra := hex.EncodeToString(found.ra[:])[:16]
		rb := hex.EncodeToString(found.rb[:])[:16]
		t.Logf("%s vs %s: minimal witness N=%d P=%d", pr.a, pr.b, found.n, found.p)
		for i := range found.theta {
			t.Logf("  client %d: weight=%v theta=%v", i, found.weights[i], found.theta[i])
		}
		t.Logf("  root(%s)=%s  root(%s)=%s  max|diff|=%.3e",
			pr.a, ra, pr.b, rb, found.maxDiff)
	}
}

// searchBudget is the maximum number of assignments examined per (N,P) cell.
// The default bounds the worst-case combinatorial blowup: the largest cell
// holds 9.3e23 assignments, so an unbounded search would never terminate.
func searchBudget() int {
	if s := os.Getenv("DFRL_SEARCH_BUDGET"); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			return v
		}
	}
	return 2_000_000
}

// gridSize returns the exact number of assignments at (n,p) as a float64.
// float64 is deliberate: the true count reaches 9.3e23 at (N=5,P=4), which
// overflows int64. Computing this in int silently wraps to a negative value
// and makes a search loop bounded by it examine nothing at all, while
// reporting no error — so the count is never computed in int here.
func gridSize(n, p, nTheta, nWeight int) float64 {
	perClient := float64(nWeight) * math.Pow(float64(nTheta), float64(p))
	return math.Pow(perClient, float64(n))
}

// forEachAssignment enumerates assignments of (weight, theta-row) pairs to n
// clients, theta-row being a length-p vector whose coordinates are each drawn
// independently from thetaCandidates. It calls fn for each and stops when fn
// returns false or the space is exhausted.
//
// The enumeration is an odometer over n per-client digits rather than an
// index into a precomputed total, so it is correct for spaces far larger than
// int64 can express, and it runs on the caller's goroutine so an early stop
// cannot strand a producer. Order is lexicographic in the digit vector, which
// means any truncated prefix is a structured subset and never a random sample.
//
// fn must not retain theta or weights: both are reused across calls.
func forEachAssignment(n, p int, thetaCandidates, weightCandidates []float32, fn func(theta [][]float32, weights []float32) bool) {
	perClient := len(weightCandidates) * pow(len(thetaCandidates), p)
	if perClient <= 0 || n <= 0 {
		return
	}
	digits := make([]int, n)
	theta := make([][]float32, n)
	for i := range theta {
		theta[i] = make([]float32, p)
	}
	weights := make([]float32, n)

	decode := func(i int) {
		c := digits[i]
		weights[i] = weightCandidates[c%len(weightCandidates)]
		c /= len(weightCandidates)
		for j := 0; j < p; j++ {
			theta[i][j] = thetaCandidates[c%len(thetaCandidates)]
			c /= len(thetaCandidates)
		}
	}
	for i := 0; i < n; i++ {
		decode(i)
	}

	for {
		if !fn(theta, weights) {
			return
		}
		i := 0
		for ; i < n; i++ {
			digits[i]++
			if digits[i] < perClient {
				decode(i)
				break
			}
			digits[i] = 0
			decode(i)
		}
		if i == n {
			return // odometer wrapped: space exhausted
		}
	}
}

// pow is used only for per-client digit widths (at most 3*12^4 = 62208), which
// are far from any overflow boundary. Whole-grid sizes are computed by
// gridSize in float64 instead; see the note there.
func pow(base, exp int) int {
	r := 1
	for i := 0; i < exp; i++ {
		r *= base
	}
	return r
}

func copyTheta(theta [][]float32) [][]float32 {
	out := make([][]float32, len(theta))
	for i := range theta {
		out[i] = append([]float32(nil), theta[i]...)
	}
	return out
}

func equalF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func maxAbsDiff64(a, b []float32) float64 {
	var m float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

// TestEnumeratorCoversExactlyTheGrid pins the property that every absence
// claim drawn from TestMinimalCounterexample depends on: when allowed to run
// to completion, forEachAssignment visits every assignment in a cell exactly
// once, and gridSize predicts that count.
//
// This exists because the failure it guards against is silent. An enumerator
// bounded by a count computed in int overflows to a negative value at
// (N=4,P=4) and then visits nothing, while the search above reports "no
// divergence found" for a space it never examined — an absence of evidence
// presented as evidence of absence.
func TestEnumeratorCoversExactlyTheGrid(t *testing.T) {
	thetas := []float32{1, -1, 2}
	weights := []float32{1, 2}

	for n := 1; n <= 3; n++ {
		for p := 1; p <= 3; p++ {
			want := gridSize(n, p, len(thetas), len(weights))
			seen := make(map[string]struct{})
			count := 0
			forEachAssignment(n, p, thetas, weights,
				func(theta [][]float32, w []float32) bool {
					count++
					k := ""
					for i := range theta {
						k += strconv.FormatFloat(float64(w[i]), 'x', -1, 32) + ":"
						for _, v := range theta[i] {
							k += strconv.FormatFloat(float64(v), 'x', -1, 32) + ","
						}
						k += ";"
					}
					seen[k] = struct{}{}
					return true
				})
			if float64(count) != want {
				t.Errorf("N=%d P=%d: enumerator visited %d assignments, grid holds %.0f",
					n, p, count, want)
			}
			if len(seen) != count {
				t.Errorf("N=%d P=%d: %d visits but only %d distinct assignments",
					n, p, count, len(seen))
			}
		}
	}
}
