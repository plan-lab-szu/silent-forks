package dfrlagg

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

// TestGenerateDifferentialVectors answers a coverage objection the nine curated
// golden vectors invite on their own: nine cases, however carefully chosen, are
// a thin conformance suite for a specification that claims any conforming
// implementation on any architecture agrees byte for byte.
//
// It emits a large randomized case set in the same schema as vectors.json, so
// pyref/check_vectors.py verifies it unchanged -- the independent Python
// implementation shares no code and no dependency with this package, which is
// what makes the comparison meaningful. Randomization is seeded from a fixed
// constant, so the suite is reproducible rather than flaky, and the emitted
// file records every case's parameters, so a failure is replayable without
// rerunning the generator.
//
// Every generated case is VALID by construction (Section 2.2's bounds), because
// a case that both implementations reject identically tests error handling
// rather than agreement. Malformed input is covered separately by the
// precompile's rejection tests.
//
//	go test -run TestGenerateDifferentialVectors ./...
//	python pyref/check_vectors.py vectors/differential.json
//
// Set DFRL_DIFF_CASES to change the case count (default 2000).
func TestGenerateDifferentialVectors(t *testing.T) {
	const masterSeed = 0x5F04B5 // fixed: the suite must be reproducible

	cases := 2000
	if v := os.Getenv("DFRL_DIFF_CASES"); v != "" {
		if n, err := parsePositive(v); err == nil {
			cases = n
		}
	}

	r := NewSplitMix64(masterSeed)
	out := make([]vectorCase, 0, cases)

	for i := 0; i < cases; i++ {
		c := randomCase(r, i)
		res, err := c.run()
		if err != nil {
			t.Fatalf("case %s is invalid by construction, which is a generator bug: %v", c.Name, err)
		}
		root, err := StateRoot(res)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		enc, err := Encode(res)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		c.Root = hex.EncodeToString(root[:])
		c.EncLen = len(enc)
		out = append(out, c)
	}

	doc := vectorFile{
		Spec:  "DETERMINISM_SPEC.md",
		Hash:  "sha3-256 over canonical encoding",
		Codec: "big-endian, rank||dims||F||len||int64 data",
		Gen:   "differential_test.go, randomized, masterSeed=0x5F04B5",
		Cases: out,
	}
	blob, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := "vectors/differential.json"
	if err := os.WriteFile(path, append(blob, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}

	// Report the shape of what was covered, so the suite's breadth is a
	// reported number rather than an adjective.
	var maxN, maxP, minF, maxF, trimmed int
	minF = 255
	for _, c := range out {
		if c.N > maxN {
			maxN = c.N
		}
		if c.P > maxP {
			maxP = c.P
		}
		if int(c.F) < minF {
			minF = int(c.F)
		}
		if int(c.F) > maxF {
			maxF = int(c.F)
		}
		if c.Op == "trimmed" {
			trimmed++
		}
	}
	t.Logf("wrote %s: %d cases", path, len(out))
	t.Logf("  fedavg %d, trimmed %d", len(out)-trimmed, trimmed)
	t.Logf("  N up to %d, P up to %d, F in [%d,%d]", maxN, maxP, minF, maxF)
	t.Logf("verify with: python pyref/check_vectors.py %s", path)
}

// randomCase draws a uniformly valid case. Bounds come from limits.go rather
// than being restated, so a change to the protocol parameters cannot leave the
// generator silently producing invalid or unrepresentative cases.
func randomCase(r *SplitMix64, i int) vectorCase {
	// N: small values dominate, because that is where association order has
	// the fewest terms to disagree about and where a specification bug is
	// least likely to be masked by averaging. A tail reaches larger N.
	n := 1 + int(r.Next()%12)
	if r.Next()%8 == 0 {
		n = 1 + int(r.Next()%200)
	}

	// P: likewise skewed small, with a tail. Kept modest so 2000 cases run in
	// seconds; the paper-scale P=250,000 case remains in the curated set.
	p := 1 + int(r.Next()%64)
	if r.Next()%16 == 0 {
		p = 1 + int(r.Next()%2048)
	}

	// F over the range a deployment might plausibly choose. The quantized
	// magnitude bound is derived from F so |theta_hat| stays inside
	// ParamBound: bound <= 2^(a-1) keeps the generator's values legal with a
	// bit of headroom.
	f := uint8(8 + r.Next()%17) // 8..24
	bound := int64(1) << (ParamBits - 1 - r.Next()%6)

	// Weights: at least 1, and small enough that N*max(w) stays under
	// SumBound even at the largest N drawn above.
	maxW := SumBound / int64(n)
	if maxW > WeightBound-1 {
		maxW = WeightBound - 1
	}
	if maxW < 1 {
		maxW = 1
	}
	weights := make([]int64, n)
	for j := range weights {
		weights[j] = 1 + int64(r.Next()%uint64(maxW))
	}

	c := vectorCase{
		Name:    "diff-" + itoa(i),
		Op:      "fedavg",
		Seed:    r.Next(),
		N:       n,
		P:       p,
		Bound:   bound,
		F:       f,
		Weights: weights,
	}

	// Trimmed mean needs at least one surviving value: n - 2k >= 1.
	if n >= 3 && r.Next()%3 == 0 {
		c.Op = "trimmed"
		c.K = 1 + int(r.Next()%uint64((n-1)/2))
	}

	// Exercise multi-dimensional shapes on a minority of cases: the canonical
	// encoding carries rank and dims, so a rank-1 suite would never test them.
	if p >= 6 && r.Next()%5 == 0 {
		for _, d := range []uint32{2, 3} {
			if uint32(p)%d == 0 {
				c.Dims = []uint32{d, uint32(p) / d}
				break
			}
		}
	}
	return c
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

func parsePositive(s string) (int, error) {
	n := 0
	for _, ch := range s {
		if ch < '0' || ch > '9' {
			return 0, os.ErrInvalid
		}
		n = n*10 + int(ch-'0')
	}
	if n <= 0 {
		return 0, os.ErrInvalid
	}
	return n, nil
}
