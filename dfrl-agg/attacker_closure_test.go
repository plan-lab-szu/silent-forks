package dfrlagg

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Real federated-MNIST round data exported by fl-workload/export_x4_rounds.py.
// Honest client updates come from actual training, not from a generator, so the
// attacker in these tests is choosing its own update against a fixed, realistic
// honest population rather than against synthetic noise.
const x4ExportDir = "../fl-workload/runs/x4_export"

type x4Manifest struct {
	Clients   int   `json:"clients"`
	Rounds    int   `json:"rounds"`
	Dims      []int `json:"dims"`
	F         int   `json:"f"`
	RoundMeta []struct {
		Round   int   `json:"round"`
		Weights []int `json:"weights"`
	} `json:"round_meta"`
}

// loadRealRound returns the dequantized float32 updates and weights for one
// exported round. Dequantization is exact: a quantized value below 2^24 divided
// by 2^F is representable in float32 without rounding, so the inputs carry no
// rounding of their own and any divergence is attributable to the aggregation.
func loadRealRound(t *testing.T, round int) (theta [][]float32, weights []float32) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(x4ExportDir, "manifest.json"))
	if err != nil {
		t.Skipf("no X4 export at %s (run python fl-workload/export_x4_rounds.py): %v", x4ExportDir, err)
	}
	var m x4Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	if round >= len(m.RoundMeta) {
		t.Fatalf("round %d not in export (%d rounds)", round, len(m.RoundMeta))
	}
	roundDir := filepath.Join(x4ExportDir, "round"+pad4(round))
	entries, err := os.ReadDir(roundDir)
	if err != nil {
		t.Fatal(err)
	}
	scale := float32(1) / float32(int64(1)<<m.F)
	for _, e := range entries {
		name := e.Name()
		if len(name) < 6 || name[:6] != "client" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(roundDir, name))
		if err != nil {
			t.Fatal(err)
		}
		tq, err := Decode(b)
		if err != nil {
			t.Fatalf("decode %s: %v", name, err)
		}
		row := make([]float32, len(tq.Data))
		for i, v := range tq.Data {
			row[i] = float32(v) * scale
		}
		theta = append(theta, row)
	}
	for _, w := range m.RoundMeta[round].Weights {
		weights = append(weights, float32(w))
	}
	if len(theta) != len(weights) {
		t.Fatalf("got %d client tensors but %d weights", len(theta), len(weights))
	}
	return theta, weights
}

func pad4(n int) string {
	s := []byte("0000")
	for i := 3; i >= 0 && n > 0; i-- {
		s[i] = byte('0' + n%10)
		n /= 10
	}
	return string(s)
}

// distinctRoots reports how many distinct float32 aggregates the conforming
// strategies produce on one input, i.e. how many states honest validators
// running different conforming implementations would commit.
func distinctRoots(theta [][]float32, weights []float32) (int, map[[32]byte][]FloatStrategy) {
	byRoot := map[[32]byte][]FloatStrategy{}
	for _, st := range AllFloatStrategies {
		r := FloatRoot(FloatAvg(theta, weights, st))
		byRoot[r] = append(byRoot[r], st)
	}
	return len(byRoot), byRoot
}

// slice returns a copy of theta restricted to coordinates [off, off+p).
func sliceTheta(theta [][]float32, off, p int) [][]float32 {
	out := make([][]float32, len(theta))
	for i, row := range theta {
		out[i] = append([]float32(nil), row[off:off+p]...)
	}
	return out
}

// TestSurveyHonestAgreementRegimes locates configurations in which the honest
// population alone does NOT diverge -- every conforming strategy agrees, so the
// deployment looks and is healthy. Those are the only configurations in which
// an attacker's own choice of update can be said to cause anything: where
// honest inputs already diverge, there is nothing for an attacker to do.
//
// Exploratory: prints a survey, asserts nothing.
func TestSurveyHonestAgreementRegimes(t *testing.T) {
	theta, weights := loadRealRound(t, 0)
	p := len(theta[0])
	t.Logf("real round 0: N=%d clients, P=%d params, weights[0..3]=%v", len(theta), p, weights[:4])

	full, _ := distinctRoots(theta, weights)
	t.Logf("full tensor (P=%d): %d distinct roots among %d strategies", p, full, len(AllFloatStrategies))

	for _, pp := range []int{1, 2, 3, 4, 8, 16, 64, 256} {
		agree, total := 0, 0
		firstAgreeOff := -1
		for off := 0; off+pp <= p && off < 4000; off += pp {
			total++
			n, _ := distinctRoots(sliceTheta(theta, off, pp), weights)
			if n == 1 {
				agree++
				if firstAgreeOff < 0 {
					firstAgreeOff = off
				}
			}
		}
		t.Logf("  P'=%-4d windows: %d/%d agree on all 6 strategies (first agreeing offset %d)",
			pp, agree, total, firstAgreeOff)
	}
}
