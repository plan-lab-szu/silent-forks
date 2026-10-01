package dfrlagg

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Scope re-verification for the honest-agreement claim (prerequisite work
// agreed 2026-09-17). TestSurveyHonestAgreementRegimes established the claim
// on round 0 only, over DISJOINT windows, and only across the first 4,000 of
// the 7,850 coordinates. Three limits followed from that, and the manuscript
// had to be scoped to "this round and model" and "the windows we scan".
//
// This file removes two of the three limits without needing new training data:
//
//	1. every exported round, not round 0 alone;
//	2. the full coordinate range, not the first 4,000;
//	3. SLIDING windows at stride 1 in addition to disjoint ones, so a claim of
//	   the form "no P'-coordinate window agrees" covers every contiguous window
//	   position rather than a sample of them.
//
// The third limit -- a second model and several training seeds -- still needs a
// new export; export_x4_rounds.py hardcodes clients/rounds/alpha/seed and one
// model_kind, and models.build already supports "mlp".
//
// Exploratory: prints a survey, asserts only that agreement counts never exceed
// the number of windows examined.

// agreementCount reports how many windows of width w, taken at the given
// stride over [0, p), leave all six conforming strategies on one commitment.
func agreementCount(theta [][]float32, weights []float32, w, stride int) (agree, total int, firstAgree int) {
	p := len(theta[0])
	firstAgree = -1
	for off := 0; off+w <= p; off += stride {
		total++
		if n, _ := distinctRoots(sliceTheta(theta, off, w), weights); n == 1 {
			agree++
			if firstAgree < 0 {
				firstAgree = off
			}
		}
	}
	return agree, total, firstAgree
}

func exportedRoundCount(t *testing.T) int {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(x4ExportDir, "manifest.json"))
	if err != nil {
		t.Skipf("no X4 export at %s (run python fl-workload/export_x4_rounds.py): %v", x4ExportDir, err)
	}
	var m x4Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	return len(m.RoundMeta)
}

// TestScopeSurveyEveryRoundFullRange is the disjoint-window survey, rerun over
// every exported round and the whole coordinate range.
func TestScopeSurveyEveryRoundFullRange(t *testing.T) {
	rounds := exportedRoundCount(t)
	widths := []int{1, 2, 4, 8, 16, 32, 64, 256}

	t.Logf("exported rounds: %d; strategies: %d", rounds, len(AllFloatStrategies))
	for r := 0; r < rounds; r++ {
		theta, weights := loadRealRound(t, r)
		p := len(theta[0])
		full, byRoot := distinctRoots(theta, weights)
		t.Logf("round %d: N=%d P=%d -- full tensor gives %d distinct commitments among %d strategies",
			r, len(theta), p, full, len(AllFloatStrategies))
		for root, sts := range byRoot {
			t.Logf("    commitment %x.. shared by %d strategies: %v", root[:4], len(sts), sts)
		}
		for _, w := range widths {
			agree, total, first := agreementCount(theta, weights, w, w)
			if agree > total {
				t.Fatalf("round %d width %d: agree %d > total %d", r, w, agree, total)
			}
			t.Logf("    disjoint P'=%-4d %5d/%-5d agree (first agreeing offset %d)",
				w, agree, total, first)
		}
	}
}

// TestScopeSurveySlidingWindows strengthens the "no agreeing window" half of
// the claim: at stride 1 every contiguous window position is examined, so a
// zero count rules out agreement anywhere in the model at that width rather
// than in a sampled subset.
func TestScopeSurveySlidingWindows(t *testing.T) {
	rounds := exportedRoundCount(t)
	widths := []int{1, 2, 4, 8, 16}

	for r := 0; r < rounds; r++ {
		theta, weights := loadRealRound(t, r)
		t.Logf("round %d: sliding windows at stride 1 over all %d coordinates", r, len(theta[0]))
		for _, w := range widths {
			agree, total, first := agreementCount(theta, weights, w, 1)
			if agree > total {
				t.Fatalf("round %d width %d: agree %d > total %d", r, w, agree, total)
			}
			pct := 100 * float64(agree) / float64(total)
			t.Logf("    sliding  P'=%-3d %5d/%-5d agree (%.2f%%, first agreeing offset %d)",
				w, agree, total, pct, first)
		}
	}
}
