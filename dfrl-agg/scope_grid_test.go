package dfrlagg

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// Second half of the scope re-verification agreed 2026-09-17: does the
// honest-agreement picture survive a change of model and a change of training
// seed? TestScopeSurvey* covered every round and the full coordinate range but
// only the one exported run (logreg, seed 1).
//
// fl-workload/export_x4_rounds.py now takes --model/--seed/--out, and the grid
// below was exported with it:
//
//	python export_x4_rounds.py --model {logreg,mlp} --seed {1,2,3} \
//	    --backend python --out runs/scope_export/<model>_seed<seed>
//
// Two notes on that export. It uses the Python reference aggregator because an
// OS application-control policy on this machine refuses to execute the freshly
// built Go binary; re-exporting the original X4 configuration this way
// reproduced all 60 files byte-identically, aggregate and state root included,
// so the two backends are interchangeable here. And logreg/seed1 duplicates the
// original export on purpose, as a control that the grid loader sees the same
// data the paper's tests do.
//
// Exploratory: prints a survey, asserts only internal consistency.

const scopeGridDir = "../fl-workload/runs/scope_export"

type scopeManifest struct {
	Clients   int    `json:"clients"`
	Rounds    int    `json:"rounds"`
	Dims      []int  `json:"dims"`
	F         int    `json:"f"`
	Model     string `json:"model"`
	Seed      int    `json:"seed"`
	Backend   string `json:"backend"`
	RoundMeta []struct {
		Round   int   `json:"round"`
		Weights []int `json:"weights"`
	} `json:"round_meta"`
}

func readScopeManifest(t *testing.T, dir string) scopeManifest {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest in %s: %v", dir, err)
	}
	var m scopeManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// loadRoundFrom is loadRealRound with the export directory supplied, so the
// paper's own loader stays untouched.
func loadRoundFrom(t *testing.T, dir string, m scopeManifest, round int) (theta [][]float32, weights []float32) {
	t.Helper()
	roundDir := filepath.Join(dir, "round"+pad4(round))
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
		t.Fatalf("%s round %d: %d tensors but %d weights", dir, round, len(theta), len(weights))
	}
	return theta, weights
}

func scopeGridRuns(t *testing.T) []string {
	t.Helper()
	entries, err := os.ReadDir(scopeGridDir)
	if err != nil {
		t.Skipf("no scope export at %s (see the command in this file's header): %v",
			scopeGridDir, err)
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			out = append(out, filepath.Join(scopeGridDir, e.Name()))
		}
	}
	sort.Strings(out)
	if len(out) == 0 {
		t.Skipf("no run directories under %s", scopeGridDir)
	}
	return out
}

// TestScopeGridDisjointWindows repeats the disjoint-window survey for every
// (model, seed) run over its whole coordinate range. With 195 (model, seed,
// round) cells the per-round lines are summarised per run: what matters is
// whether any cell departs from the pattern, not each cell's exact count.
func TestScopeGridDisjointWindows(t *testing.T) {
	widths := []int{1, 2, 4, 8, 16, 32, 64, 256}
	cells := 0
	for _, dir := range scopeGridRuns(t) {
		m := readScopeManifest(t, dir)
		distinctSeen := map[int]int{}
		lo := make(map[int]int, len(widths))
		hi := make(map[int]int, len(widths))
		tot := make(map[int]int, len(widths))
		for i, w := range widths {
			_ = i
			lo[w], hi[w] = -1, -1
		}
		for r := 0; r < m.Rounds; r++ {
			theta, weights := loadRoundFrom(t, dir, m, r)
			full, _ := distinctRoots(theta, weights)
			distinctSeen[full]++
			for _, w := range widths {
				agree, total, _ := agreementCount(theta, weights, w, w)
				if agree > total {
					t.Fatalf("%s r%d w%d: agree %d > total %d", dir, r, w, agree, total)
				}
				tot[w] = total
				if lo[w] < 0 || agree < lo[w] {
					lo[w] = agree
				}
				if agree > hi[w] {
					hi[w] = agree
				}
			}
			cells++
		}
		t.Logf("=== %-14s model=%-6s seed=%d P=%-7d rounds=%d backend=%s",
			filepath.Base(dir), m.Model, m.Seed, m.Dims[0], m.Rounds, m.Backend)
		t.Logf("      full-tensor distinct commitments over %d rounds: %s",
			m.Rounds, histogram(distinctSeen))
		for _, w := range widths {
			if hi[w] == 0 {
				t.Logf("      disjoint P'=%-4d 0 of %d in every round", w, tot[w])
				continue
			}
			t.Logf("      disjoint P'=%-4d %d..%d of %d (%.2f%%..%.2f%%)",
				w, lo[w], hi[w], tot[w],
				100*float64(lo[w])/float64(tot[w]), 100*float64(hi[w])/float64(tot[w]))
		}
	}
	t.Logf("cells surveyed: %d", cells)
}

// TestScopeGridSixteenCoordinateWindows is the load-bearing half: the abstract
// says no 16-coordinate window agrees, so every window position is checked at
// every (model, seed, round) rather than a sample.
func TestScopeGridSixteenCoordinateWindows(t *testing.T) {
	worst, cells, windows, agreements := 0, 0, 0, 0
	for _, dir := range scopeGridRuns(t) {
		m := readScopeManifest(t, dir)
		runAgree, runWindows := 0, 0
		for r := 0; r < m.Rounds; r++ {
			theta, weights := loadRoundFrom(t, dir, m, r)
			agree, total, _ := agreementCount(theta, weights, 16, 1)
			runAgree += agree
			runWindows += total
			if agree > worst {
				worst = agree
			}
			cells++
		}
		windows += runWindows
		agreements += runAgree
		t.Logf("%-14s model=%-6s seed=%d rounds=%-3d sliding P'=16: %d agreeing of %d window positions",
			filepath.Base(dir), m.Model, m.Seed, m.Rounds, runAgree, runWindows)
	}
	t.Logf("grid total: %d cells, %d window positions examined, %d agreements "+
		"(worst single round: %d)", cells, windows, agreements, worst)
}

func histogram(m map[int]int) string {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	out := ""
	for _, k := range keys {
		if out != "" {
			out += ", "
		}
		out += fmt.Sprintf("%d in %d round(s)", k, m[k])
	}
	return out
}
