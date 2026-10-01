package dfrlagg

import (
	"encoding/hex"
	"encoding/json"
	"flag"
	"os"
	"testing"
)

// Golden vectors pin the exact output bytes of the aggregation primitives to a
// committed file. Two independent uses:
//
//	1. Cross-architecture: CI runs this same test on x86-64 and arm64. Byte
//	   equality there is the core positive evidence for X6.
//	2. Cross-language: pyref/ regenerates the identical inputs from the same
//	   seeds and checks the same roots, which is the "validators implement the
//	   spec independently" story rather than "one codebase agrees with itself".
//
// Regenerate with: go test -run TestGoldenVectors -update
// Any change to the file is a protocol change and must be reviewed as one.

var updateVectors = flag.Bool("update", false, "rewrite vectors/vectors.json from the current implementation")

const vectorsPath = "vectors/vectors.json"

type vectorCase struct {
	Name    string   `json:"name"`
	Op      string   `json:"op"`
	Seed    uint64   `json:"seed"`
	N       int      `json:"n"`
	P       int      `json:"p"`
	Dims    []uint32 `json:"dims,omitempty"`
	Bound   int64    `json:"bound"`
	F       uint8    `json:"f"`
	Weights []int64  `json:"weights"`
	K       int      `json:"k,omitempty"`
	Root    string   `json:"root"`
	EncLen  int      `json:"encoded_len"`
}

type vectorFile struct {
	Spec  string       `json:"spec"`
	Hash  string       `json:"hash"`
	Codec string       `json:"codec"`
	Gen   string       `json:"generator"`
	Cases []vectorCase `json:"cases"`
}

func repeatWeights(v int64, n int) []int64 {
	w := make([]int64, n)
	for i := range w {
		w[i] = v
	}
	return w
}

func ascendingWeights(n int) []int64 {
	w := make([]int64, n)
	for i := range w {
		w[i] = int64(i + 1)
	}
	return w
}

// vectorSpecs is the authoritative list of cases. Roots live in the JSON file.
func vectorSpecs() []vectorCase {
	return []vectorCase{
		{Name: "fedavg-paper-scale", Op: "fedavg", Seed: 20260823, N: 10, P: 250000,
			Bound: 1 << 17, F: DefaultF, Weights: PaperWeights},
		{Name: "fedavg-n3-small", Op: "fedavg", Seed: 3, N: 3, P: 1024,
			Bound: 1 << 17, F: DefaultF, Weights: []int64{1000, 1200, 900}},
		{Name: "fedavg-rank3-shape", Op: "fedavg", Seed: 17, N: 4, P: 512,
			Dims: []uint32{4, 8, 16}, Bound: 1 << 20, F: DefaultF,
			Weights: []int64{7, 11, 13, 17}},
		{Name: "fedavg-max-clients", Op: "fedavg", Seed: 1023, N: ClientBound - 1, P: 16,
			Bound: 1 << 12, F: DefaultF, Weights: ascendingWeights(ClientBound - 1)},
		{Name: "fedavg-bit-budget-extreme", Op: "fedavg", Seed: 63, N: 512, P: 32,
			Bound: ParamBound, F: DefaultF, Weights: repeatWeights(WeightBound-1, 512)},
		{Name: "fedavg-f12", Op: "fedavg", Seed: 12, N: 8, P: 256,
			Bound: 1 << 11, F: 12, Weights: ascendingWeights(8)},
		{Name: "fedavg-f24", Op: "fedavg", Seed: 24, N: 8, P: 256,
			Bound: 1 << 23, F: 24, Weights: ascendingWeights(8)},
		{Name: "trimmed-k3-tie-heavy", Op: "trimmed", Seed: 5, N: 11, P: 2048,
			Bound: 4, F: DefaultF, Weights: ascendingWeights(11), K: 3},
		{Name: "trimmed-k1-outliers", Op: "trimmed", Seed: 77, N: 7, P: 512,
			Bound: 1 << 20, F: DefaultF, Weights: repeatWeights(1, 7), K: 1},
	}
}

func (c vectorCase) run() (Tensor, error) {
	ups := GenUpdates(c.Seed, c.N, c.P, c.Weights, c.Bound, c.F)
	if len(c.Dims) > 0 {
		for i := range ups {
			ups[i].Tensor.Dims = append([]uint32(nil), c.Dims...)
		}
	}
	if c.Op == "trimmed" {
		return TrimmedMean(ups, c.K)
	}
	return FedAvg(ups)
}

func TestGoldenVectors(t *testing.T) {
	specs := vectorSpecs()
	computed := make([]vectorCase, 0, len(specs))
	for _, c := range specs {
		out, err := c.run()
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		root, err := StateRoot(out)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		enc, err := Encode(out)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		c.Root = hex.EncodeToString(root[:])
		c.EncLen = len(enc)
		computed = append(computed, c)
	}

	if *updateVectors {
		f := vectorFile{
			Spec:  "DETERMINISM_SPEC.md v0.2 (2026-08-24)",
			Hash:  "SHA3-256 (FIPS-202) over the canonical encoding of the output tensor",
			Codec: "rank:u8 || dims:u32[rank] || F:u8 || len:u64 || data:i64[len], all big-endian",
			Gen:   "inputs from splitmix64(seed + 0x1000*client_index), value = next() % (2*bound-1) - (bound-1)",
			Cases: computed,
		}
		b, err := json.MarshalIndent(f, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(vectorsPath, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d cases to %s", len(computed), vectorsPath)
		return
	}

	raw, err := os.ReadFile(vectorsPath)
	if err != nil {
		t.Fatalf("%v (run: go test -run TestGoldenVectors -update)", err)
	}
	var want vectorFile
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	byName := make(map[string]vectorCase, len(want.Cases))
	for _, c := range want.Cases {
		byName[c.Name] = c
	}
	if len(want.Cases) != len(computed) {
		t.Errorf("vector count: file has %d, implementation defines %d", len(want.Cases), len(computed))
	}
	for _, got := range computed {
		w, ok := byName[got.Name]
		if !ok {
			t.Errorf("%s: missing from %s", got.Name, vectorsPath)
			continue
		}
		if got.Root != w.Root {
			t.Errorf("%s: state root %s, golden %s", got.Name, got.Root, w.Root)
		}
		if got.EncLen != w.EncLen {
			t.Errorf("%s: encoded length %d, golden %d", got.Name, got.EncLen, w.EncLen)
		}
	}
}
