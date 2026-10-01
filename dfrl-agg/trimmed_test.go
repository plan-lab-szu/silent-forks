package dfrlagg

import (
	"bytes"
	"sort"
	"testing"
)

func TestTrimmedMeanAgainstReference(t *testing.T) {
	ups := GenUpdates(31337, 9, 512, []int64{1, 1, 1, 1, 1, 1, 1, 1, 1}, 1<<18, DefaultF)
	const k = 2
	out, err := TrimmedMean(ups, k)
	if err != nil {
		t.Fatal(err)
	}
	for j := range out.Data {
		col := make([]int64, len(ups))
		for i := range ups {
			col[i] = ups[i].Tensor.Data[j]
		}
		sort.Slice(col, func(a, b int) bool { return col[a] < col[b] })
		var acc int64
		for i := k; i < len(col)-k; i++ {
			acc += col[i]
		}
		if want := refDivRoundHalfEven(acc, int64(len(col)-2*k)); out.Data[j] != want {
			t.Fatalf("element %d = %d, want %d", j, out.Data[j], want)
		}
	}
}

// TestTrimmedMeanOrderIndependence checks that permuting submissions cannot
// change the retained numerical multiset. Values come from a small range so
// ties are frequent; equal values remain interchangeable at trim boundaries.
func TestTrimmedMeanOrderIndependence(t *testing.T) {
	ups := GenUpdates(5, 11, 2048, []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 4, DefaultF)
	base, err := TrimmedMean(ups, 3)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Encode(base)
	if err != nil {
		t.Fatal(err)
	}
	r := NewSplitMix64(11)
	for i := 0; i < 1000; i++ {
		got, err := TrimmedMean(shuffle(ups, r), 3)
		if err != nil {
			t.Fatal(err)
		}
		gb, err := Encode(got)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(want, gb) {
			t.Fatalf("iteration %d: trimmed mean changed under permuted order", i)
		}
	}
}

// TestTrimmedMeanBoundsOutlierInfluence is the mechanism behind U3: a Byzantine
// client submitting an extreme update must be trimmed away, whereas FedAvg
// aggregates it faithfully. This is a unit-level illustration; the throughput
// and fairness degradation curves are X5, in Phase 4.
func TestTrimmedMeanBoundsOutlierInfluence(t *testing.T) {
	const n, p = 7, 64
	ups := GenUpdates(77, n, p, []int64{1, 1, 1, 1, 1, 1, 1}, 1<<10, DefaultF)
	// Two clients go maximally adversarial in opposite directions.
	ups[1].Tensor = ups[1].Tensor.Clone()
	ups[5].Tensor = ups[5].Tensor.Clone()
	for j := 0; j < p; j++ {
		ups[1].Tensor.Data[j] = ParamBound - 1
		ups[5].Tensor.Data[j] = -(ParamBound - 1)
	}
	avg, err := FedAvg(ups)
	if err != nil {
		t.Fatal(err)
	}
	trim, err := TrimmedMean(ups, 1)
	if err != nil {
		t.Fatal(err)
	}
	honest := make([]Update, 0, n-2)
	for i := range ups {
		if i != 1 && i != 5 {
			honest = append(honest, ups[i])
		}
	}
	clean, err := TrimmedMean(honest, 0)
	if err != nil {
		t.Fatal(err)
	}
	var devAvg, devTrim int64
	for j := 0; j < p; j++ {
		devAvg += abs64(avg.Data[j] - clean.Data[j])
		devTrim += abs64(trim.Data[j] - clean.Data[j])
	}
	if devTrim >= devAvg {
		t.Fatalf("trimming did not reduce deviation: trimmed=%d fedavg=%d", devTrim, devAvg)
	}
	t.Logf("deviation from honest-only mean: fedavg=%d trimmed=%d", devAvg, devTrim)
}

func TestTrimmedMeanRejectsBadK(t *testing.T) {
	ups := GenUpdates(1, 5, 8, []int64{1, 1, 1, 1, 1}, 1<<10, DefaultF)
	for _, k := range []int{-1, 3, 5} {
		if _, err := TrimmedMean(ups, k); err != ErrTrimParam {
			t.Errorf("k=%d: got %v, want ErrTrimParam", k, err)
		}
	}
	if _, err := TrimmedMean(ups, 2); err != nil {
		t.Errorf("k=2 on n=5 rejected: %v", err)
	}
}

// TestTrimmedMeanValidatesIgnoredWeight pins the first half of the weight
// field's contract under OP_ROBUSTAGG: the operator ignores the value, but it
// must still validate it. An implementation that skipped the check because
// "weights are ignored anyway" would accept a submission set that a
// conforming implementation rejects, which is a consensus split.
func TestTrimmedMeanValidatesIgnoredWeight(t *testing.T) {
	good := GenUpdates(7, 9, 64, []int64{1, 1, 1, 1, 1, 1, 1, 1, 1}, 1<<12, DefaultF)
	const k = 2

	zeroW := append([]Update(nil), good...)
	zeroW[3].Weight = 0
	if _, err := TrimmedMean(zeroW, k); err != ErrWeightRange {
		t.Errorf("zero weight: got %v, want ErrWeightRange", err)
	}

	bigW := append([]Update(nil), good...)
	bigW[3].Weight = WeightBound
	if _, err := TrimmedMean(bigW, k); err != ErrWeightRange {
		t.Errorf("weight at bound: got %v, want ErrWeightRange", err)
	}
}

// TestTrimmedMeanIgnoresInRangeWeight pins the second half: any two in-range
// weight assignments over the same tensors give the same aggregate. Two
// calldata byte strings differing only in this field therefore denote one
// output, so the specification's one-to-one requirement is a property of the
// tensor encoding and not of operator calldata.
func TestTrimmedMeanIgnoresInRangeWeight(t *testing.T) {
	const k = 2
	base := GenUpdates(11, 9, 256, []int64{1, 1, 1, 1, 1, 1, 1, 1, 1}, 1<<12, DefaultF)
	want, err := TrimmedMean(base, k)
	if err != nil {
		t.Fatal(err)
	}

	alt := append([]Update(nil), base...)
	for i := range alt {
		alt[i].Weight = int64(1 + i*7919) // distinct, all inside [1, WeightBound)
	}
	got, err := TrimmedMean(alt, k)
	if err != nil {
		t.Fatal(err)
	}
	if got.F != want.F || len(got.Data) != len(want.Data) {
		t.Fatalf("shape/F changed: F %d vs %d, len %d vs %d",
			got.F, want.F, len(got.Data), len(want.Data))
	}
	for j := range want.Data {
		if got.Data[j] != want.Data[j] {
			t.Fatalf("element %d changed with weights: %d vs %d",
				j, got.Data[j], want.Data[j])
		}
	}
}
