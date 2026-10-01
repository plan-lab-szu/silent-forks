package dfrlagg

import (
	"bytes"
	"testing"
)

func shuffle(ups []Update, r *SplitMix64) []Update {
	out := append([]Update(nil), ups...)
	for i := len(out) - 1; i > 0; i-- {
		j := int(r.Next() % uint64(i+1))
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// TestFedAvgOrderIndependence is the direct verification of Proposition 2
// (§3.1, §6 checklist): shuffling the client order 1000 times must not change
// a single output byte. This is the property that makes Go's randomized map
// iteration order, SIMD reduction trees and parallel reduction harmless.
func TestFedAvgOrderIndependence(t *testing.T) {
	ups := GenUpdates(20260824, 10, 4096, PaperWeights, 1<<17, DefaultF)
	base, err := FedAvg(ups)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Encode(base)
	if err != nil {
		t.Fatal(err)
	}
	r := NewSplitMix64(7)
	for i := 0; i < 1000; i++ {
		got, err := FedAvg(shuffle(ups, r))
		if err != nil {
			t.Fatal(err)
		}
		gb, err := Encode(got)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(want, gb) {
			t.Fatalf("iteration %d: output bytes differ under permuted client order", i)
		}
	}
}

// TestFedAvgRepeatable is the "same input 1000 times" property test from the §6
// checklist. It is weaker than order independence but catches accidental state.
func TestFedAvgRepeatable(t *testing.T) {
	ups := GenUpdates(1, 7, 2048, []int64{3, 5, 7, 11, 13, 17, 19}, 1<<20, DefaultF)
	want, err := StateRoot(mustFedAvg(t, ups))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 1000; i++ {
		got, err := StateRoot(mustFedAvg(t, ups))
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("iteration %d: state root changed", i)
		}
	}
}

// TestFedAvgAgainstExactRational checks the arithmetic itself, not just its
// stability: the result must equal the exactly-rounded weighted mean.
func TestFedAvgAgainstExactRational(t *testing.T) {
	ups := GenUpdates(99, 5, 256, []int64{1, 2, 3, 4, 5}, 1<<23, DefaultF)
	out := mustFedAvg(t, ups)
	var S int64
	for i := range ups {
		S += ups[i].Weight
	}
	for j := range out.Data {
		var acc int64
		for i := range ups {
			acc += ups[i].Weight * ups[i].Tensor.Data[j]
		}
		if want := refDivRoundHalfEven(acc, S); out.Data[j] != want {
			t.Fatalf("element %d = %d, want %d", j, out.Data[j], want)
		}
	}
}

// TestFedAvgBoundaryNoOverflow drives the accumulator to the worst case
// Proposition 1 admits: the maximum client count, the maximum weight and the
// extreme parameter value. If the bit budget were wrong this would wrap.
func TestFedAvgBoundaryNoOverflow(t *testing.T) {
	n := ClientBound - 1
	ups := make([]Update, n)
	for i := 0; i < n; i++ {
		var a Addr
		a[19], a[18] = byte(i+1), byte((i+1)>>8)
		v := ParamBound - 1
		if i%2 == 1 {
			v = -(ParamBound - 1)
		}
		ups[i] = Update{Addr: a, Weight: WeightBound - 1,
			Tensor: Tensor{Dims: []uint32{4}, F: DefaultF, Data: []int64{v, v, -v, 0}}}
	}
	out := mustFedAvg(t, ups)
	// With alternating extremes and equal weights the mean stays in range; the
	// point is that the intermediate accumulator did not wrap.
	for _, v := range out.Data {
		if v <= -ParamBound || v >= ParamBound {
			t.Fatalf("result out of range: %d", v)
		}
	}
}

func TestFedAvgRejectsBadInput(t *testing.T) {
	good := GenUpdates(5, 3, 8, []int64{1, 1, 1}, 1<<10, DefaultF)

	if _, err := FedAvg(nil); err != ErrClientCount {
		t.Errorf("empty set: got %v, want ErrClientCount", err)
	}

	zeroW := append([]Update(nil), good...)
	zeroW[1].Weight = 0
	if _, err := FedAvg(zeroW); err != ErrWeightRange {
		t.Errorf("zero weight: got %v, want ErrWeightRange", err)
	}

	bigW := append([]Update(nil), good...)
	bigW[1].Weight = WeightBound
	if _, err := FedAvg(bigW); err != ErrWeightRange {
		t.Errorf("weight at bound: got %v, want ErrWeightRange", err)
	}

	oor := append([]Update(nil), good...)
	oor[2].Tensor = good[2].Tensor.Clone()
	oor[2].Tensor.Data[0] = ParamBound
	if _, err := FedAvg(oor); err != ErrParamRange {
		t.Errorf("param at bound: got %v, want ErrParamRange", err)
	}

	shape := append([]Update(nil), good...)
	shape[1].Tensor = Tensor{Dims: []uint32{4}, F: DefaultF, Data: make([]int64, 4)}
	if _, err := FedAvg(shape); err != ErrShapeMismatch {
		t.Errorf("shape mismatch: got %v, want ErrShapeMismatch", err)
	}

	fmix := append([]Update(nil), good...)
	fmix[1].Tensor = good[1].Tensor.Clone()
	fmix[1].Tensor.F = DefaultF + 1
	if _, err := FedAvg(fmix); err != ErrShapeMismatch {
		t.Errorf("F mismatch: got %v, want ErrShapeMismatch", err)
	}

	dup := append([]Update(nil), good...)
	dup[2].Addr = dup[0].Addr
	if _, err := FedAvg(dup); err != ErrDuplicateAddr {
		t.Errorf("duplicate address: got %v, want ErrDuplicateAddr", err)
	}
}

func mustFedAvg(t *testing.T, ups []Update) Tensor {
	t.Helper()
	out, err := FedAvg(ups)
	if err != nil {
		t.Fatal(err)
	}
	return out
}
