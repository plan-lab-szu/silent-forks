package dfrlagg

import (
	"math"
	"testing"
)

func TestQuantizeRoundTripError(t *testing.T) {
	// Quantization error must stay within 2^-(F+1), the bound quoted in §2.1
	// and used to argue that quantization noise is far below DQN training noise.
	r := NewSplitMix64(4242)
	const p = 10000
	theta := make([]float32, p)
	for i := range theta {
		theta[i] = (float32(r.Next()>>40) - 8388608) * 0x1p-24 // in [-0.5, 0.5)
	}
	tq, err := Quantize(theta, []uint32{p}, DefaultF)
	if err != nil {
		t.Fatal(err)
	}
	back := Dequantize(tq)
	bound := QuantErrorBound(DefaultF)
	worst := 0.0
	for i := range theta {
		d := math.Abs(float64(theta[i]) - float64(back[i]))
		if d > worst {
			worst = d
		}
	}
	if worst > bound {
		t.Fatalf("worst quantization error %g exceeds bound %g", worst, bound)
	}
	t.Logf("F=%d worst error %.3e (bound %.3e)", DefaultF, worst, bound)
}

func TestQuantizeRejectsOutOfRange(t *testing.T) {
	// MaxAbsFloat(20) = 16: a parameter at or beyond it must be rejected, not
	// clamped. Clamping would silently aggregate a model the client did not send.
	for _, v := range []float32{16, -16, 1e9, float32(math.Inf(1)), float32(math.NaN())} {
		if _, err := Quantize([]float32{v}, []uint32{1}, DefaultF); err != ErrParamRange {
			t.Errorf("Quantize(%v) error = %v, want ErrParamRange", v, err)
		}
	}
	if _, err := Quantize([]float32{15.9}, []uint32{1}, DefaultF); err != nil {
		t.Errorf("Quantize(15.9) unexpectedly rejected: %v", err)
	}
}

func TestQuantizeShapeMismatch(t *testing.T) {
	if _, err := Quantize([]float32{1, 2, 3}, []uint32{2, 2}, DefaultF); err != ErrEncoding {
		t.Errorf("got %v, want ErrEncoding", err)
	}
}
