package dfrlagg

import "math"

// Quantization lives on the CLIENT side, outside the consensus boundary (§1).
// It is the only place in this package (besides floatref.go) where floating
// point appears, and that is deliberate: DQN training cannot avoid float, so
// the boundary is drawn immediately after quantization. Whatever the client
// produces IS the definition of the input, so a client-side rounding
// difference is not a consensus fault — it merely yields a different, equally
// legal submission.

// Quantize maps float32 parameters to fixed point with fractional bit count f:
//
//	theta_hat = round_half_even(theta * 2^f)
//
// Values that fall outside the representable range are REJECTED, not clamped:
// silently clamping would let a client's out-of-range model be aggregated as
// something it is not.
func Quantize(theta []float32, dims []uint32, f uint8) (Tensor, error) {
	n, ok := numel(dims)
	if !ok || uint64(len(theta)) != n {
		return Tensor{}, ErrEncoding
	}
	scale := math.Ldexp(1, int(f))
	data := make([]int64, len(theta))
	for i, v := range theta {
		x := float64(v) * scale
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return Tensor{}, ErrParamRange
		}
		q := int64(math.RoundToEven(x))
		if q <= -ParamBound || q >= ParamBound {
			return Tensor{}, ErrParamRange
		}
		data[i] = q
	}
	t := Tensor{Dims: append([]uint32(nil), dims...), F: f, Data: data}
	if err := t.Validate(); err != nil {
		return Tensor{}, err
	}
	return t, nil
}

// Dequantize maps fixed point back to float32 for the client to resume
// training from. Also off the consensus path.
func Dequantize(t Tensor) []float32 {
	scale := math.Ldexp(1, -int(t.F))
	out := make([]float32, len(t.Data))
	for i, v := range t.Data {
		out[i] = float32(float64(v) * scale)
	}
	return out
}

// MaxAbsFloat is the exclusive bound on |theta| representable at f fractional
// bits: 2^(ParamBits-f). At f=20 this is 16.
func MaxAbsFloat(f uint8) float64 {
	return math.Ldexp(1, ParamBits-int(f))
}

// QuantErrorBound is the worst-case absolute quantization error, 2^-(f+1).
func QuantErrorBound(f uint8) float64 {
	return math.Ldexp(1, -(int(f) + 1))
}
