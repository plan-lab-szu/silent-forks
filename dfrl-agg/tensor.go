package dfrlagg

// Tensor is a quantized model tensor: fixed-point int64 values with a
// protocol-wide fractional bit count F. Shape is explicit so that the
// canonical encoding is unambiguous (§4.2).
type Tensor struct {
	Dims []uint32
	F    uint8
	Data []int64
}

// Addr identifies a client submission for membership and duplicate checks.
type Addr [20]byte

// Update is one client submission: an integer weight recorded on chain (§4.3)
// and a quantized tensor.
type Update struct {
	Addr   Addr
	Weight int64
	Tensor Tensor
}

// numel returns the element count implied by Dims, and whether it fits the
// bounds we accept. Rank 0 and zero-length dimensions are rejected: a tensor
// with no elements has no canonical meaning here.
func numel(dims []uint32) (uint64, bool) {
	if len(dims) == 0 || len(dims) > MaxRank {
		return 0, false
	}
	headerBytes := uint64(headerFixed + 4*len(dims))
	maxElements := (maxCanonicalTensorBytes - headerBytes) / 8
	n := uint64(1)
	for _, d := range dims {
		if d == 0 {
			return 0, false
		}
		// Check before multiplying so an oversized shape cannot wrap and so
		// every accepted tensor fits the uint32 length prefix of the EVM ABI.
		if n > maxElements/uint64(d) {
			return 0, false
		}
		n *= uint64(d)
	}
	return n, true
}

// Validate enforces §2.2 on a tensor: self-consistent shape and every element
// within the representable range.
func (t Tensor) Validate() error {
	n, ok := numel(t.Dims)
	if !ok || uint64(len(t.Data)) != n {
		return ErrEncoding
	}
	for _, v := range t.Data {
		if v <= -ParamBound || v >= ParamBound {
			return ErrParamRange
		}
	}
	return nil
}

// sameShape reports whether b has the identical shape and F as t. Aggregation
// across mismatched shapes is rejected rather than broadcast.
func (t Tensor) sameShape(b Tensor) bool {
	if t.F != b.F || len(t.Dims) != len(b.Dims) || len(t.Data) != len(b.Data) {
		return false
	}
	for i := range t.Dims {
		if t.Dims[i] != b.Dims[i] {
			return false
		}
	}
	return true
}

// Clone returns a deep copy. Used by tests that shuffle inputs.
func (t Tensor) Clone() Tensor {
	d := make([]uint32, len(t.Dims))
	copy(d, t.Dims)
	v := make([]int64, len(t.Data))
	copy(v, t.Data)
	return Tensor{Dims: d, F: t.F, Data: v}
}
