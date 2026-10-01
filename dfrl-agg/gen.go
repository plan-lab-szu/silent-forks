package dfrlagg

// Deterministic input generation for golden vectors and for the C1 divergence
// demonstration. It must be reproducible in every language we cross-check
// against, so it uses splitmix64 (integer only, 64-bit wraparound) and maps to
// floats through exactly representable powers of two. No library RNG, no
// float rounding, nothing platform-dependent.

// SplitMix64 is a portable counter-based PRNG.
type SplitMix64 struct{ state uint64 }

func NewSplitMix64(seed uint64) *SplitMix64 { return &SplitMix64{state: seed} }

func (s *SplitMix64) Next() uint64 {
	s.state += 0x9E3779B97F4A7C15
	z := s.state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// GenQuantized fills a tensor of p elements with values uniform in
// (-bound, bound), drawn from seed.
func GenQuantized(seed uint64, p int, bound int64, f uint8) Tensor {
	r := NewSplitMix64(seed)
	data := make([]int64, p)
	span := uint64(2*bound - 1)
	for i := range data {
		data[i] = int64(r.Next()%span) - (bound - 1)
	}
	return Tensor{Dims: []uint32{uint32(p)}, F: f, Data: data}
}

// GenFloat32 draws p float32 values in [-2^-4, 2^-4), a realistic magnitude
// for DQN weights. Every value is exactly representable in float32 and the
// construction uses only exact operations, so an independent implementation
// (e.g. numpy) reproduces the identical bit patterns. That is what makes the
// cross-language divergence comparison meaningful: the inputs are provably the
// same, so any difference in the output is attributable to the summation.
func GenFloat32(seed uint64, p int) []float32 {
	r := NewSplitMix64(seed)
	out := make([]float32, p)
	for i := range out {
		u := r.Next() >> 40 // 24 bits
		out[i] = (float32(u) - 8388608) * 0x1p-27
	}
	return out
}

// PaperWeights are the per-client transition counts used in the submitted
// manuscript's example, kept so that the rewritten pipeline can be compared
// against the original prototype's setting.
var PaperWeights = []int64{1000, 1200, 900, 1100, 1050, 980, 1150, 1020, 970, 1130}

// GenUpdates builds n updates of p parameters each, with deterministic
// addresses and the given weights.
func GenUpdates(seed uint64, n, p int, weights []int64, bound int64, f uint8) []Update {
	ups := make([]Update, n)
	for i := 0; i < n; i++ {
		var a Addr
		a[19] = byte(i + 1)
		a[18] = byte((i + 1) >> 8)
		ups[i] = Update{
			Addr:   a,
			Weight: weights[i%len(weights)],
			Tensor: GenQuantized(seed+uint64(i)*0x1000, p, bound, f),
		}
	}
	return ups
}
