package dfrlagg

// FLOATING-POINT REFERENCE IMPLEMENTATION — NEGATIVE CONTROL ONLY.
//
// Everything in this file exists to DEMONSTRATE the failure mode that the rest
// of the package removes (contribution C1). It must never be reachable from
// FedAvg, TrimmedMean, or any consensus path. Nothing else in this package
// calls into it; the only callers are tests and cmd/dfrl-agg's diverge
// subcommand.
//
// Each strategy below is a defensible way to implement a weighted average.
// They are not bugs. That is the entire point: the consensus protocol says
// "compute the weighted average" and says nothing about association order,
// library version, or build configuration, so all of these are conforming
// implementations that produce different bytes.

import "crypto/sha3"

// FloatStrategy names one summation strategy.
type FloatStrategy string

const (
	// SeqFwd accumulates clients in index order — the naive loop.
	SeqFwd FloatStrategy = "seq-fwd"
	// SeqRev accumulates in reverse index order. Reachable in practice: Go
	// randomizes map iteration order, so an implementation that iterates a
	// map[Addr]Tensor visits clients in an order that varies per process.
	SeqRev FloatStrategy = "seq-rev"
	// Pairwise uses recursive pairwise summation, as numpy's np.sum does.
	Pairwise FloatStrategy = "pairwise"
	// Blocked accumulates in blocks of four with separate accumulators, the
	// shape an unrolled or vectorized dot product takes.
	Blocked FloatStrategy = "blocked-4"
	// Widened accumulates in float64 and narrows at the end — what an
	// implementation does when it tries to be MORE accurate.
	Widened FloatStrategy = "float64-acc"
	// FusedMul leaves the multiply-add unfused-or-fused at the compiler's
	// discretion. Go's specification permits an implementation to fuse x*y+z
	// into a single FMA when the intermediate result is not explicitly
	// rounded, and some backends (notably arm64) do. Whether this strategy
	// differs from SeqFwd is therefore ARCHITECTURE-DEPENDENT and must be
	// established empirically on each target, never asserted from the spec
	// text alone.
	FusedMul FloatStrategy = "fused-mul"
)

// AllFloatStrategies is the set exercised by the divergence demonstration.
var AllFloatStrategies = []FloatStrategy{SeqFwd, SeqRev, Pairwise, Blocked, Widened, FusedMul}

// FloatAvg computes sum(s_n * theta_n) / S in float32 using the named strategy.
// theta is indexed [client][param]; weights must have the same length.
func FloatAvg(theta [][]float32, weights []float32, st FloatStrategy) []float32 {
	n := len(theta)
	p := len(theta[0])
	var S float32
	for _, w := range weights {
		S += w
	}
	acc := make([]float32, p)

	switch st {
	case SeqFwd:
		for i := 0; i < n; i++ {
			for j := 0; j < p; j++ {
				acc[j] = acc[j] + float32(weights[i]*theta[i][j])
			}
		}
	case SeqRev:
		for i := n - 1; i >= 0; i-- {
			for j := 0; j < p; j++ {
				acc[j] = acc[j] + float32(weights[i]*theta[i][j])
			}
		}
	case Pairwise:
		for j := 0; j < p; j++ {
			col := make([]float32, n)
			for i := 0; i < n; i++ {
				col[i] = float32(weights[i] * theta[i][j])
			}
			acc[j] = pairwiseSum(col)
		}
	case Blocked:
		for j := 0; j < p; j++ {
			var a0, a1, a2, a3 float32
			i := 0
			for ; i+3 < n; i += 4 {
				a0 = a0 + float32(weights[i]*theta[i][j])
				a1 = a1 + float32(weights[i+1]*theta[i+1][j])
				a2 = a2 + float32(weights[i+2]*theta[i+2][j])
				a3 = a3 + float32(weights[i+3]*theta[i+3][j])
			}
			s := float32((a0 + a1) + (a2 + a3))
			for ; i < n; i++ {
				s = s + float32(weights[i]*theta[i][j])
			}
			acc[j] = s
		}
	case Widened:
		acc64 := make([]float64, p)
		for i := 0; i < n; i++ {
			for j := 0; j < p; j++ {
				acc64[j] += float64(weights[i]) * float64(theta[i][j])
			}
		}
		for j := 0; j < p; j++ {
			acc[j] = float32(acc64[j] / float64(S))
		}
		return acc
	case FusedMul:
		// No explicit float32() conversion around the product: the compiler
		// may contract this into an FMA.
		for i := 0; i < n; i++ {
			w := weights[i]
			row := theta[i]
			for j := 0; j < p; j++ {
				acc[j] += w * row[j]
			}
		}
	default:
		panic("dfrl-agg: unknown float strategy " + string(st))
	}

	for j := 0; j < p; j++ {
		acc[j] = acc[j] / S
	}
	return acc
}

func pairwiseSum(x []float32) float32 {
	if len(x) <= 8 {
		var s float32
		for _, v := range x {
			s += v
		}
		return s
	}
	h := len(x) / 2
	return pairwiseSum(x[:h]) + pairwiseSum(x[h:])
}

// FloatRoot hashes float32 output the way the submitted prototype effectively
// did: raw little-endian float32 bytes into SHA3-256. Kept separate from
// StateRoot so that nothing on the consensus path can accidentally hash floats.
func FloatRoot(x []float32) [32]byte {
	b := make([]byte, 4*len(x))
	for i, v := range x {
		u := float32bits(v)
		b[4*i+0] = byte(u)
		b[4*i+1] = byte(u >> 8)
		b[4*i+2] = byte(u >> 16)
		b[4*i+3] = byte(u >> 24)
	}
	return sha3.Sum256(b)
}
