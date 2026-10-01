// Package dfrlagg implements the deterministic aggregation primitives specified
// in DETERMINISM_SPEC.md. Every symbol here sits on the consensus execution
// path, so this package has ZERO external dependencies and performs no
// floating-point arithmetic. See floatref.go for the deliberate exception
// (negative control for C1), which is never reachable from Aggregate.
package dfrlagg

import "errors"

// Protocol parameters. See DETERMINISM_SPEC.md §2.2.
//
// Bit budget (Proposition 1): a + b + c = 24 + 19 + 10 = 53 < 63.
// b is 19 rather than 20 purely to widen that margin from 9 bits to 10.
// It is NOT forced by divRoundHalfEven's precondition (§3.2): that needs
// only den <= 2^62, and even b=20,c=10 (S < 2^30) would clear it by 32 bits.
// An earlier comment here claimed the requirement was S < 2^30, which is the
// bound for 32-bit arithmetic, not the 64-bit arithmetic this package uses.
const (
	// ParamBits is a: |theta_hat| < 2^ParamBits.
	ParamBits = 24
	// WeightBits is b: 1 <= s_n < 2^WeightBits.
	WeightBits = 19
	// ClientBits is c: 1 <= N < 2^ClientBits.
	ClientBits = 10

	ParamBound  int64 = 1 << ParamBits  // exclusive
	WeightBound int64 = 1 << WeightBits // exclusive
	ClientBound int   = 1 << ClientBits // exclusive

	// SumBound is the exclusive upper bound on S = sum(s_n), implied by
	// WeightBits + ClientBits = 29.
	SumBound int64 = 1 << (WeightBits + ClientBits)

	// DefaultF is the fixed-point fractional bit count. Protocol parameter,
	// network-wide, not node-configurable. Final value pending X10.
	DefaultF uint8 = 20

	// MaxRank bounds the tensor rank accepted by the canonical codec.
	MaxRank = 8
)

// Errors returned on invalid input. Out-of-range submissions are REJECTED,
// never wrapped around (§2.2).
var (
	ErrParamRange    = errors.New("dfrl-agg: quantized parameter out of range")
	ErrWeightRange   = errors.New("dfrl-agg: weight out of range")
	ErrClientCount   = errors.New("dfrl-agg: client count out of range")
	ErrSumRange      = errors.New("dfrl-agg: weight sum out of range")
	ErrShapeMismatch = errors.New("dfrl-agg: tensor shape or F mismatch")
	ErrEncoding      = errors.New("dfrl-agg: non-canonical encoding")
	ErrTrimParam     = errors.New("dfrl-agg: invalid trim count")
	ErrDuplicateAddr = errors.New("dfrl-agg: duplicate client address")
)
