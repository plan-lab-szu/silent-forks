package dfrlagg

// divRoundHalfEven divides num by den, rounding to nearest with ties going to
// the even quotient. This is the single rounding point of the whole protocol
// (§3.2). The rule is specified explicitly rather than inherited from any
// language default: Go truncates toward zero, Python floors, and the two
// disagree on negative operands (§4, divergence source 4).
//
// Preconditions: den > 0, and 2*|num mod den| must not overflow. Since
// |num mod den| < den, that holds for any den <= 2^62; SumBound (S < 2^29)
// clears it by 33 bits.
//
// Rounding to even is chosen over truncation because truncation always moves a
// nonintegral quotient toward zero and therefore adds directional shrinkage.
func divRoundHalfEven(num, den int64) int64 {
	q, r := num/den, num%den // q truncates toward zero; r takes the sign of num
	if r == 0 {
		return q
	}
	ar := r
	if ar < 0 {
		ar = -ar
	}
	switch twice := 2 * ar; {
	case twice > den:
		if num >= 0 {
			q++
		} else {
			q--
		}
	case twice == den:
		if q&1 != 0 { // tie: step toward the even quotient
			if num >= 0 {
				q++
			} else {
				q--
			}
		}
	}
	return q
}

// validateUpdates enforces §2.2 and §4.3 on a submission set and returns the
// exact weight sum S. It rejects rather than saturates.
func validateUpdates(ups []Update) (int64, error) {
	if len(ups) < 1 || len(ups) >= ClientBound {
		return 0, ErrClientCount
	}
	seen := make(map[Addr]struct{}, len(ups))
	var S int64
	for i := range ups {
		if _, dup := seen[ups[i].Addr]; dup {
			// §4.1 keeps only the last submission per client, so a duplicate
			// address reaching the aggregator means the caller violated the
			// membership rule.
			return 0, ErrDuplicateAddr
		}
		seen[ups[i].Addr] = struct{}{}

		if ups[i].Weight < 1 || ups[i].Weight >= WeightBound {
			return 0, ErrWeightRange
		}
		if err := ups[i].Tensor.Validate(); err != nil {
			return 0, err
		}
		if !ups[0].Tensor.sameShape(ups[i].Tensor) {
			return 0, ErrShapeMismatch
		}
		S += ups[i].Weight
	}
	if S < 1 || S >= SumBound {
		return 0, ErrSumRange
	}
	return S, nil
}

// FedAvg computes the weighted average Theta[j] = sum_n(s_n * theta_n[j]) / S
// with exact int64 accumulation and a single rounding step (§3.1).
//
// Proposition 2: integer addition is associative and commutative, and
// Proposition 1 rules out overflow, so the accumulator — and therefore the
// output bytes — do not depend on the order in which updates are visited.
// Go's randomized map iteration order, SIMD reduction trees and parallel
// reduction strategies all become irrelevant.
func FedAvg(ups []Update) (Tensor, error) {
	S, err := validateUpdates(ups)
	if err != nil {
		return Tensor{}, err
	}
	p := len(ups[0].Tensor.Data)
	acc := make([]int64, p)
	for i := range ups {
		s := ups[i].Weight
		d := ups[i].Tensor.Data
		for j := 0; j < p; j++ {
			acc[j] += s * d[j] // exact: |acc| < 2^53 by Proposition 1
		}
	}
	out := Tensor{Dims: append([]uint32(nil), ups[0].Tensor.Dims...), F: ups[0].Tensor.F, Data: acc}
	for j := 0; j < p; j++ {
		acc[j] = divRoundHalfEven(acc[j], S)
	}
	// The result is a weighted average of in-range values, so it is in range;
	// we assert it rather than assume it.
	if err := out.Validate(); err != nil {
		return Tensor{}, err
	}
	return out, nil
}
