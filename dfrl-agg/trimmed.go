package dfrlagg

import "sort"

// TrimmedMean is the second primitive (OP_ROBUSTAGG, U3): coordinate-wise
// trimmed mean over an integer domain, discarding the k smallest and k largest
// values at each coordinate and averaging the rest with equal weight.
//
// The order is numerical. Equal values are interchangeable: stable and
// unstable sorting may retain different client identities at a boundary, but
// the retained numerical multiset and aggregate are unchanged.
//
// Weights are deliberately ignored: trimming exists to bound the influence of
// any single client, which a weighted variant would reintroduce.
func TrimmedMean(ups []Update, k int) (Tensor, error) {
	if _, err := validateUpdates(ups); err != nil {
		return Tensor{}, err
	}
	n := len(ups)
	if k < 0 || 2*k >= n {
		return Tensor{}, ErrTrimParam
	}
	p := len(ups[0].Tensor.Data)
	keep := int64(n - 2*k)

	col := make([]int64, n)
	out := make([]int64, p)
	for j := 0; j < p; j++ {
		for i := range ups {
			col[i] = ups[i].Tensor.Data[j]
		}
		sort.Slice(col, func(a, b int) bool {
			return col[a] < col[b]
		})
		var acc int64
		for i := k; i < n-k; i++ {
			acc += col[i] // exact: at most 2^10 terms below 2^24
		}
		out[j] = divRoundHalfEven(acc, keep)
	}
	t := Tensor{Dims: append([]uint32(nil), ups[0].Tensor.Dims...), F: ups[0].Tensor.F, Data: out}
	if err := t.Validate(); err != nil {
		return Tensor{}, err
	}
	return t, nil
}
