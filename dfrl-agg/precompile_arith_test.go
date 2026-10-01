package dfrlagg

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

// TestPrecompileArithDiagnosis compares the EXACT arithmetic the precompile
// uses (no float32() on the product — the "FusedMul" style) against the
// SeqFwd/SeqRev style (explicit float32()) for the natural witness window.
//
// The precompile's accumulation is:
//
//	acc[j] += e.weight * e.theta[j]          // no float32()
//
// while FloatAvg's SeqFwd is:
//
//	acc[j] = acc[j] + float32(weights[i]*theta[i][j])  // explicit float32()
//
// Go's spec allows extended precision for the unwrapped form. On amd64 the
// compiler may keep intermediates in double precision, which can absorb
// ordering differences that show up at float32 precision.
func TestPrecompileArithDiagnosis(t *testing.T) {
	theta, weights := loadRealRound(t, 0)
	n := len(theta)
	p := len(theta[0])
	t.Logf("round 0: N=%d clients, P=%d params", n, p)

	const windowSize = 16
	bestOff := 112 // the window we chose

	win := sliceTheta(theta, bestOff, windowSize)

	// SeqFwd vs SeqRev (what TestExtractNaturalWitness checked):
	seqFwd := FloatAvg(win, weights, SeqFwd)
	seqRev := FloatAvg(win, weights, SeqRev)
	seqFwdRoot := FloatRoot(seqFwd)
	seqRevRoot := FloatRoot(seqRev)

	t.Logf("SeqFwd root:  %x", seqFwdRoot)
	t.Logf("SeqRev root:  %x", seqRevRoot)
	t.Logf("SeqFwd == SeqRev: %v", seqFwdRoot == seqRevRoot)

	// FusedMul fwd vs FusedMul rev (what the precompile actually does):
	fusedFwd := fusedMulAvg(win, weights, false)
	fusedRev := fusedMulAvg(win, weights, true)
	fusedFwdRoot := FloatRoot(fusedFwd)
	fusedRevRoot := FloatRoot(fusedRev)

	t.Logf("FusedMul-fwd root:  %x", fusedFwdRoot)
	t.Logf("FusedMul-rev root:  %x", fusedRevRoot)
	t.Logf("FusedMul-fwd == FusedMul-rev: %v", fusedFwdRoot == fusedRevRoot)

	// Also check the existing FusedMul strategy (fwd only):
	fusedLib := FloatAvg(win, weights, FusedMul)
	fusedLibRoot := FloatRoot(fusedLib)
	t.Logf("FloatAvg(FusedMul) root: %x", fusedLibRoot)
	t.Logf("FloatAvg(FusedMul) == FusedMul-fwd: %v", fusedLibRoot == fusedFwdRoot)

	// Per-coordinate comparison:
	t.Log("\nPer-coordinate comparison:")
	for j := 0; j < windowSize; j++ {
		sf := seqFwd[j]
		sr := seqRev[j]
		ff := fusedFwd[j]
		fr := fusedRev[j]
		marker := ""
		if sf != sr {
			marker += " SEQ-DIFF"
		}
		if ff != fr {
			marker += " FUSED-DIFF"
		}
		if sf != ff {
			marker += " SEQ≠FUSED"
		}
		t.Logf("  coord %3d: seqFwd=%.10g seqRev=%.10g fusedFwd=%.10g fusedRev=%.10g%s",
			bestOff+j, sf, sr, ff, fr, marker)
	}

	// Now scan ALL windows for one where FusedMul-fwd ≠ FusedMul-rev:
	t.Log("\nScanning for FusedMul-divergent windows...")
	fusedDivergentCount := 0
	seqDivergentCount := 0
	for off := 0; off+windowSize <= p; off += windowSize {
		w := sliceTheta(theta, off, windowSize)

		sf := FloatAvg(w, weights, SeqFwd)
		sr := FloatAvg(w, weights, SeqRev)
		ff := fusedMulAvg(w, weights, false)
		fr := fusedMulAvg(w, weights, true)

		sfRoot := FloatRoot(sf)
		srRoot := FloatRoot(sr)
		ffRoot := FloatRoot(ff)
		frRoot := FloatRoot(fr)

		if sfRoot != srRoot {
			seqDivergentCount++
		}
		if ffRoot != frRoot {
			fusedDivergentCount++
			// Check diversity too.
			diverseCoords := 0
			for j := 0; j < windowSize; j++ {
				for i := 1; i < n; i++ {
					if w[i][j] != w[0][j] {
						diverseCoords++
						break
					}
				}
			}
			diffs := 0
			for j := 0; j < windowSize; j++ {
				if ff[j] != fr[j] {
					diffs++
				}
			}
			t.Logf("  FUSED-divergent window [%d, %d): %d/%d coords differ, diversity=%d",
				off, off+windowSize, diffs, windowSize, diverseCoords)
		}
	}
	t.Logf("total Seq-divergent windows: %d", seqDivergentCount)
	t.Logf("total Fused-divergent windows: %d", fusedDivergentCount)

	// Also check: does decoding from precompile binary format change anything?
	// The precompile decodes weights as uint64 then casts to float32.
	t.Log("\nChecking precompile decode path...")
	inputHex := encodeVulnPrecompileInput(win, weights)
	raw, _ := hexDecode(inputHex)
	entries := decodePrecompileEntries(raw)
	t.Logf("decoded %d entries from precompile binary format", len(entries))
	for i, e := range entries {
		t.Logf("  entry %d: weight=%.1f (orig=%.1f) theta[0]=%v (orig=%v)",
			i, e.weight, weights[i], e.theta[0], win[i][0])
	}
}

// fusedMulAvg mimics the precompile's exact arithmetic: no float32() on the
// product, so the compiler may use extended precision or FMA contraction.
func fusedMulAvg(theta [][]float32, weights []float32, reverse bool) []float32 {
	n := len(theta)
	params := len(theta[0])
	acc := make([]float32, params)
	var wsum float32
	if reverse {
		for i := n - 1; i >= 0; i-- {
			wsum += weights[i]
			for j := 0; j < params; j++ {
				acc[j] += weights[i] * theta[i][j]
			}
		}
	} else {
		for i := 0; i < n; i++ {
			wsum += weights[i]
			for j := 0; j < params; j++ {
				acc[j] += weights[i] * theta[i][j]
			}
		}
	}
	for j := 0; j < params; j++ {
		acc[j] /= wsum
	}
	return acc
}

type precompileEntry struct {
	weight float32
	theta  []float32
}

func decodePrecompileEntries(raw []byte) []precompileEntry {
	numClients := binary.BigEndian.Uint32(raw)
	off := 4
	entries := make([]precompileEntry, numClients)
	for i := range entries {
		w := binary.BigEndian.Uint64(raw[off:])
		off += 8
		p := int(binary.BigEndian.Uint32(raw[off:]))
		off += 4
		theta := make([]float32, p)
		for j := 0; j < p; j++ {
			theta[j] = math.Float32frombits(binary.BigEndian.Uint32(raw[off:]))
			off += 4
		}
		entries[i] = precompileEntry{
			weight: float32(w),
			theta:  theta,
		}
	}
	return entries
}

func hexDecode(s string) ([]byte, error) {
	b := make([]byte, len(s)/2)
	for i := 0; i < len(s); i += 2 {
		hi := unhex(s[i])
		lo := unhex(s[i+1])
		b[i/2] = hi<<4 | lo
	}
	return b, nil
}

func unhex(c byte) byte {
	switch {
	case c >= '0' && c <= '9':
		return c - '0'
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10
	}
	return 0
}

// TestPrecompileWeightDecode checks whether the weight round-trip through
// uint64 encoding changes the float32 value. The precompile stores weights
// as uint64(float32), so non-integer weights would be truncated.
func TestPrecompileWeightDecode(t *testing.T) {
	theta, weights := loadRealRound(t, 0)
	_ = theta

	for i, w := range weights {
		u := uint64(w)
		back := float32(u)
		if w != back {
			t.Logf("weight %d: original=%.10g → uint64=%d → float32=%.10g MISMATCH",
				i, w, u, back)
		} else {
			t.Logf("weight %d: %.10g → %d → %.10g OK", i, w, u, back)
		}
	}

	// Also check: does the precompile's Run function divide by wsum computed
	// as float32 sum of the weights, while FloatAvg divides by its own wsum?
	// Are these the same?
	var wsumFwd float32
	for i := 0; i < len(weights); i++ {
		wsumFwd += weights[i]
	}
	var wsumRev float32
	for i := len(weights) - 1; i >= 0; i-- {
		wsumRev += weights[i]
	}
	t.Logf("wsum fwd=%.10g rev=%.10g same=%v", wsumFwd, wsumRev, wsumFwd == wsumRev)
	t.Logf("wsum bits: fwd=%08x rev=%08x",
		math.Float32bits(wsumFwd), math.Float32bits(wsumRev))

	// What about through the precompile path (uint64 -> float32)?
	var wsumPrecompFwd float32
	for i := 0; i < len(weights); i++ {
		wsumPrecompFwd += float32(uint64(weights[i]))
	}
	var wsumPrecompRev float32
	for i := len(weights) - 1; i >= 0; i-- {
		wsumPrecompRev += float32(uint64(weights[i]))
	}
	t.Logf("wsum (precomp decode) fwd=%.10g rev=%.10g same=%v",
		wsumPrecompFwd, wsumPrecompRev, wsumPrecompFwd == wsumPrecompRev)

	// Check: is float32(uint64(w)) == w for all weights?
	fmt.Println()
	for i, w := range weights {
		via := float32(uint64(w))
		t.Logf("weight %d: original float32 bits=%08x, via uint64 bits=%08x, same=%v",
			i, math.Float32bits(w), math.Float32bits(via), w == via)
	}
}
