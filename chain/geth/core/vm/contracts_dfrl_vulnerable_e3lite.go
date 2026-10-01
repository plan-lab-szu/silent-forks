//go:build e3lite_vulnerable

// Copyright 2026 The DT-DFRL authors.
//
// THIS FILE IS A NEGATIVE-CONTROL ARTIFACT FOR THE E3-LITE LIVENESS-ATTACK
// DEMONSTRATION. IT IS NOT PART OF THE DFRL SYSTEM.
//
// It exists to answer one question empirically: does the state-root
// divergence this paper diagnoses (see dfrl-agg/floatref.go, C1) actually
// translate into a real consensus liveness failure on a real multi-client
// devnet, or is that only a plausible-sounding extrapolation? Building a
// deliberately naive aggregation precompile -- the design this paper argues
// against, and the design the group's own prior work (OJ-COMS 2024) shipped
// -- is the minimal way to find out: run it as a mandatory state transition
// on heterogeneous execution-layer nodes and see whether they actually
// disagree on the resulting payload.
//
// The isolation from the real system is enforced at COMPILE TIME, not by a
// runtime flag someone could leave on by accident:
//
//  1. This file only exists in a binary built with `-tags e3lite_vulnerable`.
//     The paper's and the repo's own build/test instructions never pass that
//     tag, so `go build ./cmd/geth` and every CI job produce a binary that
//     does not even contain this code.
//  2. It is registered at its own address (0x0103), never added to
//     PrecompiledContractsDFRL. A node running the real DFRL fork -- even
//     with the DFRL fork timestamp active -- cannot reach this precompile
//     regardless of build tags, because activePrecompiledContracts (see
//     contracts.go) only ever dereferences PrecompiledContractsDFRL for
//     rules.IsDFRL. Wiring this address in requires the separate init()
//     below, itself only compiled under the tag.
//  3. It deliberately does NOT import dfrl-agg's floatref.go. That file's own
//     doc comment asserts "nothing else in this package calls into it; the
//     only callers are tests and cmd/dfrl-agg's diverge subcommand" -- an
//     invariant worth keeping literally true rather than quietly voiding it
//     from a different module. This file instead reimplements the same
//     mechanism floatref.go calls FusedMul: accumulate in float32 without an
//     explicit rounding barrier around the product, so whether the multiply
//     and the add fuse into a single FMA instruction is left to the Go
//     compiler's per-architecture code generation -- exactly what
//     TestFloatDivergenceReport and TestMinimalCounterexample (dfrl-agg) and
//     the arm64-divergence.yml CI workflow have measured as genuinely
//     architecture-dependent (FMADDS/FMADDD on arm64, no FMA on amd64;
//     confirmed independently on windows/arm64, linux/arm64, and linux/amd64
//     hardware -- see HANDBOOK.md and dfrl-agg/minimal_counterexample_test.go
//     for the specific witnesses). Duplication of ~10 lines is the price of
//     keeping that module boundary honest; it is worth paying once.
//
// If this experiment succeeds (a real INVALID payload / real finality stall
// on the Kurtosis devnet), the paper cites it as the empirical answer to "so
// what, concretely, goes wrong". If it does not reproduce at devnet scale,
// that is also reportable: the paper's liveness-attack framing would need to
// stay at the level of the formalized W(r) < 2/3 argument (Tier-1 item 3)
// without an end-to-end demo, and this file's existence and result either
// way belongs in Section 9's honesty disclaimer, not silently dropped.

package vm

import (
	"encoding/binary"
	"errors"
	"math"

	"github.com/ethereum/go-ethereum/common"
)

// DFRLVulnerableFloatAggAddress is deliberately far from DFRLFedAvgAddress
// (0x0101) and DFRLRobustAggAddress (0x0102) in numbering intent even though
// it is numerically adjacent (0x0103): it is not "DFRL's third precompile",
// it is a standalone artifact that happens to share the same reserved
// address range for convenience. See the file-level comment for why it can
// never appear in a binary built without -tags e3lite_vulnerable.
var DFRLVulnerableFloatAggAddress = common.BytesToAddress([]byte{0x01, 0x03})

// ---------------------------------------------------------------------------
// Input encoding. Distinct from decodeDFRLInput (contracts_dfrl.go) on
// purpose: that decoder expects dfrl-agg's canonical QUANTIZED tensor
// encoding, because the real system's whole point is that clients quantize
// before submitting (U1) so the consensus-critical path never touches
// floating point. This precompile exists to emulate the design that skips
// that step -- clients submit raw float32 model weights, the chain
// aggregates them in float32 as part of a mandatory state transition. So the
// input here is raw float32, not a dfrl-agg Tensor.
//
// Layout, all big-endian, no padding:
//
//	numClients   uint32
//	repeated numClients times:
//	    weight     uint64
//	    paramCount uint32
//	    theta      [paramCount]float32   (4 bytes each, IEEE 754 big-endian)
//
// Output: [paramCount]float32, big-endian, the weighted average. A caller
// that wants a commitment hashes this output itself, same convention as
// fedAvgPrecompile.

const dfrlVulnInputHeaderLen = 4     // numClients
const dfrlVulnEntryHeaderLen = 8 + 4 // weight + paramCount

var (
	errDFRLVulnTruncated     = errors.New("dfrl-vuln: truncated input")
	errDFRLVulnTrailing      = errors.New("dfrl-vuln: trailing bytes after last entry")
	errDFRLVulnNoClients     = errors.New("dfrl-vuln: zero clients")
	errDFRLVulnShapeMismatch = errors.New("dfrl-vuln: client parameter counts differ")
)

type dfrlVulnEntry struct {
	weight float32
	theta  []float32
}

func decodeDFRLVulnInput(input []byte) ([]dfrlVulnEntry, error) {
	if len(input) < dfrlVulnInputHeaderLen {
		return nil, errDFRLVulnTruncated
	}
	n := binary.BigEndian.Uint32(input)
	if n == 0 {
		return nil, errDFRLVulnNoClients
	}
	off := dfrlVulnInputHeaderLen
	entries := make([]dfrlVulnEntry, 0, n)
	wantParams := -1
	for i := uint32(0); i < n; i++ {
		if len(input)-off < dfrlVulnEntryHeaderLen {
			return nil, errDFRLVulnTruncated
		}
		weight := binary.BigEndian.Uint64(input[off:])
		off += 8
		params := int(binary.BigEndian.Uint32(input[off:]))
		off += 4
		need := params * 4
		if len(input)-off < need {
			return nil, errDFRLVulnTruncated
		}
		if wantParams == -1 {
			wantParams = params
		} else if params != wantParams {
			return nil, errDFRLVulnShapeMismatch
		}
		theta := make([]float32, params)
		for j := 0; j < params; j++ {
			bits := binary.BigEndian.Uint32(input[off:])
			theta[j] = math.Float32frombits(bits)
			off += 4
		}
		entries = append(entries, dfrlVulnEntry{weight: float32(weight), theta: theta})
	}
	if off != len(input) {
		return nil, errDFRLVulnTrailing
	}
	return entries, nil
}

// ---------------------------------------------------------------------------
// Gas. Not a calibrated model -- this precompile is a negative-control demo
// artifact, not a claim about the real system's cost, so it reuses the real
// FedAvg precompile's affine constants (dfrlGasBase/dfrlGasPerParam,
// contracts_dfrl.go) purely so RequiredGas is monotonic and non-zero, never
// as a statement that this is what the vulnerable design would actually cost
// on a real chain.

type vulnerableFloatAggPrecompile struct{}

func (c *vulnerableFloatAggPrecompile) Name() string { return "dfrlVulnerableFloatAgg_E3LITE_ONLY" }

func (c *vulnerableFloatAggPrecompile) RequiredGas(input []byte) uint64 {
	n, params := estimateShapeVuln(input)
	return checkedDFRLGas(uint64(n), uint64(params), false)
}

// estimateShapeVuln walks the vulnerable precompile's input layout
// (numClients + N×(weight uint64 + paramCount uint32 + paramCount×float32))
// to extract the client count and max parameter count for gas pricing.
// estimateShape (contracts_dfrl.go) cannot be reused here because it expects
// the FedAvg tensor format where the entry payload length is in bytes, not in
// float32 element count.
func estimateShapeVuln(input []byte) (n uint32, params uint32) {
	if len(input) < dfrlVulnInputHeaderLen {
		return 0, 0
	}
	n = binary.BigEndian.Uint32(input)
	off := dfrlVulnInputHeaderLen
	for i := uint32(0); i < n; i++ {
		if len(input)-off < dfrlVulnEntryHeaderLen {
			return n, params
		}
		pcount := binary.BigEndian.Uint32(input[off+8:])
		if pcount > params {
			params = pcount
		}
		off += dfrlVulnEntryHeaderLen
		byteLen := uint64(pcount) * 4
		if byteLen > uint64(len(input)-off) {
			return n, params
		}
		off += int(byteLen)
	}
	return n, params
}

// Run computes sum(weight_i * theta_i) / sum(weight_i) in float32, entry by
// entry, WITHOUT wrapping weight_i*theta_i in an explicit float32(...)
// conversion. That absence is the entire point: it leaves the compiler free
// to contract the multiply-add into a single FMA instruction, a choice the
// Go language spec permits and which floatref.go's FusedMul strategy (and
// this file's own package-level doc comment) documents as measured
// architecture-dependent. This is the mandatory-state-transition analogue of
// that same unguarded accumulation, now reachable from a real EVM call
// instead of only from a Go test.
func (c *vulnerableFloatAggPrecompile) Run(input []byte) ([]byte, error) {
	entries, err := decodeDFRLVulnInput(input)
	if err != nil {
		return nil, err
	}
	params := len(entries[0].theta)
	// Accumulation order is selected at COMPILE TIME (see
	// contracts_dfrl_vulnerable_order_{fwd,rev}.go). Two binaries built from
	// this same source with different tags are two conforming implementations
	// of the same rule that need not agree -- axis 1 of the diagnosis. Making
	// it a build tag rather than a runtime flag matters: the experiment claims
	// independently built clients disagree, not that one client can be
	// switched.
	acc, wsum := dfrlVulnAccumulate(entries, params)
	out := make([]byte, params*4)
	for j := 0; j < params; j++ {
		v := acc[j] / wsum
		binary.BigEndian.PutUint32(out[j*4:], math.Float32bits(v))
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Wiring: add this precompile to PrecompiledContractsDFRL ONLY under this
// build tag. PrecompiledContractsDFRL (contracts_dfrl.go) is fully
// constructed by its package-level var initializer before any init() in this
// package runs (Go spec: package-level variable initialization completes
// before init() functions execute), so mutating the map here is well-defined
// and race-free -- this is the same pattern PrecompiledAddressesDFRL already
// uses to derive its slice from the map in contracts_dfrl.go's own init().
func init() {
	PrecompiledContractsDFRL[DFRLVulnerableFloatAggAddress] = &vulnerableFloatAggPrecompile{}
	PrecompiledAddressesDFRL = append(PrecompiledAddressesDFRL, DFRLVulnerableFloatAggAddress)
}

// DFRLVulnOrderName reports the accumulation order compiled into this binary
// ("seq-fwd" or "seq-rev"). Exported so that a test in another package can
// record which of the two variants produced a given post-state root, which is
// the whole point of the cross-binary comparison.
func DFRLVulnOrderName() string { return dfrlVulnOrderName }
