// Copyright 2026 The DT-DFRL authors.
// This file adds a research precompile to go-ethereum; it is not part of
// upstream go-ethereum and is gated behind the DFRL fork timestamp
// (params.ChainConfig.DFRLTime), preserving upstream execution behavior
// before activation.

package vm

import (
	"encoding/binary"
	"errors"
	"math"
	"math/bits"

	dfrlagg "dfrl-agg"
	"github.com/ethereum/go-ethereum/common"
)

// DFRLFedAvgAddress is the address of the deterministic FedAvg precompile.
//
// Address choice, per HANDBOOK.md's opcode-selection discipline: avoid
// 0x01-0x11 (standard precompiles through Osaka) and 0x0100 (EIP-7212
// P256Verify). This paper's own reply-letter text commits to "chain ID /
// fork block gating" plus an address in the private-chain reserved range;
// 0x0101 is the next slot after P256Verify and is unoccupied as of the
// go-ethereum commit this fork is pinned to (see HANDBOOK.md "opcode 事实").
var DFRLFedAvgAddress = common.BytesToAddress([]byte{0x01, 0x01})

// ---------------------------------------------------------------------------
// Input encoding.
//
// This is the ABI the FedAvg precompile speaks. It is deliberately NOT the
// dfrl-agg canonical tensor encoding by itself: a precompile call aggregates
// SEVERAL clients' tensors at once, so the input is a length-prefixed sequence
// of (weight, canonical-tensor) pairs. Every tensor is decoded with
// dfrlagg.Decode, which already enforces the canonical-encoding and range
// checks from DETERMINISM_SPEC.md section 4.2 -- this file adds no separate
// validation logic for the tensor bytes, so there is exactly one place that
// decides what a legal tensor looks like.
//
// Layout, all big-endian, no padding:
//
//	numClients   uint32
//	repeated numClients times:
//	    weight     uint64
//	    tensorLen  uint32
//	    tensorData [tensorLen]byte   (dfrl-agg canonical encoding)
//
// Output: the canonical encoding (dfrlagg.Encode) of the aggregated tensor.
// A caller that wants the state root hashes the output itself -- the
// precompile returns the tensor, not a hash of it, so the caller controls
// what commitment scheme wraps it (this matters for U4, where the on-chain
// commitment is a Merkle root over metadata, not a raw hash of the payload).

const dfrlInputHeaderLen = 4     // numClients
const dfrlEntryHeaderLen = 8 + 4 // weight + tensorLen

var (
	errDFRLTruncated = errors.New("dfrl: truncated input")
	errDFRLTrailing  = errors.New("dfrl: trailing bytes after last entry")
	errDFRLNoClients = errors.New("dfrl: zero clients")
	errDFRLClients   = errors.New("dfrl: client count exceeds active protocol bound")
	errDFRLTensor    = errors.New("dfrl: tensor decode failed")
	errDFRLF         = errors.New("dfrl: tensor F differs from active protocol value")
)

// decodeDFRLInput parses the ABI above into dfrl-agg Updates. Distinct client
// identities are synthesized from entry positions because the precompile does
// not receive per-entry senders; the calling contract enforces one entry per
// submitting account. The active fork fixes F=DefaultF even though the generic
// library and conformance vectors support other protocol configurations.
func decodeDFRLInput(input []byte) ([]dfrlagg.Update, error) {
	if len(input) < dfrlInputHeaderLen {
		return nil, errDFRLTruncated
	}
	n := binary.BigEndian.Uint32(input)
	if n == 0 {
		return nil, errDFRLNoClients
	}
	// Check the protocol bound before using n as an allocation capacity.
	// Otherwise a four-byte adversarial header could force a huge allocation
	// even though the aggregation library would reject the client count later.
	if uint64(n) >= uint64(dfrlagg.ClientBound) {
		return nil, errDFRLClients
	}
	off := dfrlInputHeaderLen
	ups := make([]dfrlagg.Update, 0, n)
	for i := uint32(0); i < n; i++ {
		if len(input)-off < dfrlEntryHeaderLen {
			return nil, errDFRLTruncated
		}
		weight := int64(binary.BigEndian.Uint64(input[off:]))
		off += 8
		tlen := binary.BigEndian.Uint32(input[off:])
		off += 4
		if uint32(len(input)-off) < tlen {
			return nil, errDFRLTruncated
		}
		tensor, err := dfrlagg.Decode(input[off : off+int(tlen)])
		if err != nil {
			return nil, errDFRLTensor
		}
		if tensor.F != dfrlagg.DefaultF {
			return nil, errDFRLF
		}
		off += int(tlen)

		var addr dfrlagg.Addr
		addr[19] = byte(i + 1)
		addr[18] = byte((i + 1) >> 8)
		ups = append(ups, dfrlagg.Update{Addr: addr, Weight: weight, Tensor: tensor})
	}
	if off != len(input) {
		return nil, errDFRLTrailing
	}
	return ups, nil
}

// ---------------------------------------------------------------------------
// Gas.
//
// Model per R2-10: G = C_base + mu1*|Theta| + mu2*N*|Theta|, an affine model
// in the PARAMETER count that additionally scales with client count -- not
// the single-variable affine model the submitted manuscript used, which R2-10
// correctly flagged as ignoring N.
//
// mu1, mu2 and C_base below are a weighted least-squares fit (weight 1/ns^2,
// i.e. minimizing RELATIVE error) to wall-clock timings from
// TestFedAvgGasBenchmark in contracts_dfrl_bench_test.go, measured on this
// development machine (see fit_gas_model.py and HANDBOOK.md for the exact
// commands, the raw grid, and why an unweighted fit is wrong here: it is
// dominated by the largest |Theta| rows and mispredicts the small-|Theta|
// regime -- exactly the calldata-dominated regime R1-6c and R2-10 care about
// -- by 40-60x).
//
// This is explicitly NOT a validator-hardware calibration: it times Go code
// on a Windows development workstation, not a testbed node (Phase 3c does not
// exist yet). Converting nanoseconds to gas at a 1:1 rate is a stated
// REPORTING CONVENTION, not a claim about Ethereum's real gas/time ratio --
// there is no single agreed constant for that, and picking one without saying
// so would be exactly the "gas number as unexplained constant" complaint
// R2-10 raised about the submitted manuscript. What the constants below DO
// support honestly: the functional SHAPE (affine in |Theta|, with an
// additional N*|Theta| term R2-10 asked for) and its monotonicity, which do
// not depend on the ns-to-gas conversion rate. Re-run and refit once Phase 3c
// has real testbed nodes; until then this is "measured and disclosed", not
// "final".
const (
	dfrlGasBase      uint64 = 826 // C_base, ns (see comment above re: 1ns=1gas convention)
	dfrlGasPerParam  uint64 = 22  // mu1: marginal ns per parameter, holding N fixed
	dfrlGasPerNParam uint64 = 5   // mu2: additional marginal ns per (client, parameter) pair
)

// checkedDFRLGas computes the experimental execution-gas model without allowing
// any uint64 intermediate to wrap. Returning MaxUint64 makes an invalid or
// unrepresentable declared shape unaffordable under an ordinary block gas
// limit; unlike a finite cap, it cannot turn more work into a cheaper call.
// RobustAgg adds mu1*N*P*ceil(log2 N), matching the comparison-sort work
// performed independently at each of P coordinates.
func checkedDFRLGas(n, params uint64, robust bool) uint64 {
	if n == 0 || n >= uint64(dfrlagg.ClientBound) {
		return math.MaxUint64
	}
	gas := dfrlGasBase
	addProduct := func(a, b uint64) bool {
		if a != 0 && b > math.MaxUint64/a {
			return false
		}
		term := a * b
		if term > math.MaxUint64-gas {
			return false
		}
		gas += term
		return true
	}
	if !addProduct(dfrlGasPerParam, params) {
		return math.MaxUint64
	}
	if n != 0 && params > math.MaxUint64/n {
		return math.MaxUint64
	}
	nParams := n * params
	if !addProduct(dfrlGasPerNParam, nParams) {
		return math.MaxUint64
	}
	if robust {
		ceilLog2N := uint64(bits.Len64(n - 1))
		if ceilLog2N != 0 && nParams > math.MaxUint64/ceilLog2N {
			return math.MaxUint64
		}
		if !addProduct(dfrlGasPerParam, nParams*ceilLog2N) {
			return math.MaxUint64
		}
	}
	return gas
}

type fedAvgPrecompile struct{}

func (c *fedAvgPrecompile) Name() string { return "dfrlFedAvg" }

// RequiredGas must not decode tensor elements (that would make pricing as
// expensive as execution). It walks only the entry headers, charging against
// the largest declared tensor shape. A malformed later entry therefore cannot
// hide expensive parsing behind a small first tensor.
func (c *fedAvgPrecompile) RequiredGas(input []byte) uint64 {
	n, params := estimateShape(input)
	return checkedDFRLGas(uint64(n), uint64(params), false)
}

// estimateShape reads N and a parameter-count estimate from entry headers,
// without decoding tensor elements. It tolerates malformed input without
// panicking and retains an overstated tensorLen as a charged claim.
func estimateShape(input []byte) (n uint32, params uint32) {
	if len(input) < dfrlInputHeaderLen {
		return 0, uint32(len(input))
	}
	n = binary.BigEndian.Uint32(input)
	off := dfrlInputHeaderLen
	for i := uint32(0); i < n; i++ {
		if len(input)-off < dfrlEntryHeaderLen {
			if uint32(len(input)) > params {
				params = uint32(len(input))
			}
			return n, params
		}
		tlen := binary.BigEndian.Uint32(input[off+8:])
		claimed := uint32(0)
		if tlen >= 14 {
			// Rank-one tensors have a 14-byte header; higher ranks add four
			// bytes per dimension. Ceiling division therefore never
			// underestimates the element count represented by tensorLen.
			claimed = uint32((uint64(tlen) - 14 + 7) / 8)
		}
		if claimed > params {
			params = claimed
		}
		off += dfrlEntryHeaderLen
		if uint64(tlen) > uint64(len(input)-off) {
			return n, params
		}
		off += int(tlen)
	}
	return n, params
}

// Run performs the actual aggregation via dfrl-agg -- the SAME code the
// off-chain aggregator (fl-workload/aggbridge.py via cmd/dfrl-agg) and the
// fused opcode (instructions_dfrl.go) call. That shared implementation is what
// makes U6 (on-chain equals off-chain) and X6 (validators agree byte for
// byte) properties of the build rather than something to reconcile later.
func (c *fedAvgPrecompile) Run(input []byte) ([]byte, error) {
	ups, err := decodeDFRLInput(input)
	if err != nil {
		return nil, err
	}
	out, err := dfrlagg.FedAvg(ups)
	if err != nil {
		return nil, err
	}
	return dfrlagg.Encode(out)
}

// ---------------------------------------------------------------------------
// DFRLRobustAggAddress / robustAggPrecompile: OP_ROBUSTAGG (U3), the
// coordinate-wise trimmed mean. Same ABI as FedAvg with one extra 4-byte
// trim-count header field, because the shared decode/gas/dispatch plumbing
// already exists once FedAvg's is written and the marginal cost of adding it
// now is small compared to writing it again in Phase 4.
var DFRLRobustAggAddress = common.BytesToAddress([]byte{0x01, 0x02})

type robustAggPrecompile struct{}

func (c *robustAggPrecompile) Name() string { return "dfrlRobustAgg" }

func (c *robustAggPrecompile) RequiredGas(input []byte) uint64 {
	if len(input) < 4 {
		return math.MaxUint64
	}
	n, params := estimateShape(input[4:])
	return checkedDFRLGas(uint64(n), uint64(params), true)
}

func (c *robustAggPrecompile) Run(input []byte) ([]byte, error) {
	if len(input) < 4 {
		return nil, errDFRLTruncated
	}
	k := int(binary.BigEndian.Uint32(input))
	ups, err := decodeDFRLInput(input[4:])
	if err != nil {
		return nil, err
	}
	out, err := dfrlagg.TrimmedMean(ups, k)
	if err != nil {
		return nil, err
	}
	return dfrlagg.Encode(out)
}

// PrecompiledContractsDFRL layers the two DFRL precompiles on top of the
// Osaka set (see params.ChainConfig.IsDFRL's doc comment for why Osaka
// specifically). It is cloned rather than mutated in place, so the upstream
// Osaka set remains available unchanged before DFRL activation.
var PrecompiledContractsDFRL = func() PrecompiledContracts {
	m := make(PrecompiledContracts, len(PrecompiledContractsOsaka)+2)
	for k, v := range PrecompiledContractsOsaka {
		m[k] = v
	}
	m[DFRLFedAvgAddress] = &fedAvgPrecompile{}
	m[DFRLRobustAggAddress] = &robustAggPrecompile{}
	return m
}()

// PrecompiledAddressesDFRL mirrors the PrecompiledAddressesXxx pattern used
// for every other fork elsewhere in this file.
var PrecompiledAddressesDFRL []common.Address

func init() {
	for k := range PrecompiledContractsDFRL {
		PrecompiledAddressesDFRL = append(PrecompiledAddressesDFRL, k)
	}
}
