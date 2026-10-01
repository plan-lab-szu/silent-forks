// Copyright 2026 The DT-DFRL authors.
// This file is not part of upstream go-ethereum.

package vm

import (
	dfrlagg "dfrl-agg"

	"github.com/holiman/uint256"
)

// FEDAVG stack convention, top to bottom (first-listed = top, matching this
// codebase's convention for every multi-arg opcode -- see CALLDATACOPY,
// STATICCALL): inOffset, inSize, n, outOffset, outSize.
//
//   - inOffset/inSize: memory region holding the SAME calldata ABI the
//     FedAvg precompile decodes (core/vm/contracts_dfrl.go): a
//     length-prefixed sequence of (weight, canonical-tensor) pairs.
//   - n: declared client count. Charged for by gasFedAvg BEFORE inOffset's
//     bytes are guaranteed to be resized into memory (dynamicGas runs prior
//     to mem.Resize, so it cannot safely read memory content -- see the
//     interpreter's run loop). The precompile does not need this because a
//     CALL's argsOffset/argsSize are already materialized into a []byte
//     before RequiredGas ever runs; a fused opcode reading raw EVM memory has
//     no such guarantee, so this is an explicit argument here where the
//     precompile could infer it from a peek at the input bytes.
//   - outOffset/outSize: caller-declared output buffer. FEDAVG succeeds only
//     when outSize can hold the complete canonical result; it never reports
//     success for a truncated encoding.
//
// Push: a single success flag (1 = aggregated and written, 0 = rejected).
// This is a deliberate departure from returning a hard error: a plain
// arithmetic-style opcode has no CALL-style success/failure channel of its
// own to report validation failures through, and INVALID/REVERT-style
// abort-the-whole-frame semantics would make this primitive strictly worse to
// use than the precompile on invalid input (the precompile's caller can
// branch on STATICCALL's success bit; a hard-erroring opcode's caller
// cannot). Pushing a flag preserves that symmetry, which matters for X9's
// precompile-vs-opcode comparison being a fair one.
//
// Gas is charged for the DECLARED shape (n, inSize) before validation runs,
// same policy as the precompile's RequiredGas: an adversarial submission that
// turns out to be invalid still pays for the shape it claimed.
func opFedAvg(pc *uint64, evm *EVM, scope *ScopeContext) ([]byte, error) {
	inOffset, inSize, n := scope.Stack.pop3()
	outOffset := scope.Stack.pop()
	outSize := scope.Stack.pop()

	inOffset64, of1 := inOffset.Uint64WithOverflow()
	inSize64, of2 := inSize.Uint64WithOverflow()
	n64, ofN := n.Uint64WithOverflow()
	outOffset64, of3 := outOffset.Uint64WithOverflow()
	outSize64, of4 := outSize.Uint64WithOverflow()
	if of1 || of2 || ofN || of3 || of4 {
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}

	inBytes := scope.Memory.GetPtr(inOffset64, inSize64)
	// dynamicGas must use the stack-declared N because it runs before memory
	// expansion. Bind that declaration to the payload before doing any
	// input-dependent work, so the caller cannot buy gas for one shape and
	// execute another.
	if len(inBytes) < dfrlInputHeaderLen || uint64(readDFRLClientCount(inBytes)) != n64 {
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}
	ups, err := decodeDFRLInput(inBytes)
	if err != nil {
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}
	result, err := dfrlagg.FedAvg(ups)
	if err != nil {
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}
	enc, err := dfrlagg.Encode(result)
	if err != nil {
		// Unreachable if FedAvg succeeded (it only returns already-valid
		// tensors), kept as a hard failure signal rather than silently
		// hidden so a future change to Encode's validation cannot regress
		// into a silent wrong-success return.
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}

	if outSize64 < uint64(len(enc)) {
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}
	scope.Memory.Set(outOffset64, uint64(len(enc)), enc)
	scope.Stack.push(new(uint256.Int).SetOne())
	return nil, nil
}

// gasFedAvg mirrors the precompile's R2-10 model (base + mu1*|Theta| +
// mu2*N*|Theta|, see contracts_dfrl.go for the disclosed benchmark this is
// fit from) plus the standard EVM memory-expansion charge every
// memory-touching opcode pays. |Theta| cannot be read from memory content
// here (dynamicGas runs before the interpreter resizes memory to cover the
// input region -- see the doc comment on opFedAvg's `n` argument), so it is
// conservatively approximated from the total input bytes per client -- a
// proxy, not an exact count (see
// TestFedAvgOpcodeGasMonotonic in instructions_dfrl_test.go).
func gasFedAvg(evm *EVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (GasCosts, error) {
	memGas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return GasCosts{}, err
	}
	inSize, of1 := stack.back(1).Uint64WithOverflow()
	n, of2 := stack.back(2).Uint64WithOverflow()
	if of1 || of2 {
		return GasCosts{}, ErrGasUintOverflow
	}
	if n == 0 || n >= uint64(dfrlagg.ClientBound) {
		return GasCosts{}, ErrGasUintOverflow
	}
	// A valid input uses eight bytes per parameter, plus headers. Ceiling
	// division therefore overestimates rather than underestimates P.
	bytesPerClient := inSize / n
	if inSize%n != 0 {
		bytesPerClient++
	}
	paramsPerClient := bytesPerClient / 8
	if bytesPerClient%8 != 0 {
		paramsPerClient++
	}
	execGas := checkedDFRLGas(n, paramsPerClient, false)
	if execGas == ^uint64(0) {
		return GasCosts{}, ErrGasUintOverflow
	}
	total := memGas + execGas
	if total < memGas { // overflow of the addition itself
		return GasCosts{}, ErrGasUintOverflow
	}
	return GasCosts{ExecutionGas: total}, nil
}

// memoryFedAvg reports the memory span the opcode will touch: the larger of
// its input region and its caller-declared output region, matching the
// pattern of every other variable-memory opcode in this package (e.g.
// memoryCallDataCopy, memoryKeccak256).
func memoryFedAvg(stack *Stack) (uint64, bool) {
	inSz, of1 := calcMemSize64(stack.back(0), stack.back(1))
	outSz, of2 := calcMemSize64(stack.back(3), stack.back(4))
	if of1 || of2 {
		return 0, true
	}
	if outSz > inSz {
		return outSz, false
	}
	return inSz, false
}

// ROBUSTAGG stack convention, top to bottom: inOffset, inSize, n, outOffset,
// outSize -- identical to FEDAVG's. The memory region at inOffset holds the
// SAME layout the RobustAgg precompile decodes (contracts_dfrl.go's
// robustAggPrecompile.Run): a 4-byte big-endian trim count k, followed by the
// same length-prefixed (weight, canonical-tensor) sequence FEDAVG reads. k
// travels inside the memory payload rather than as its own stack argument
// because it is data the precompile ABI already puts there, not a shape
// parameter dynamicGas needs before memory is resized (unlike n, which gas
// charging does need -- see opFedAvg's doc comment for why n is a stack arg
// and k is not: k does not affect gasRobustAgg's cost function).
func opRobustAgg(pc *uint64, evm *EVM, scope *ScopeContext) ([]byte, error) {
	inOffset, inSize, n := scope.Stack.pop3()
	outOffset := scope.Stack.pop()
	outSize := scope.Stack.pop()

	inOffset64, of1 := inOffset.Uint64WithOverflow()
	inSize64, of2 := inSize.Uint64WithOverflow()
	n64, ofN := n.Uint64WithOverflow()
	outOffset64, of3 := outOffset.Uint64WithOverflow()
	outSize64, of4 := outSize.Uint64WithOverflow()
	if of1 || of2 || ofN || of3 || of4 {
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}

	inBytes := scope.Memory.GetPtr(inOffset64, inSize64)
	if len(inBytes) < 4 {
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}
	if len(inBytes[4:]) < dfrlInputHeaderLen || uint64(readDFRLClientCount(inBytes[4:])) != n64 {
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}
	k := int(uint32(inBytes[0])<<24 | uint32(inBytes[1])<<16 | uint32(inBytes[2])<<8 | uint32(inBytes[3]))
	ups, err := decodeDFRLInput(inBytes[4:])
	if err != nil {
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}
	result, err := dfrlagg.TrimmedMean(ups, k)
	if err != nil {
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}
	enc, err := dfrlagg.Encode(result)
	if err != nil {
		// Unreachable if TrimmedMean succeeded, kept as a hard failure signal
		// for the same reason opFedAvg keeps its equivalent check.
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}

	if outSize64 < uint64(len(enc)) {
		scope.Stack.push(new(uint256.Int))
		return nil, nil
	}
	scope.Memory.Set(outOffset64, uint64(len(enc)), enc)
	scope.Stack.push(new(uint256.Int).SetOne())
	return nil, nil
}

// gasRobustAgg mirrors robustAggPrecompile.RequiredGas's model (FedAvg's cost
// function plus an N*log2(N) comparison-sort factor per coordinate, see
// contracts_dfrl.go),
// under the same n-cannot-be-read-from-memory-yet constraint gasFedAvg
// documents.
func gasRobustAgg(evm *EVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (GasCosts, error) {
	memGas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return GasCosts{}, err
	}
	inSize, of1 := stack.back(1).Uint64WithOverflow()
	n, of2 := stack.back(2).Uint64WithOverflow()
	if of1 || of2 {
		return GasCosts{}, ErrGasUintOverflow
	}
	if n == 0 || n >= uint64(dfrlagg.ClientBound) {
		return GasCosts{}, ErrGasUintOverflow
	}
	bytesPerClient := inSize / n
	if inSize%n != 0 {
		bytesPerClient++
	}
	paramsPerClient := bytesPerClient / 8
	if bytesPerClient%8 != 0 {
		paramsPerClient++
	}
	execGas := checkedDFRLGas(n, paramsPerClient, true)
	if execGas == ^uint64(0) {
		return GasCosts{}, ErrGasUintOverflow
	}
	total := memGas + execGas
	if total < memGas {
		return GasCosts{}, ErrGasUintOverflow
	}
	return GasCosts{ExecutionGas: total}, nil
}

// memoryRobustAgg is identical in shape to memoryFedAvg: the larger of the
// input and caller-declared output regions. The extra 4-byte k header inside
// inSize does not change which span dominates.
func memoryRobustAgg(stack *Stack) (uint64, bool) {
	inSz, of1 := calcMemSize64(stack.back(0), stack.back(1))
	outSz, of2 := calcMemSize64(stack.back(3), stack.back(4))
	if of1 || of2 {
		return 0, true
	}
	if outSz > inSz {
		return outSz, false
	}
	return inSz, false
}

// newDFRLInstructionSet layers FEDAVG on top of Osaka, exactly mirroring
// PrecompiledContractsDFRL's scoping choice in contracts_dfrl.go (and for the
// same reason: this is a permissioned research testbed we fully control, not
// a mainnet fork coexisting with arbitrary earlier rule sets).
//
// This registers FEDAVG as a COMPLETE new operation struct (execute,
// constantGas, dynamicGas, minStack, maxStack, memorySize all defined
// together) rather than patching any existing jump-table entry -- the
// project's own post-mortem on the submitted manuscript's opcode error
// identified exactly this discipline as the fix for R1-3/R2-9/R2-10: patching
// only `execute` on an existing slot would silently inherit that slot's old
// stack-validation and gas logic, producing gas numbers with no real
// relationship to |Theta|.
func newDFRLInstructionSet() JumpTable {
	instructionSet := newOsakaInstructionSet()
	instructionSet[FEDAVG] = &operation{
		execute:     opFedAvg,
		constantGas: 0, // all cost is dynamic (gasFedAvg); no free-standing constant component
		dynamicGas:  gasFedAvg,
		minStack:    minStack(5, 1),
		maxStack:    maxStack(5, 1),
		memorySize:  memoryFedAvg,
	}
	instructionSet[ROBUSTAGG] = &operation{
		execute:     opRobustAgg,
		constantGas: 0,
		dynamicGas:  gasRobustAgg,
		minStack:    minStack(5, 1),
		maxStack:    maxStack(5, 1),
		memorySize:  memoryRobustAgg,
	}
	return validate(instructionSet)
}

var dfrlInstructionSet = newDFRLInstructionSet()

func readDFRLClientCount(input []byte) uint32 {
	return uint32(input[0])<<24 | uint32(input[1])<<16 | uint32(input[2])<<8 | uint32(input[3])
}
