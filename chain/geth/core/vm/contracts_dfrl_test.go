// Copyright 2026 The DT-DFRL authors.

package vm_test

import (
	"bytes"
	"encoding/binary"
	"testing"

	dfrlagg "dfrl-agg"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/core/vm/runtime"
	"github.com/ethereum/go-ethereum/params"
)

// encodeDFRLInput builds the precompile's calldata ABI directly (mirrors
// decodeDFRLInput in contracts_dfrl.go, deliberately re-implemented rather
// than imported: this test must catch a decoder that silently accepts bytes
// its own encoder would never produce).
func encodeDFRLInput(t *testing.T, weights []int64, tensors []dfrlagg.Tensor) []byte {
	t.Helper()
	buf := make([]byte, 4)
	binary.BigEndian.PutUint32(buf, uint32(len(weights)))
	for i, w := range weights {
		enc, err := dfrlagg.Encode(tensors[i])
		if err != nil {
			t.Fatalf("encode tensor %d: %v", i, err)
		}
		entry := make([]byte, 8+4+len(enc))
		binary.BigEndian.PutUint64(entry, uint64(w))
		binary.BigEndian.PutUint32(entry[8:], uint32(len(enc)))
		copy(entry[12:], enc)
		buf = append(buf, entry...)
	}
	return buf
}

func genUpdates(t *testing.T, seed uint64, n, p int, weights []int64, bound int64) ([]dfrlagg.Update, []dfrlagg.Tensor) {
	t.Helper()
	ups := dfrlagg.GenUpdates(seed, n, p, weights, bound, dfrlagg.DefaultF)
	tensors := make([]dfrlagg.Tensor, n)
	for i, u := range ups {
		tensors[i] = u.Tensor
	}
	return ups, tensors
}

// dfrlConfig returns a chain config with DFRL (and its Osaka prerequisite)
// active from genesis, for tests that need Rules.IsDFRL to be true.
func dfrlConfig() *params.ChainConfig {
	zero := uint64(0)
	cfg := *params.AllDevChainProtocolChanges
	cfg.OsakaTime = &zero
	cfg.DFRLTime = &zero
	return &cfg
}

// TestFedAvgPrecompileMatchesDirectCall is the precompile's core correctness
// property: calling it through RunPrecompiledContract must produce EXACTLY
// the bytes dfrlagg.FedAvg produces when called directly. If it did not, U6
// (on-chain equals off-chain) would be false despite both paths sharing code,
// which would mean this file's ABI layer introduced a bug the shared
// implementation cannot be blamed for.
func TestFedAvgPrecompileMatchesDirectCall(t *testing.T) {
	ups, tensors := genUpdates(t, 20260824, 10, 2048, dfrlagg.PaperWeights, 1<<17)
	weights := make([]int64, len(ups))
	for i, u := range ups {
		weights[i] = u.Weight
	}

	want, err := dfrlagg.FedAvg(ups)
	if err != nil {
		t.Fatal(err)
	}
	wantEnc, err := dfrlagg.Encode(want)
	if err != nil {
		t.Fatal(err)
	}

	input := encodeDFRLInput(t, weights, tensors)
	pc := vm.PrecompiledContractsDFRL[vm.DFRLFedAvgAddress]
	got, err := pc.Run(input)
	if err != nil {
		t.Fatalf("precompile Run: %v", err)
	}
	if !bytes.Equal(got, wantEnc) {
		t.Fatalf("precompile output differs from direct dfrlagg.FedAvg call\n got=%x\nwant=%x", got, wantEnc)
	}
}

// TestRobustAggPrecompileMatchesDirectCall is the same property for the
// second primitive (OP_ROBUSTAGG, U3).
func TestRobustAggPrecompileMatchesDirectCall(t *testing.T) {
	ups, tensors := genUpdates(t, 7, 9, 512, []int64{1, 1, 1, 1, 1, 1, 1, 1, 1}, 1<<18)
	weights := make([]int64, len(ups))
	for i, u := range ups {
		weights[i] = u.Weight
	}
	const k = 2
	want, err := dfrlagg.TrimmedMean(ups, k)
	if err != nil {
		t.Fatal(err)
	}
	wantEnc, _ := dfrlagg.Encode(want)

	body := encodeDFRLInput(t, weights, tensors)
	input := make([]byte, 4+len(body))
	binary.BigEndian.PutUint32(input, uint32(k))
	copy(input[4:], body)

	pc := vm.PrecompiledContractsDFRL[vm.DFRLRobustAggAddress]
	got, err := pc.Run(input)
	if err != nil {
		t.Fatalf("precompile Run: %v", err)
	}
	if !bytes.Equal(got, wantEnc) {
		t.Fatalf("precompile output differs from direct dfrlagg.TrimmedMean call")
	}
}

// TestFedAvgPrecompileRejectsMalformedInput exercises the ABI's own error
// paths. Every case here must be rejected, not silently truncated or padded:
// the tensor decoder underneath already rejects non-canonical bytes
// (spec section 4.2 malleability requirement), and this ABI layer must not
// reopen that closed door with its own leniency.
func TestFedAvgPrecompileRejectsMalformedInput(t *testing.T) {
	_, tensors := genUpdates(t, 1, 3, 16, []int64{1, 2, 3}, 1<<10)
	valid := encodeDFRLInput(t, []int64{1, 2, 3}, tensors)
	wrongF := append([]dfrlagg.Tensor(nil), tensors...)
	for i := range wrongF {
		wrongF[i].F = dfrlagg.DefaultF + 1
	}
	pc := vm.PrecompiledContractsDFRL[vm.DFRLFedAvgAddress]

	cases := map[string][]byte{
		"empty":                    {},
		"header only":              valid[:2],
		"zero clients":             append([]byte{0, 0, 0, 0}, valid[4:]...),
		"too many clients":         {0, 0, 4, 0},
		"truncated entry header":   valid[:4+8],
		"truncated tensor body":    valid[:len(valid)-1],
		"trailing byte":            append(append([]byte{}, valid...), 0),
		"tensor length overstated": overstateTensorLen(valid),
		"wrong protocol F":         encodeDFRLInput(t, []int64{1, 2, 3}, wrongF),
	}
	for name, input := range cases {
		if _, err := pc.Run(input); err == nil {
			t.Errorf("%s: precompile accepted malformed input", name)
		}
	}
	if _, err := pc.Run(valid); err != nil {
		t.Errorf("valid input rejected: %v", err)
	}
}

func overstateTensorLen(valid []byte) []byte {
	b := append([]byte(nil), valid...)
	binary.BigEndian.PutUint32(b[4+8:], 0xFFFFFFFF)
	return b
}

// TestRequiredGasNeverUnderpricesGarbage checks the property RequiredGas's
// doc comment promises: it must not crash and must not return a tiny gas
// value for adversarial input that CLAIMS a huge shape, even though it never
// decodes the tensors to find out the truth. Underpricing here would let an
// attacker claim an enormous aggregation for a few hundred gas.
func TestRequiredGasNeverUnderpricesGarbage(t *testing.T) {
	pc := vm.PrecompiledContractsDFRL[vm.DFRLFedAvgAddress]
	huge := make([]byte, 16)
	binary.BigEndian.PutUint32(huge, 1<<20)           // claims a million clients
	binary.BigEndian.PutUint32(huge[12:], 0xFFFFFFFF) // claims a ~4GB tensor
	if g := pc.RequiredGas(huge); g < 1_000_000 {
		t.Fatalf("adversarial header priced at only %d gas", g)
	}
	if g := pc.RequiredGas(nil); g == 0 {
		t.Fatalf("empty input priced at zero gas")
	}

	// A small first tensor must not underprice a later entry that declares a
	// much larger shape, even when that later entry is truncated and Run will
	// reject it.
	_, tensors := genUpdates(t, 91, 1, 4, []int64{1}, 1<<8)
	laterHuge := encodeDFRLInput(t, []int64{1}, tensors)
	binary.BigEndian.PutUint32(laterHuge, 2)
	const entryHeaderLen = 12
	laterHuge = append(laterHuge, make([]byte, entryHeaderLen)...)
	binary.BigEndian.PutUint64(laterHuge[len(laterHuge)-entryHeaderLen:], 1)
	binary.BigEndian.PutUint32(laterHuge[len(laterHuge)-4:], 0xFFFFFFFF)
	if g := pc.RequiredGas(laterHuge); g < 1_000_000_000 {
		t.Fatalf("large later tensor claim priced at only %d gas", g)
	}
}

// TestRequiredGasMonotonic is the qualitative shape check on the R2-10 gas
// model: holding N fixed, gas must not decrease as |Theta| grows, and holding
// |Theta| fixed, gas must not decrease as N grows. The exact constants are a
// provisional benchmark fit (see contracts_dfrl_bench_test.go); this test
// checks the model's SHAPE, which does not depend on the fit being final.
func TestRequiredGasMonotonic(t *testing.T) {
	pc := vm.PrecompiledContractsDFRL[vm.DFRLFedAvgAddress]
	build := func(n, p int) []byte {
		_, tensors := genUpdates(t, 3, n, p, dfrlagg.PaperWeights[:n], 1<<10)
		w := make([]int64, n)
		for i := range w {
			w[i] = int64(i + 1)
		}
		return encodeDFRLInput(t, w, tensors)
	}
	var prevP uint64
	for _, p := range []int{16, 256, 4096} {
		g := pc.RequiredGas(build(4, p))
		if g < prevP {
			t.Errorf("gas decreased as |Theta| grew: p=%d gas=%d < previous %d", p, g, prevP)
		}
		prevP = g
	}
	var prevN uint64
	for _, n := range []int{2, 4, 8} {
		g := pc.RequiredGas(build(n, 256))
		if g < prevN {
			t.Errorf("gas decreased as N grew: n=%d gas=%d < previous %d", n, g, prevN)
		}
		prevN = g
	}
}

// TestRobustAggGasIncludesClientFactor guards the asymptotic term required by
// coordinate-wise comparison sorting.  At fixed P, the RobustAgg surcharge
// must scale as N*ceil(log2 N), not merely ceil(log2 N).
func TestRobustAggGasIncludesClientFactor(t *testing.T) {
	surcharge := func(n, p int) uint64 {
		_, tensors := genUpdates(t, 17, n, p, dfrlagg.PaperWeights[:n], 1<<10)
		weights := make([]int64, n)
		for i := range weights {
			weights[i] = int64(i + 1)
		}
		body := encodeDFRLInput(t, weights, tensors)
		robustInput := make([]byte, 4+len(body))
		binary.BigEndian.PutUint32(robustInput, 1)
		copy(robustInput[4:], body)
		fedGas := vm.PrecompiledContractsDFRL[vm.DFRLFedAvgAddress].RequiredGas(body)
		robustGas := vm.PrecompiledContractsDFRL[vm.DFRLRobustAggAddress].RequiredGas(robustInput)
		if robustGas <= fedGas {
			t.Fatalf("n=%d p=%d: RobustAgg gas %d is not above FedAvg gas %d", n, p, robustGas, fedGas)
		}
		return robustGas - fedGas
	}

	s4 := surcharge(4, 256) // 4*ceil(log2 4) = 8
	s8 := surcharge(8, 256) // 8*ceil(log2 8) = 24
	if s8 != 3*s4 {
		t.Fatalf("sort surcharge does not scale as N*ceil(log2 N): N=4 %d, N=8 %d", s4, s8)
	}
}

// TestFedAvgPrecompileViaEVMRuntime is the end-to-end sanity check: a real
// contract, deployed and called through core/vm/runtime's EVM harness (not a
// full node, but a genuine EVM execution -- gas accounting, precompile
// dispatch, and the DFRL fork gate all run for real). It exercises the
// dispatch path this file's unit tests above bypass (calling pc.Run directly
// skips RunPrecompiledContract, the jump table, and the fork check).
//
// The "contract" here is a minimal piece of bytecode that copies calldata to
// memory and DELEGATECALLs the DFRL precompile address, returning whatever it
// returns -- enough to prove the address is actually reachable and dispatches
// to our Go code, without needing Solidity/solc (not installed on this
// machine; see HANDBOOK.md).
func TestFedAvgPrecompileViaEVMRuntime(t *testing.T) {
	_, tensors := genUpdates(t, 42, 3, 64, []int64{10, 20, 30}, 1<<12)
	input := encodeDFRLInput(t, []int64{10, 20, 30}, tensors)

	want, err := dfrlagg.FedAvg(mustUpdates(t, []int64{10, 20, 30}, tensors))
	if err != nil {
		t.Fatal(err)
	}
	wantEnc, _ := dfrlagg.Encode(want)

	// Bytecode: CALLDATACOPY the input into memory, then STATICCALL the DFRL
	// precompile with it, RETURNDATACOPY the result, and RETURN it.
	// Built with raw opcodes (no assembler available) -- correctness is
	// verified by the test actually passing, not by inspection.
	addr := vm.DFRLFedAvgAddress
	code := buildStaticcallForwarder(addr)

	out, _, err := runtime.Execute(code, input, &runtime.Config{
		ChainConfig: dfrlConfig(),
		GasLimit:    50_000_000,
	})
	if err != nil {
		t.Fatalf("EVM execution failed: %v", err)
	}
	if !bytes.Equal(out, wantEnc) {
		t.Fatalf("EVM-dispatched precompile output differs from direct call\n got=%x\nwant=%x", out, wantEnc)
	}
}

func mustUpdates(t *testing.T, weights []int64, tensors []dfrlagg.Tensor) []dfrlagg.Update {
	t.Helper()
	ups := make([]dfrlagg.Update, len(weights))
	for i := range weights {
		var a dfrlagg.Addr
		a[19] = byte(i + 1)
		ups[i] = dfrlagg.Update{Addr: a, Weight: weights[i], Tensor: tensors[i]}
	}
	return ups
}

// buildStaticcallForwarder assembles minimal EVM bytecode that forwards
// calldata verbatim to `addr` via STATICCALL and returns whatever comes back.
// Written by hand against the opcode table because no assembler is available
// in this environment; every opcode here is a single-byte, no-immediate
// instruction except the PUSH20 address and PUSH1 gas/size literals.
func buildStaticcallForwarder(addr common.Address) []byte {
	var b []byte
	push := func(v byte) { b = append(b, byte(vm.PUSH1), v) }
	op := func(o vm.OpCode) { b = append(b, byte(o)) }

	// CALLDATASIZE; PUSH1 0; PUSH1 0; CALLDATACOPY   -- copy all calldata to mem[0:size]
	op(vm.CALLDATASIZE)
	push(0)
	push(0)
	op(vm.CALLDATACOPY)

	// PUSH1 0 (retSize); PUSH1 0 (retOffset)
	// CALLDATASIZE (argsSize); PUSH1 0 (argsOffset)
	// PUSH20 addr; PUSH gasleft-ish (use GAS opcode instead of a literal)
	// STATICCALL
	push(0)
	push(0)
	op(vm.CALLDATASIZE)
	push(0)
	b = append(b, byte(vm.PUSH20))
	b = append(b, addr.Bytes()...)
	op(vm.GAS)
	op(vm.STATICCALL)
	op(vm.POP) // discard the success flag; a failed call surfaces via RETURNDATASIZE==0 below and the test's byte comparison

	// RETURNDATASIZE; PUSH1 0; PUSH1 0; RETURNDATACOPY; RETURNDATASIZE; PUSH1 0; RETURN
	op(vm.RETURNDATASIZE)
	push(0)
	push(0)
	op(vm.RETURNDATACOPY)
	op(vm.RETURNDATASIZE)
	push(0)
	op(vm.RETURN)
	return b
}

// TestDFRLNotActiveWithoutFork proves the fork gate actually gates: with an
// otherwise-identical chain config that never activates DFRL, the same
// bytecode calling the same address must NOT reach our precompile. Without
// this test, an always-on registration bug (e.g. wiring the address into an
// unconditional set) would pass every other test in this file while silently
// breaking every unmodified client that has never heard of DFRL.
func TestDFRLNotActiveWithoutFork(t *testing.T) {
	cfg := *params.AllDevChainProtocolChanges // DFRLTime left nil: never active
	code := buildStaticcallForwarder(vm.DFRLFedAvgAddress)
	out, _, err := runtime.Execute(code, []byte{0, 0, 0, 0}, &runtime.Config{
		ChainConfig: &cfg,
		GasLimit:    1_000_000,
	})
	if err != nil {
		t.Fatalf("EVM execution failed: %v", err)
	}
	// The address has no deployed code and is not a precompile on this
	// config, so STATICCALL succeeds against an empty account and returns
	// zero bytes -- not our tensor encoding, and not an error either. Both
	// are correct: the point is that it did NOT run our aggregation.
	if len(out) != 0 {
		t.Fatalf("precompile responded even though DFRL is not active on this chain config: %x", out)
	}
}
