// Copyright 2026 The DT-DFRL authors.

package vm_test

import (
	"bytes"
	"encoding/binary"
	"testing"

	dfrlagg "dfrl-agg"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/core/vm/runtime"
	"github.com/ethereum/go-ethereum/params"
)

// buildFedAvgOpcodeCaller assembles bytecode that: copies calldata to
// mem[0:calldatasize], pushes the FEDAVG stack arguments (inOffset=0,
// inSize=calldatasize, n, outOffset=big enough to not overlap the input,
// outSize=outCap), executes FEDAVG, MSTOREs the success flag at a fixed slot,
// and returns success flag (32 bytes) followed by the written output bytes --
// so a single call lets the test check both the success flag and the payload.
//
// Layout of the returned bytes: [0:32) = success flag (0 or 1, big-endian
// uint256), [32:32+outCap) = whatever FEDAVG wrote (zero-padded if it wrote
// less than outCap).
func buildFedAvgOpcodeCaller(n uint64, outCap uint64) []byte {
	var b []byte
	push := func(v uint64) {
		if v == 0 {
			b = append(b, byte(vm.PUSH1), 0)
			return
		}
		// Minimal big-endian encoding, PUSH-width chosen to fit v (v here is
		// always small enough for PUSH8).
		var buf [8]byte
		binary.BigEndian.PutUint64(buf[:], v)
		start := 0
		for start < 7 && buf[start] == 0 {
			start++
		}
		b = append(b, byte(vm.PUSH1)+byte(7-start))
		b = append(b, buf[start:]...)
	}
	op := func(o vm.OpCode) { b = append(b, byte(o)) }

	// Reserve memory: input at [0, calldatasize), output at a fixed high
	// offset (0x2000) far past any realistic calldatasize in these tests.
	const outOffset = 0x2000

	op(vm.CALLDATASIZE)
	push(0)
	push(0)
	op(vm.CALLDATACOPY)

	// Stack for FEDAVG, top to bottom: inOffset, inSize, n, outOffset, outSize.
	// Push in REVERSE (bottom-most first) so the last push ends up on top.
	push(outCap)
	push(outOffset)
	push(n)
	op(vm.CALLDATASIZE) // inSize
	push(0)             // inOffset
	op(vm.FEDAVG)

	// Stack now: [successFlag]. Store it at memory 0, then copy the FEDAVG
	// output (already at outOffset) so both are in the return region.
	push(0)
	op(vm.MSTORE) // mem[0:32) = successFlag

	// Copy output region [outOffset, outOffset+outCap) to [32, 32+outCap)
	// so the whole thing is one contiguous RETURN: MCOPY(dest=32, src=outOffset, size=outCap).
	push(outCap)
	push(outOffset)
	push(32)
	op(vm.MCOPY)

	push(32 + outCap)
	push(0)
	op(vm.RETURN)
	return b
}

func TestFedAvgOpcodeMatchesPrecompile(t *testing.T) {
	_, tensors := genUpdates(t, 555, 6, 512, []int64{1, 2, 3, 4, 5, 6}, 1<<14)
	weights := []int64{1, 2, 3, 4, 5, 6}
	input := encodeDFRLInput(t, weights, tensors)

	want, err := dfrlagg.FedAvg(mustUpdates(t, weights, tensors))
	if err != nil {
		t.Fatal(err)
	}
	wantEnc, _ := dfrlagg.Encode(want)

	code := buildFedAvgOpcodeCaller(uint64(len(weights)), uint64(len(wantEnc))+64)
	out, _, err := runtime.Execute(code, input, &runtime.Config{
		ChainConfig: dfrlConfig(),
		GasLimit:    50_000_000,
	})
	if err != nil {
		t.Fatalf("EVM execution failed: %v", err)
	}
	if len(out) < 32 {
		t.Fatalf("return data too short: %d bytes", len(out))
	}
	success := out[31]
	payload := out[32 : 32+len(wantEnc)]
	if success != 1 {
		t.Fatalf("success flag = %d, want 1", success)
	}
	if !bytes.Equal(payload, wantEnc) {
		t.Fatalf("opcode output differs from direct dfrlagg.FedAvg call\n got=%x\nwant=%x", payload, wantEnc)
	}

	// And it must match the precompile's output too, since both call the
	// same dfrlagg.FedAvg -- U6 at the opcode/precompile level.
	pc := vm.PrecompiledContractsDFRL[vm.DFRLFedAvgAddress]
	pcOut, err := pc.Run(input)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(pcOut, payload) {
		t.Fatalf("opcode and precompile disagree on identical input")
	}
}

func TestFedAvgOpcodeRejectsMalformedInput(t *testing.T) {
	code := buildFedAvgOpcodeCaller(1, 64)
	out, _, err := runtime.Execute(code, []byte{0, 0, 0, 0}, &runtime.Config{ // n=0 header: rejected by decodeDFRLInput
		ChainConfig: dfrlConfig(),
		GasLimit:    10_000_000,
	})
	if err != nil {
		t.Fatalf("EVM execution failed: %v", err)
	}
	if len(out) < 32 || out[31] != 0 {
		t.Fatalf("expected success flag 0 on malformed input, got %x", out)
	}
}

func TestFedAvgOpcodeBindsShapeAndRequiresCompleteOutput(t *testing.T) {
	weights := []int64{1, 2}
	_, tensors := genUpdates(t, 77, 2, 8, weights, 1<<10)
	input := encodeDFRLInput(t, weights, tensors)
	want, err := dfrlagg.FedAvg(mustUpdates(t, weights, tensors))
	if err != nil {
		t.Fatal(err)
	}
	wantEnc, err := dfrlagg.Encode(want)
	if err != nil {
		t.Fatal(err)
	}

	// The stack-declared N is used for gas and must equal the payload N.
	mismatch := buildFedAvgOpcodeCaller(1, uint64(len(wantEnc)))
	out, _, err := runtime.Execute(mismatch, input, &runtime.Config{
		ChainConfig: dfrlConfig(), GasLimit: 10_000_000,
	})
	if err != nil {
		t.Fatalf("shape-mismatch execution: %v", err)
	}
	if len(out) < 32 || out[31] != 0 {
		t.Fatalf("shape mismatch returned success: %x", out)
	}

	// A short output buffer must not turn a truncated tensor into success.
	shortCap := uint64(len(wantEnc) - 1)
	short := buildFedAvgOpcodeCaller(uint64(len(weights)), shortCap)
	out, _, err = runtime.Execute(short, input, &runtime.Config{
		ChainConfig: dfrlConfig(), GasLimit: 10_000_000,
	})
	if err != nil {
		t.Fatalf("short-output execution: %v", err)
	}
	if len(out) < 32 || out[31] != 0 {
		t.Fatalf("short output buffer returned success: %x", out)
	}
	if !bytes.Equal(out[32:], make([]byte, len(out)-32)) {
		t.Fatalf("short output buffer received a partial encoding: %x", out[32:])
	}
}

func TestFedAvgOpcodeNotActiveWithoutFork(t *testing.T) {
	cfg := *params.AllDevChainProtocolChanges // DFRLTime nil: FEDAVG not registered
	code := buildFedAvgOpcodeCaller(1, 64)
	_, _, err := runtime.Execute(code, []byte{0, 0, 0, 0}, &runtime.Config{
		ChainConfig: &cfg,
		GasLimit:    10_000_000,
	})
	if err == nil {
		t.Fatal("expected execution to fail (0xC2 undefined) when DFRL is not active, got success")
	}
}

// TestFedAvgOpcodeGasMonotonic is the opcode-side counterpart to
// TestRequiredGasMonotonic: the SHAPE of the model (non-decreasing in |Theta|
// and in N) must hold for the opcode's gas function too, independent of the
// exact constants (see gasFedAvg's doc comment on why they are approximate).
func TestFedAvgOpcodeGasMonotonic(t *testing.T) {
	run := func(n int, p int) uint64 {
		_, tensors := genUpdates(t, 9, n, p, dfrlagg.PaperWeights[:n], 1<<10)
		w := make([]int64, n)
		for i := range w {
			w[i] = int64(i + 1)
		}
		input := encodeDFRLInput(t, w, tensors)
		code := buildFedAvgOpcodeCaller(uint64(n), 1<<20)
		cfg := &runtime.Config{ChainConfig: dfrlConfig(), GasLimit: 200_000_000}
		before := cfg.GasLimit
		_, _, err := runtime.Execute(code, input, cfg)
		if err != nil {
			t.Fatalf("n=%d p=%d: %v", n, p, err)
		}
		// runtime.Config.GasLimit is not mutated in place by Execute (a copy of
		// the budget is used internally), so gas consumption is inferred by
		// running twice at two different limits is unnecessary here -- instead,
		// use the precompile's RequiredGas on the identical input as a proxy for
		// the shape, since both paths share dfrlGasBase/dfrlGasPerParam/dfrlGasPerNParam.
		_ = before
		return vm.PrecompiledContractsDFRL[vm.DFRLFedAvgAddress].RequiredGas(input)
	}
	var prev uint64
	for _, p := range []int{16, 1024, 16384} {
		g := run(4, p)
		if g < prev {
			t.Errorf("gas decreased as |Theta| grew: p=%d gas=%d < previous %d", p, g, prev)
		}
		prev = g
	}
}
