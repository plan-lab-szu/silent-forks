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

// buildRobustAggOpcodeCaller mirrors buildFedAvgOpcodeCaller exactly, except
// it emits ROBUSTAGG instead of FEDAVG. The k trim-count header is expected
// to already be the first 4 bytes of calldata (robustAggInput below builds
// it that way), matching the precompile's own input layout.
func buildRobustAggOpcodeCaller(n uint64, outCap uint64) []byte {
	var b []byte
	push := func(v uint64) {
		if v == 0 {
			b = append(b, byte(vm.PUSH1), 0)
			return
		}
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
	const outOffset = 0x2000

	op(vm.CALLDATASIZE)
	push(0)
	push(0)
	op(vm.CALLDATACOPY)

	push(outCap)
	push(outOffset)
	push(n)
	op(vm.CALLDATASIZE)
	push(0)
	op(vm.ROBUSTAGG)

	push(0)
	op(vm.MSTORE)

	push(outCap)
	push(outOffset)
	push(32)
	op(vm.MCOPY)

	push(32 + outCap)
	push(0)
	op(vm.RETURN)
	return b
}

// robustAggInput builds the RobustAgg ABI: 4-byte big-endian k, followed by
// the same length-prefixed input encodeDFRLInput already produces for FedAvg.
func robustAggInput(t *testing.T, k int, weights []int64, tensors []dfrlagg.Tensor) []byte {
	t.Helper()
	body := encodeDFRLInput(t, weights, tensors)
	out := make([]byte, 4+len(body))
	binary.BigEndian.PutUint32(out, uint32(k))
	copy(out[4:], body)
	return out
}

func TestRobustAggOpcodeMatchesPrecompile(t *testing.T) {
	const n, p, k = 7, 512, 1
	weights := make([]int64, n)
	for i := range weights {
		weights[i] = int64(i + 1)
	}
	_, tensors := genUpdates(t, 555, n, p, weights, 1<<14)
	input := robustAggInput(t, k, weights, tensors)

	want, err := dfrlagg.TrimmedMean(mustUpdates(t, weights, tensors), k)
	if err != nil {
		t.Fatal(err)
	}
	wantEnc, _ := dfrlagg.Encode(want)

	code := buildRobustAggOpcodeCaller(uint64(n), uint64(len(wantEnc))+64)
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
		t.Fatalf("opcode output differs from direct dfrlagg.TrimmedMean call\n got=%x\nwant=%x", payload, wantEnc)
	}

	pc := vm.PrecompiledContractsDFRL[vm.DFRLRobustAggAddress]
	pcOut, err := pc.Run(input)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(pcOut, payload) {
		t.Fatalf("opcode and precompile disagree on identical input")
	}
}

func TestRobustAggOpcodeRejectsMalformedInput(t *testing.T) {
	code := buildRobustAggOpcodeCaller(1, 64)
	// k=0 header followed by an n=0 DFRL body: rejected by decodeDFRLInput.
	out, _, err := runtime.Execute(code, []byte{0, 0, 0, 0, 0, 0, 0, 0}, &runtime.Config{
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

func TestRobustAggOpcodeBindsShapeAndRequiresCompleteOutput(t *testing.T) {
	weights := []int64{1, 1, 1}
	_, tensors := genUpdates(t, 88, 3, 8, weights, 1<<10)
	input := robustAggInput(t, 1, weights, tensors)
	want, err := dfrlagg.TrimmedMean(mustUpdates(t, weights, tensors), 1)
	if err != nil {
		t.Fatal(err)
	}
	wantEnc, err := dfrlagg.Encode(want)
	if err != nil {
		t.Fatal(err)
	}

	mismatch := buildRobustAggOpcodeCaller(1, uint64(len(wantEnc)))
	out, _, err := runtime.Execute(mismatch, input, &runtime.Config{
		ChainConfig: dfrlConfig(), GasLimit: 10_000_000,
	})
	if err != nil {
		t.Fatalf("shape-mismatch execution: %v", err)
	}
	if len(out) < 32 || out[31] != 0 {
		t.Fatalf("shape mismatch returned success: %x", out)
	}

	shortCap := uint64(len(wantEnc) - 1)
	short := buildRobustAggOpcodeCaller(uint64(len(weights)), shortCap)
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

func TestRobustAggOpcodeNotActiveWithoutFork(t *testing.T) {
	cfg := *params.AllDevChainProtocolChanges
	code := buildRobustAggOpcodeCaller(1, 64)
	_, _, err := runtime.Execute(code, []byte{0, 0, 0, 0, 0, 0, 0, 0}, &runtime.Config{
		ChainConfig: &cfg,
		GasLimit:    10_000_000,
	})
	if err == nil {
		t.Fatal("expected execution to fail (0xC3 undefined) when DFRL is not active, got success")
	}
}

func TestRobustAggOpcodeGasMonotonic(t *testing.T) {
	run := func(n int, p int) uint64 {
		w := make([]int64, n)
		for i := range w {
			w[i] = int64(i + 1)
		}
		_, tensors := genUpdates(t, 9, n, p, w, 1<<10)
		input := robustAggInput(t, 1, w, tensors)
		code := buildRobustAggOpcodeCaller(uint64(n), 1<<20)
		cfg := &runtime.Config{ChainConfig: dfrlConfig(), GasLimit: 200_000_000}
		_, _, err := runtime.Execute(code, input, cfg)
		if err != nil {
			t.Fatalf("n=%d p=%d: %v", n, p, err)
		}
		return vm.PrecompiledContractsDFRL[vm.DFRLRobustAggAddress].RequiredGas(input)
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
