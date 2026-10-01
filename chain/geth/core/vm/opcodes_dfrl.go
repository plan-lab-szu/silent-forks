// Copyright 2026 The DT-DFRL authors.
// This file is not part of upstream go-ethereum. It defines one new opcode,
// gated behind the DFRL fork (see params.ChainConfig.IsDFRL and
// core/vm/instructions_dfrl.go), and registers its name in the same lookup
// tables every other opcode uses so String()/disassembly work for it.

package vm

// FEDAVG is the fused deterministic-FedAvg opcode.
//
// Address 0xC2 (the "+2" offset from the submitted manuscript's mistaken
// 0xF2 is intentional -- see HANDBOOK.md "R1-3 专项跟进"): verified FREE as
// of the go-ethereum commit this fork is pinned to, by grepping
// core/vm/opcodes.go directly rather than trusting memory or an older
// document (see HANDBOOK.md "opcode 事实", dated 2026-08-24). 0xC0-0xC9 are
// entirely unallocated on that commit; 0xC2 is chosen within that range to
// keep the "+2" mnemonic the manuscript already used.
const FEDAVG OpCode = 0xc2

// ROBUSTAGG is the fused deterministic-TrimmedMean opcode (U3, Phase 4).
// 0xC3 -- next free slot after FEDAVG within the same confirmed-free 0xC0-0xC9
// range (see the opcode-table grep this fork's HANDBOOK.md records).
const ROBUSTAGG OpCode = 0xc3

func init() {
	// Deliberately NOT editing the opCodeToString/stringToOp literal blocks in
	// opcodes.go -- this red line exists precisely so a diff against upstream
	// stays additive and easy to review: nothing already there is touched.
	opCodeToString[FEDAVG] = "FEDAVG"
	stringToOp["FEDAVG"] = FEDAVG
	opCodeToString[ROBUSTAGG] = "ROBUSTAGG"
	stringToOp["ROBUSTAGG"] = ROBUSTAGG
}
