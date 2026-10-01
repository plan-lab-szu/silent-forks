//go:build e3lite_vulnerable

// Copyright 2026 The DT-DFRL authors.

package catalyst

import (
	"context"
	"encoding/json"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/beacon/engine"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/core/vm"
)

// Shared machinery for the three payload-exchange experiments: the record that
// travels between the two nodes, and the judging step that every experiment
// performs identically. Kept in one place so that a guard added for one
// experiment protects all three, and so the cross-machine mode below has a
// single definition rather than three.

// dfrlExchangeBeaconRoot is fixed so both nodes build over identical payload
// attributes; a differing beacon root would change the state root for a reason
// unrelated to the aggregation.
var dfrlExchangeBeaconRoot = common.Hash{0x42}

// dfrlHost records who produced a record. Only meaningful for the
// cross-machine runs, where the whole point is that the two halves came from
// different hosts, but recorded always so that a same-machine log and a
// cross-machine log are the same shape.
type dfrlHost struct {
	Hostname  string `json:"hostname"`
	GOOS      string `json:"goos"`
	GOARCH    string `json:"goarch"`
	GoVersion string `json:"goVersion"`
	NumCPU    int    `json:"numCPU"`
}

func dfrlHostInfo() dfrlHost {
	name, err := os.Hostname()
	if err != nil {
		name = "unknown"
	}
	return dfrlHost{
		Hostname: name, GOOS: runtime.GOOS, GOARCH: runtime.GOARCH,
		GoVersion: runtime.Version(), NumCPU: runtime.NumCPU(),
	}
}

// dfrlPeerEnvelope is the transport record. It carries the envelope unmodified;
// everything else is provenance, for the log and for the guards.
type dfrlPeerEnvelope struct {
	Order     string                           `json:"order"`
	Host      dfrlHost                         `json:"host"`
	StateRoot common.Hash                      `json:"stateRoot"`
	BlockHash common.Hash                      `json:"blockHash"`
	Envelope  *engine.ExecutionPayloadEnvelope `json:"envelope"`
}

// DFRL_XM_EXPECT selects which cross-machine cell this run is.
//
//	""/"differ"  the peer compiled the OTHER accumulation order. This is the
//	             same-machine design: the compiled order is the only difference
//	             between the two nodes, so the outcome is attributable to it.
//
//	"same-order" the peer compiled the SAME order, on a different machine, and
//	             agreement is asserted. This is the attribution cell, and it is
//	             asymmetric: it is conclusive when it PASSES and inconclusive
//	             when it fails.
//
//	             A pass says the commitment did not move although the host did,
//	             so the differ cell's divergence is attributable to the order.
//	             The more the two hosts differ -- architecture, toolchain, core
//	             count -- the stronger that is, because more was allowed to vary
//	             and the commitment still held.
//
//	             A failure is ambiguous whenever the hosts differ in more than
//	             the order, because architecture alone has been measured to move
//	             the unspecified path (evidence/e3lite_arm64_vs_amd64.log) and is
//	             then a known alternative cause. The failure path below says so
//	             rather than letting a disagreement read as "the order does not
//	             explain the divergence".
//
//	"same-order-observe"
//	             the peer compiled the SAME order on a host that differs in more
//	             than the order -- a different architecture, typically. Every
//	             guard still runs, but the roots and the verdict are RECORDED
//	             instead of asserted, because here they are the measurement
//	             rather than a pass/fail criterion. Use this to characterise a
//	             mismatched pair; do not cite it as attribution to the order.
const (
	dfrlXMDiffer    = "differ"
	dfrlXMSameOrder = "same-order"
	dfrlXMObserve   = "same-order-observe"
)

func dfrlXMMode() string {
	switch v := os.Getenv("DFRL_XM_EXPECT"); v {
	case "", dfrlXMDiffer:
		return dfrlXMDiffer
	case dfrlXMSameOrder, dfrlXMObserve:
		return v
	default:
		panic("DFRL_XM_EXPECT must be empty, \"differ\", \"same-order\" or \"same-order-observe\"")
	}
}

// submitPayload delivers an envelope through the Engine API version this chain
// config needs, so the verdict comes from the same path a real consensus client
// would drive. V4 is the Prague/Osaka version; V5 additionally requires
// Amsterdam fields these payloads do not carry.
func submitPayload(t *testing.T, api *ConsensusAPI, env *engine.ExecutionPayloadEnvelope) engine.PayloadStatusV1 {
	t.Helper()
	requests := make([]hexutil.Bytes, len(env.Requests))
	for i, r := range env.Requests {
		requests[i] = r
	}
	status, err := api.NewPayloadV4(context.Background(), *env.ExecutionPayload,
		[]common.Hash{}, &dfrlExchangeBeaconRoot, requests)
	if err != nil {
		t.Fatalf("NewPayloadV4: %v", err)
	}
	return status
}

// writeEnvelope publishes this node's envelope for the peer, verbatim.
func writeEnvelope(t *testing.T, path string, own *engine.ExecutionPayloadEnvelope) {
	t.Helper()
	rec := dfrlPeerEnvelope{
		Order:     vm.DFRLVulnOrderName(),
		Host:      dfrlHostInfo(),
		StateRoot: own.ExecutionPayload.StateRoot,
		BlockHash: own.ExecutionPayload.BlockHash,
		Envelope:  own,
	}
	blob, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote own envelope to %s", path)
}

// readEnvelope loads the peer's record. DFRL_WAIT_SECS makes it poll for the
// file first, which is what the cross-machine driver needs: the two halves run
// on two hosts and the record arrives through a synced directory, so the file
// may not exist yet when this process starts.
func readEnvelope(t *testing.T, path string) dfrlPeerEnvelope {
	t.Helper()
	if secs := os.Getenv("DFRL_WAIT_SECS"); secs != "" {
		limit, err := strconv.Atoi(secs)
		if err != nil {
			t.Fatalf("DFRL_WAIT_SECS=%q is not a number", secs)
		}
		deadline := time.Now().Add(time.Duration(limit) * time.Second)
		for {
			if _, err := os.Stat(path); err == nil {
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("waited %ds for the peer envelope at %s and it did not appear; "+
					"either the other host has not run its publish step or the "+
					"directory is not syncing", limit, path)
			}
			time.Sleep(2 * time.Second)
		}
	}
	blob, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read peer envelope: %v", err)
	}
	var peer dfrlPeerEnvelope
	if err := json.Unmarshal(blob, &peer); err != nil {
		t.Fatalf("decode peer envelope: %v", err)
	}
	if peer.Envelope == nil || peer.Envelope.ExecutionPayload == nil {
		t.Fatal("peer record carries no execution payload")
	}
	t.Logf("peer envelope from %s", path)
	t.Logf("  order %s  host %s (%s/%s, %s, %d cpu)", peer.Order,
		peer.Host.Hostname, peer.Host.GOOS, peer.Host.GOARCH, peer.Host.GoVersion, peer.Host.NumCPU)
	t.Logf("  state root %v  block hash %v", peer.StateRoot, peer.BlockHash)
	return peer
}

// judgePeerEnvelope runs every guard and then the verdict.
//
// expectAgreementWhenOrdersDiffer says what this experiment claims when the two
// nodes compiled different orders: false for the unspecified arithmetic, true
// for the specified one. In the same-order cross-machine cell the expectation
// is agreement regardless, so the flag is ignored there.
//
// The guards exist because each of them, if violated, would turn an artifact
// into what looks like a result: a payload mangled in transport, two nodes that
// were never building over the same state, two nodes that compiled the same
// thing, or two nodes given different transactions to execute.
func judgePeerEnvelope(t *testing.T, api *ConsensusAPI, own *engine.ExecutionPayloadEnvelope,
	peer dfrlPeerEnvelope, expectAgreementWhenOrdersDiffer bool) {
	t.Helper()
	mode := dfrlXMMode()
	sameOrder := mode == dfrlXMSameOrder || mode == dfrlXMObserve
	expectAgreement := expectAgreementWhenOrdersDiffer || mode == dfrlXMSameOrder
	// Did anything besides the accumulation order vary? Decides how much a
	// same-order pass is worth and how little a same-order failure licenses.
	crossArch := peer.Host.GOARCH != runtime.GOARCH || peer.Host.GOOS != runtime.GOOS

	// Transport integrity: the block hash must recompute from the payload's own
	// fields, or the round trip changed something.
	if _, err := engine.ExecutableDataToBlock(*peer.Envelope.ExecutionPayload,
		[]common.Hash{}, &dfrlExchangeBeaconRoot, peer.Envelope.Requests); err != nil {
		t.Fatalf("peer payload did not survive transport intact (%v); "+
			"any verdict below would be a transport artifact, not a consensus verdict", err)
	}
	if peer.Envelope.ExecutionPayload.BlockHash != peer.BlockHash {
		t.Fatalf("peer block hash changed in transport: %v != %v",
			peer.Envelope.ExecutionPayload.BlockHash, peer.BlockHash)
	}

	// The two nodes must have compiled the relationship this run is testing.
	if sameOrder {
		if peer.Order != vm.DFRLVulnOrderName() {
			t.Fatalf("DFRL_XM_EXPECT=same-order but the peer compiled %s and this node %s",
				peer.Order, vm.DFRLVulnOrderName())
		}
		if peer.Host.Hostname == dfrlHostInfo().Hostname &&
			peer.Host.GOARCH == runtime.GOARCH && peer.Host.GOOS == runtime.GOOS {
			t.Fatalf("same-order cell ran twice on the same host (%s, %s/%s); "+
				"it only carries information across two different machines",
				peer.Host.Hostname, peer.Host.GOOS, peer.Host.GOARCH)
		}
		label := "same-order cell"
		if mode == dfrlXMObserve {
			label = "same-order OBSERVATION (not attribution: the hosts differ in more than the order)"
		}
		t.Logf("%s: %s on %s/%s vs %s/%s", label, peer.Order,
			peer.Host.GOOS, peer.Host.GOARCH, runtime.GOOS, runtime.GOARCH)
		if mode == dfrlXMSameOrder && crossArch {
			t.Logf("  the two hosts also differ in architecture, so agreement here is " +
				"stronger than a same-architecture pass, and a disagreement would be " +
				"inconclusive rather than a refutation")
		}
	} else if peer.Order == vm.DFRLVulnOrderName() {
		t.Fatalf("peer compiled the same accumulation order (%s); "+
			"run the second process with the other build tag, or set DFRL_XM_EXPECT=same-order",
			peer.Order)
	}

	// Same genesis, therefore same parent. Reported before the root comparison
	// so a genesis mismatch -- the likely failure across two machines -- is not
	// misread as a statement about the arithmetic.
	if peer.Envelope.ExecutionPayload.ParentHash != own.ExecutionPayload.ParentHash {
		t.Fatalf("peer built over a different parent (%v vs %v); the two genesis states are "+
			"not identical, so nothing below is a statement about the arithmetic",
			peer.Envelope.ExecutionPayload.ParentHash, own.ExecutionPayload.ParentHash)
	}

	// Same transactions, byte for byte. Vacuous for the transaction-free
	// mandatory path; load-bearing for the two transaction-driven experiments.
	if got, want := len(peer.Envelope.ExecutionPayload.Transactions), len(own.ExecutionPayload.Transactions); got != want {
		t.Fatalf("peer payload carries %d transactions, ours %d", got, want)
	}
	for i := range own.ExecutionPayload.Transactions {
		if string(peer.Envelope.ExecutionPayload.Transactions[i]) != string(own.ExecutionPayload.Transactions[i]) {
			t.Fatalf("transaction %d differs between the two nodes; the comparison "+
				"would be confounded by differing inputs", i)
		}
	}
	if n := len(own.ExecutionPayload.Transactions); n > 0 {
		t.Logf("both nodes built byte-identical transaction lists (%d txs)", n)
	}

	// The roots.
	if mode == dfrlXMObserve {
		agree := peer.StateRoot == own.ExecutionPayload.StateRoot
		t.Logf("OBSERVED roots: %s on %s/%s %v vs %s on %s/%s %v -- %s",
			peer.Order, peer.Host.GOOS, peer.Host.GOARCH, peer.StateRoot,
			vm.DFRLVulnOrderName(), runtime.GOOS, runtime.GOARCH, own.ExecutionPayload.StateRoot,
			map[bool]string{true: "IDENTICAL", false: "DIFFERENT"}[agree])
		status := submitPayload(t, api, peer.Envelope)
		t.Logf("OBSERVED verdict: %v (%v)", status.Status, errString(status.ValidationError))
		t.Logf("OBSERVATION: two hosts compiling %s, differing in architecture "+
			"(%s/%s vs %s/%s), produced %s post-state roots. This characterises "+
			"this pair of hosts. It does NOT attribute anything to the "+
			"accumulation order, because the order is not the only thing held "+
			"fixed here.", peer.Order, peer.Host.GOOS, peer.Host.GOARCH,
			runtime.GOOS, runtime.GOARCH,
			map[bool]string{true: "identical", false: "differing"}[agree])
		return
	}
	if expectAgreement {
		if peer.StateRoot != own.ExecutionPayload.StateRoot {
			if mode == dfrlXMSameOrder && crossArch {
				t.Fatalf("same-order cell disagreed, and these two hosts differ in more than "+
					"the accumulation order (%s/%s vs %s/%s). This does NOT show that the "+
					"differ cell's divergence is unattributable to the order: architecture "+
					"alone has been measured to move the unspecified path "+
					"(evidence/e3lite_arm64_vs_amd64.log), so it is a known alternative "+
					"cause here. For a conclusive answer use two hosts of the same "+
					"architecture; to record this pair's behaviour instead, set "+
					"DFRL_XM_EXPECT=same-order-observe. Roots: %s %v vs %s %v",
					peer.Host.GOOS, peer.Host.GOARCH, runtime.GOOS, runtime.GOARCH,
					vm.DFRLVulnOrderName(), own.ExecutionPayload.StateRoot, peer.Order, peer.StateRoot)
			}
			t.Fatalf("expected agreement but the roots differ: %s %v vs %s %v",
				vm.DFRLVulnOrderName(), own.ExecutionPayload.StateRoot, peer.Order, peer.StateRoot)
		}
		t.Logf("identical post-state root: %v", peer.StateRoot)
		if mode == dfrlXMSameOrder && crossArch {
			t.Logf("ATTRIBUTION: %s compiled on %s/%s (%s, %d cpu) and on %s/%s (%s, %d cpu) "+
				"committed the SAME post-state root. Architecture, toolchain and core count "+
				"all varied and the commitment did not move, so the divergence seen when the "+
				"order differs is attributable to the order and not to the host. Scope: this "+
				"holds for %s. Host-stability is a property of the order, not of the pipeline "+
				"-- the other order has to be tested separately and may well be host-sensitive.",
				peer.Order, peer.Host.GOOS, peer.Host.GOARCH, peer.Host.GoVersion, peer.Host.NumCPU,
				runtime.GOOS, runtime.GOARCH, runtime.Version(), runtime.NumCPU(), peer.Order)
		}
	} else {
		if peer.StateRoot == own.ExecutionPayload.StateRoot {
			t.Fatalf("both orders committed the same state root %v; there is nothing to reject",
				peer.StateRoot)
		}
		t.Logf("same parent, different post-state roots: %s %v vs %s %v",
			vm.DFRLVulnOrderName(), own.ExecutionPayload.StateRoot, peer.Order, peer.StateRoot)
	}

	// The verdict.
	status := submitPayload(t, api, peer.Envelope)
	t.Logf("peer payload verdict: %v (%v)", status.Status, errString(status.ValidationError))
	want := engine.INVALID
	if expectAgreement {
		want = engine.VALID
	}
	if status.Status != want {
		t.Fatalf("expected %v for the peer's payload, got %v", want, status.Status)
	}
	if expectAgreement {
		t.Logf("RESULT: %s node (%s/%s) and %s node (%s/%s) agree byte for byte and accept each other",
			peer.Order, peer.Host.GOOS, peer.Host.GOARCH,
			vm.DFRLVulnOrderName(), runtime.GOOS, runtime.GOARCH)
	} else {
		t.Logf("RESULT: %s node (%s/%s) built a payload the %s node (%s/%s) rejects, "+
			"with no field edited between them",
			peer.Order, peer.Host.GOOS, peer.Host.GOARCH,
			vm.DFRLVulnOrderName(), runtime.GOOS, runtime.GOARCH)
	}
}
