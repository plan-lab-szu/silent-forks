package catalyst

import (
	"context"
	"strings"
	"testing"

	"github.com/ethereum/go-ethereum/beacon/engine"
	"github.com/ethereum/go-ethereum/common"
)

// stateRootFromReverseBuild is the post-state root that the SECOND conforming
// implementation actually produced, measured -- not invented -- by
// TestMandatoryVulnAggPostStateRoot in package core:
//
//	seq-fwd  0xed1ea05bf71a4c22c45966214a0ae1afa1a9a5a4544d702748aeca159628f4b1
//	seq-rev  0x7f6373be5bf958f0599b4a60993b115651545d18c63af99c9fa4b82249e8784f
//
// Two binaries built from one source tree with different accumulation orders,
// on one machine and one architecture, over byte-identical genesis state and
// byte-identical contract code, with no transactions in the block. See
// evidence/mandatory_aggregation_divergence.log.
//
// The value's only role here is to be a state root that some conforming
// implementation genuinely committed and that this node did not. Any such root
// would do; using the measured one keeps the two experiments composable rather
// than hypothetical.
const stateRootFromReverseBuild = "0x7f6373be5bf958f0599b4a60993b115651545d18c63af99c9fa4b82249e8784f"

// TestDFRLDivergentStateRootIsINVALID completes the liveness argument's final
// step. Package core establishes that two conforming implementations commit
// different post-state roots at a mandatory aggregation boundary. What remains
// is the consensus layer's verdict on such a payload, and it must come from the
// Engine API rather than from our own comparison of two hex strings.
//
// The payload delivered here is internally consistent: its block hash is
// recomputed after the state root is substituted, so it cannot be rejected for
// a header/hash mismatch. The only thing wrong with it is that re-executing its
// (empty) transaction list over the same parent yields a different state root
// than the header claims -- exactly the position a validator is in when it
// receives a block built by a peer whose aggregation differs.
//
// Rejection therefore has to come from re-execution, which is what
// core/block_validator.go reports as "invalid merkle root".
func TestDFRLDivergentStateRootIsINVALID(t *testing.T) {
	genesis, preMergeBlocks := generateMergeChain(10, false)
	n, ethservice := startEthService(t, genesis, preMergeBlocks)
	defer n.Close()

	parent := ethservice.BlockChain().CurrentBlock()
	api := newConsensusAPIWithoutHeartbeat(ethservice)

	// Control: the node's own payload over this parent is accepted.
	good := getNewPayload(t, api, parent, nil, nil)
	goodStatus, err := api.NewPayloadV1(context.Background(), *good)
	if err != nil {
		t.Fatal(err)
	}
	if goodStatus.Status != engine.VALID {
		t.Fatalf("control payload should be VALID, got %v (%v)", goodStatus.Status, errString(goodStatus.ValidationError))
	}
	t.Logf("control: node's own payload over parent %v is %v", parent.Hash(), goodStatus.Status)
	t.Logf("         its state root %v", good.StateRoot)

	// The attack case: same parent, same (empty) transaction list, but the
	// state root a differently-built peer would have committed.
	diverged := getNewPayload(t, api, parent, nil, nil)
	diverged.StateRoot = common.HexToHash(stateRootFromReverseBuild)
	if diverged.StateRoot == good.StateRoot {
		t.Fatal("substituted root equals the local root; the test would prove nothing")
	}
	diverged = setBlockhash(diverged)

	status, err := api.NewPayloadV1(context.Background(), *diverged)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("divergent payload: state root %v", diverged.StateRoot)
	t.Logf("                   engine_newPayload -> %v", status.Status)
	if status.ValidationError != nil {
		t.Logf("                   validationError: %v", *status.ValidationError)
	}

	if status.Status != engine.INVALID {
		t.Fatalf("expected INVALID for a payload whose state root disagrees with re-execution, got %v", status.Status)
	}
	// The reason matters as much as the verdict: an INVALID for some unrelated
	// structural defect would not support the claim.
	if status.ValidationError == nil || !strings.Contains(strings.ToLower(*status.ValidationError), "merkle root") {
		t.Fatalf("expected rejection to cite the merkle root, got %v", errString(status.ValidationError))
	}
}

func errString(s *string) string {
	if s == nil {
		return "<nil>"
	}
	return *s
}
