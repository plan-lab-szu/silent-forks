package dfrlagg

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestExtractNaturalWitness finds a 16-coordinate window from unperturbed
// round-0 training data where seq-fwd and seq-rev naturally diverge, and
// outputs the vulnerable precompile's binary encoding for use in a Solidity
// contract. The input is protocol-valid Q20 submissions from ordinary
// federated training without client-side perturbation.
//
// Run:
//
//	go test -v -run TestExtractNaturalWitness ./...
//
// The test prints the hex encoding and writes:
//   - chain/contracts/MandatoryVulnAgg_Natural.sol
//   - The precompile input hex for dfrlTxvWitnessInput
func TestExtractNaturalWitness(t *testing.T) {
	theta, weights := loadRealRound(t, 0)
	n := len(theta)
	p := len(theta[0])
	t.Logf("round 0: N=%d clients, P=%d params", n, p)

	const windowSize = 16

	// Scan non-overlapping windows. We want one where:
	//  1. seq-fwd ≠ seq-rev (so the float path diverges)
	//  2. at least some clients have different theta values (so the witness
	//     is visibly heterogeneous, not just "identical values × different weights")
	found := false
	var bestOff int
	for off := 0; off+windowSize <= p; off += windowSize {
		win := sliceTheta(theta, off, windowSize)

		// Check client diversity: count coords where not all clients agree.
		diverseCoords := 0
		for j := 0; j < windowSize; j++ {
			for i := 1; i < n; i++ {
				if win[i][j] != win[0][j] {
					diverseCoords++
					break
				}
			}
		}

		fwd := FloatAvg(win, weights, SeqFwd)
		rev := FloatAvg(win, weights, SeqRev)
		rootFwd := FloatRoot(fwd)
		rootRev := FloatRoot(rev)
		if rootFwd != rootRev && diverseCoords > 0 {
			bestOff = off
			found = true
			t.Logf("FOUND divergent + diverse window at offset %d (coords [%d, %d))",
				off, off, off+windowSize)
			t.Logf("  %d of %d coords have cross-client diversity", diverseCoords, windowSize)
			t.Logf("  seq-fwd root: %x", rootFwd)
			t.Logf("  seq-rev root: %x", rootRev)

			diffs := 0
			for j := 0; j < windowSize; j++ {
				if fwd[j] != rev[j] {
					diffs++
					t.Logf("  coord %d: fwd=%.10g rev=%.10g delta=%g",
						off+j, fwd[j], rev[j], fwd[j]-rev[j])
				}
			}
			t.Logf("  %d of %d coordinates in aggregate differ", diffs, windowSize)
			break
		}
		if rootFwd != rootRev && !found {
			t.Logf("  offset %d: divergent but all-identical theta (diverseCoords=0), skipping", off)
		}
	}
	if !found {
		t.Fatal("no divergent 16-coordinate window found in round 0")
	}

	// Verify all six strategies on this window.
	win := sliceTheta(theta, bestOff, windowSize)
	nDistinct, byRoot := distinctRoots(win, weights)
	t.Logf("all strategies on window [%d, %d): %d distinct commitments", bestOff, bestOff+windowSize, nDistinct)
	for root, strats := range byRoot {
		t.Logf("  %x → %v", root[:8], strats)
	}

	// Encode the precompile input.
	inputHex := encodeVulnPrecompileInput(win, weights)
	t.Logf("precompile input (%d bytes, %d hex chars):", len(inputHex)/2, len(inputHex))
	t.Logf("  %s", inputHex)

	// Write Solidity contract.
	solPath := filepath.Join("..", "chain", "contracts", "MandatoryVulnAgg_Natural.sol")
	sol := fmt.Sprintf(`// SPDX-License-Identifier: LGPL-3.0-only
pragma solidity ^0.8.20;

/// @title Mandatory vulnerable aggregation (UNPERTURBED natural training input)
/// @notice Same role as MandatoryVulnAgg.sol but the hardcoded input is an
/// unperturbed 16-coordinate window from real federated-MNIST round 0:
/// 10 clients, original training weights, no client-side perturbation.
/// Window coordinates [%d, %d) of 7850.
///
/// This closes the narrative gap between the scope-grid proof (natural
/// divergence at the commitment level) and the chain-level payload exchange.
contract MandatoryVulnAgg_Natural {
    bytes32 public lastCommitment;
    uint256 public lastRound;

    address constant VULN_AGG = address(0x103);

    fallback() external {
        bytes memory input = hex"%s";

        (bool ok, bytes memory out) = VULN_AGG.staticcall(input);
        require(ok, "vuln agg failed");

        lastCommitment = keccak256(out);
        if (msg.data.length >= 32) {
            lastRound = abi.decode(msg.data, (uint256));
        }
    }
}
`, bestOff, bestOff+windowSize, inputHex)

	if err := os.MkdirAll(filepath.Dir(solPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(solPath, []byte(sol), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s", solPath)

	// Also output the witness for the tx-path test (same hex, prefixed with 0x).
	t.Logf("dfrlTxvWitnessInput (for api_dfrl_txpath natural variant):")
	t.Logf("  0x%s", inputHex)

	// Output the per-client data for the defence-control variant.
	t.Logf("defence-control witness (10 clients × 16 coords):")
	for i := 0; i < n; i++ {
		t.Logf("  client %d: weight=%d theta[%d..%d]=%v",
			i, int(weights[i]), bestOff, bestOff+windowSize, win[i])
	}
}

// encodeVulnPrecompileInput encodes theta[N][P] and weights[N] in the format
// contracts_dfrl_vulnerable_e3lite.go expects:
//
//	numClients   uint32
//	per client:  weight uint64, paramCount uint32, theta [P]float32 (all big-endian)
func encodeVulnPrecompileInput(theta [][]float32, weights []float32) string {
	n := len(theta)
	p := len(theta[0])
	size := 4 + n*(8+4+p*4)
	buf := make([]byte, size)
	off := 0

	binary.BigEndian.PutUint32(buf[off:], uint32(n))
	off += 4

	for i := 0; i < n; i++ {
		binary.BigEndian.PutUint64(buf[off:], uint64(weights[i]))
		off += 8
		binary.BigEndian.PutUint32(buf[off:], uint32(p))
		off += 4
		for j := 0; j < p; j++ {
			binary.BigEndian.PutUint32(buf[off:], math.Float32bits(theta[i][j]))
			off += 4
		}
	}

	return hex.EncodeToString(buf)
}
