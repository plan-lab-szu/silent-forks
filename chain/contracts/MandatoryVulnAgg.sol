// SPDX-License-Identifier: LGPL-3.0-only
pragma solidity ^0.8.20;

/// @title Mandatory vulnerable aggregation target (E3-LITE / Engine API experiment only)
/// @notice Deployed at params.DFRLAggregatorAddress and invoked by the
/// height-scheduled system call in core.ProcessDFRLAggregation. It is NOT part
/// of the DFRL system: it aggregates in floating point, which is the design the
/// paper argues against.
///
/// The client updates are hardcoded rather than read from storage. That is a
/// deliberate simplification for this experiment and it strengthens rather than
/// weakens the result: both nodes execute byte-identical contract code over
/// byte-identical inputs from byte-identical genesis state, so the ONLY thing
/// that can differ between them is the summation order compiled into the
/// vulnerable precompile at 0x0103. Anything else that diverged would be a
/// confound, and there is nothing else left to diverge.
///
/// The hardcoded input is the attacker-closure witness from the diagnosis
/// section: three real federated-MNIST client weights and values, with one
/// client's own value moved by a single quantization step.
contract MandatoryVulnAgg {
    /// @dev Keccak-256 of the aggregate, one slot, so the system call's effect
    /// on the post-state root is a single storage write.
    bytes32 public lastCommitment;

    /// @dev Round index the call most recently ran for.
    uint256 public lastRound;

    address constant VULN_AGG = address(0x103);

    /// @dev The system call arrives with the round index as 32 bytes of
    /// calldata and no function selector, so it lands here.
    fallback() external {
        bytes memory input = hex"000000030000000000001d1700000001bc13f80000000000000007d9000000013c3640000000000000001dcc000000013bfc6800";

        (bool ok, bytes memory out) = VULN_AGG.staticcall(input);
        require(ok, "vuln agg failed");

        lastCommitment = keccak256(out);
        if (msg.data.length >= 32) {
            lastRound = abi.decode(msg.data, (uint256));
        }
    }
}
