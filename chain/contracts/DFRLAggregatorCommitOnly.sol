// SPDX-License-Identifier: LGPL-3.0-only
pragma solidity ^0.8.20;

import "./DFRLInputBuilder.sol";

/// @title DFRLAggregatorCommitOnly
/// @notice Identical to DFRLAggregator.sol except aggregateModel does NOT
///         store the full result on chain -- only the commitment. Exists
///         solely to measure, by direct A/B comparison on identical inputs,
///         the SSTORE cost that storing the full payload adds. This is the
///         quantitative form of R1-6c/U4's argument (raw on-chain storage of
///         a model does not scale) instead of an assertion.
/// @dev Like the full-storage contract, this is a controlled measurement
///      harness. It assumes canonical submissions and driver-controlled round
///      closure; it does not implement permissionless admission or scheduling.
contract DFRLAggregatorCommitOnly {
    address constant DFRL_FEDAVG = address(0x0101);

    struct Submission {
        uint64 weight;
        bytes tensor;
        bool submitted;
    }

    mapping(uint256 => address[]) public submitters;
    mapping(uint256 => mapping(address => Submission)) public submissions;
    mapping(uint256 => bytes32) public committedRoot;
    mapping(uint256 => bool) public aggregated;

    event ModelUploaded(uint256 indexed round, address indexed submitter, uint64 weight, uint256 tensorLen);
    event ModelAggregated(uint256 indexed round, bytes32 root, uint256 resultLen, uint256 nSubmitters);

    error AlreadyAggregated(uint256 round);
    error NoSubmissions(uint256 round);
    error PrecompileFailed(uint256 round);

    function uploadModel(uint256 round, uint64 weight, bytes calldata tensor) external {
        if (aggregated[round]) revert AlreadyAggregated(round);
        if (!submissions[round][msg.sender].submitted) {
            submitters[round].push(msg.sender);
        }
        submissions[round][msg.sender] = Submission({weight: weight, tensor: tensor, submitted: true});
        emit ModelUploaded(round, msg.sender, weight, tensor.length);
    }

    /// @dev Same aggregation logic as DFRLAggregator.aggregateModel, but the
    ///      result is only committed (keccak256), never SSTORE'd in full.
    ///      A production deployment would emit `result` in the event above
    ///      (calldata-cheap) and let off-chain readers reconstruct the
    ///      payload, verifying it against `committedRoot`.
    function aggregateModel(uint256 round) external returns (bytes32 root) {
        if (aggregated[round]) revert AlreadyAggregated(round);
        address[] memory subs = submitters[round];
        uint256 n = subs.length;
        if (n == 0) revert NoSubmissions(round);

        DFRLInputBuilder.Entry[] memory entries = new DFRLInputBuilder.Entry[](n);
        for (uint256 i = 0; i < n; i++) {
            Submission storage s = submissions[round][subs[i]];
            entries[i] = DFRLInputBuilder.Entry({weight: s.weight, tensor: s.tensor});
        }
        bytes memory input = DFRLInputBuilder.build(entries);

        (bool ok, bytes memory result) = DFRL_FEDAVG.staticcall(input);
        if (!ok) revert PrecompileFailed(round);

        aggregated[round] = true;
        root = keccak256(result);
        committedRoot[round] = root;
        emit ModelAggregated(round, root, result.length, n);
    }

    function submitterCount(uint256 round) external view returns (uint256) {
        return submitters[round].length;
    }
}
