// SPDX-License-Identifier: LGPL-3.0-only
pragma solidity ^0.8.20;

import "./DFRLInputBuilder.sol";

/// @title DFRLRobustAggregator
/// @notice Phase 4 (U3): the same on-chain aggregation pattern as
///         DFRLAggregator.sol, but calling the RobustAgg precompile
///         (coordinate-wise trimmed mean, core/vm/contracts_dfrl.go's
///         DFRLRobustAggAddress = 0x0102) instead of FedAvg. Exists to show
///         the determinism methodology is not FedAvg-specific: trimmed mean
///         adds deterministic sorting, trimming, and a trim-count-dependent
///         denominator
///         to the same exact-integer specification used by FedAvg.
///
/// Scope: this contract demonstrates that a second aggregation rule commonly
/// used to bound the influence of individual clients can be made
/// consensus-safe the same way. It does NOT claim Byzantine fault tolerance
/// against adaptive or colluding adversaries -- DETERMINISM_SPEC.md's
/// non-goals list already excludes majority-collusion resistance, and that
/// scope is unchanged here.
/// This contract is likewise a controlled execution harness: the driver
/// supplies canonical tensors and controls round closure. Permissionless
/// admission checks and mandatory scheduling are outside this implementation.
///
/// Weights are accepted in uploadModel for ABI symmetry with DFRLAggregator
/// (both share DFRLInputBuilder's Entry encoding) but are ignored by
/// TrimmedMean itself (dfrl-agg/trimmed.go: "trimming exists to bound the
/// influence of any single client, which a weighted variant would
/// reintroduce").
contract DFRLRobustAggregator {
    /// DFRL_ROBUSTAGG precompile address, must match
    /// core/vm/contracts_dfrl.go's DFRLRobustAggAddress (0x0102).
    address constant DFRL_ROBUSTAGG = address(0x0102);

    struct Submission {
        uint64 weight;
        bytes tensor;
        bool submitted;
    }

    mapping(uint256 => address[]) public submitters;
    mapping(uint256 => mapping(address => Submission)) public submissions;
    mapping(uint256 => bytes) public aggregatedResult;
    mapping(uint256 => bytes32) public committedRoot;
    mapping(uint256 => bool) public aggregated;

    event ModelUploaded(uint256 indexed round, address indexed submitter, uint64 weight, uint256 tensorLen);
    event ModelAggregated(uint256 indexed round, bytes32 root, uint256 resultLen, uint256 nSubmitters, uint32 k);

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

    /// @notice Aggregate all submissions for `round` via coordinate-wise
    ///         trimmed mean, discarding the `k` smallest and `k` largest
    ///         values at each coordinate.
    /// @dev Precompile ABI (contracts_dfrl.go's robustAggPrecompile.Run):
    ///      4-byte big-endian k, followed by the same length-prefixed
    ///      (weight, tensor) sequence FedAvg's ABI uses.
    function aggregateModel(uint256 round, uint32 k) external returns (bytes32 root) {
        if (aggregated[round]) revert AlreadyAggregated(round);
        address[] memory subs = submitters[round];
        uint256 n = subs.length;
        if (n == 0) revert NoSubmissions(round);

        DFRLInputBuilder.Entry[] memory entries = new DFRLInputBuilder.Entry[](n);
        for (uint256 i = 0; i < n; i++) {
            Submission storage s = submissions[round][subs[i]];
            entries[i] = DFRLInputBuilder.Entry({weight: s.weight, tensor: s.tensor});
        }
        bytes memory body = DFRLInputBuilder.build(entries);
        bytes memory input = bytes.concat(bytes4(k), body);

        (bool ok, bytes memory result) = DFRL_ROBUSTAGG.staticcall(input);
        if (!ok) revert PrecompileFailed(round);

        aggregated[round] = true;
        aggregatedResult[round] = result;
        root = keccak256(result);
        committedRoot[round] = root;
        emit ModelAggregated(round, root, result.length, n, k);
    }

    function downloadModel(uint256 round) external view returns (bytes memory) {
        return aggregatedResult[round];
    }

    function submitterCount(uint256 round) external view returns (uint256) {
        return submitters[round].length;
    }
}
