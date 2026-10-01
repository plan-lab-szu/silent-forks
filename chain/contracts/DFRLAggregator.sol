// SPDX-License-Identifier: LGPL-3.0-only
pragma solidity ^0.8.20;

import "./DFRLInputBuilder.sol";

/// @title DFRLAggregator
/// @notice Reference contract for on-chain deterministic federated aggregation.
///         Not part of upstream go-ethereum; this contract calls the DFRL precompile
///         added in core/vm/contracts_dfrl.go.
///
/// Membership rule (spec section 4.1): the input set for round r is exactly
/// the set of UploadModel calls accepted while round r is open, and a second
/// upload from the same address REPLACES the first rather than adding a
/// second entry -- so membership is a deterministic function of on-chain
/// state, never of wall-clock timing.
///
/// Storage policy: this contract stores the FULL result on chain
/// (`aggregatedResult`) for functional testing at small scale, AND separately
/// computes a commitment (`committedRoot = keccak256(result)`), so the gas
/// cost of "full payload in SSTORE" versus "commitment only" can be measured
/// directly against each other (see contracts_test.go's
/// TestU4StorageGasComparison) -- this is the quantitative form of R1-6c/U4's
/// argument, not an assertion. A production deployment at realistic model
/// sizes (e.g. 250k parameters) would use the commitment-only path and serve
/// the payload from calldata/event logs, not from `aggregatedResult`; that
/// off-chain-retrieval wiring is outside this experimental contract.
///
/// Integration scope: this is a controlled experimental harness, not a
/// permissionless round-admission protocol. The experiment driver supplies
/// canonical, range-valid tensors and decides when to call aggregateModel.
/// A deployment exposed to arbitrary clients must validate submissions when
/// admitted and must authorize or schedule round closure; those policy steps
/// are deliberately outside this contract's execution-path comparison.
contract DFRLAggregator {
    /// DFRL_FEDAVG precompile address, must match
    /// core/vm/contracts_dfrl.go's DFRLFedAvgAddress (0x0101).
    address constant DFRL_FEDAVG = address(0x0101);

    struct Submission {
        uint64 weight;
        bytes tensor; // dfrl-agg canonical encoding
        bool submitted;
    }

    // round => submitter list (insertion order; a resubmission does not
    // create a new entry, see uploadModel)
    mapping(uint256 => address[]) public submitters;
    // round => submitter => submission (last write wins, per spec 4.1)
    mapping(uint256 => mapping(address => Submission)) public submissions;
    // round => aggregated result, full bytes (small-scale testing path)
    mapping(uint256 => bytes) public aggregatedResult;
    // round => commitment (production path, U4)
    mapping(uint256 => bytes32) public committedRoot;
    // round => whether aggregateModel has already run (rounds are one-shot)
    mapping(uint256 => bool) public aggregated;

    event ModelUploaded(uint256 indexed round, address indexed submitter, uint64 weight, uint256 tensorLen);
    event ModelAggregated(uint256 indexed round, bytes32 root, uint256 resultLen, uint256 nSubmitters);

    error AlreadyAggregated(uint256 round);
    error NoSubmissions(uint256 round);
    error PrecompileFailed(uint256 round);

    /// @notice Submit (or replace) this sender's local update for `round`.
    /// @param tensor Canonical dfrl-agg encoding of the quantized local model.
    function uploadModel(uint256 round, uint64 weight, bytes calldata tensor) external {
        if (aggregated[round]) revert AlreadyAggregated(round);
        if (!submissions[round][msg.sender].submitted) {
            submitters[round].push(msg.sender);
        }
        submissions[round][msg.sender] = Submission({weight: weight, tensor: tensor, submitted: true});
        emit ModelUploaded(round, msg.sender, weight, tensor.length);
    }

    /// @notice Aggregate all submissions for `round` via the DFRL precompile.
    /// @dev Builds the precompile's ABI (see contracts_dfrl.go): a
    ///      length-prefixed sequence of (weight uint64, tensorLen uint32,
    ///      tensor bytes) entries, in `submitters[round]` order. Order does
    ///      not affect the result (Proposition 2, dfrl-agg/fedavg.go), only
    ///      the bytes built here, which is why this contract does not need
    ///      to sort submitters.
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
        aggregatedResult[round] = result;
        root = keccak256(result);
        committedRoot[round] = root;
        emit ModelAggregated(round, root, result.length, n);
    }

    /// @notice Read back the full aggregated result (small-scale testing path
    ///         only -- see storage-policy note above).
    function downloadModel(uint256 round) external view returns (bytes memory) {
        return aggregatedResult[round];
    }

    function submitterCount(uint256 round) external view returns (uint256) {
        return submitters[round].length;
    }
}
