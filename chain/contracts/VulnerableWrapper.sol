// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

// E3-LITE NEGATIVE-CONTROL ARTIFACT. Pairs with
// core/vm/contracts_dfrl_vulnerable_e3lite.go (address 0x...0103), which only
// exists in a geth binary built with -tags e3lite_vulnerable. This contract
// is not part of the DFRL system and must never be deployed against the real
// FedAvg/RobustAgg precompiles (0x101/0x102).
//
// The point of writing the precompile's output to STORAGE, not just
// returning it, is that a divergent precompile result only becomes a
// divergent STATE ROOT -- and therefore something two validators executing
// the identical block can actually disagree about -- if it lands in state.
// Call-return data alone is not part of consensus state.
contract VulnerableWrapper {
    bytes32 public lastRoot;
    bytes public lastOutput;

    event Aggregated(bytes32 root, bytes output);

    address constant PRECOMPILE = address(0x103);

    // input is the vulnerable precompile's raw ABI (see
    // contracts_dfrl_vulnerable_e3lite.go's decodeDFRLVulnInput doc comment):
    // numClients uint32, then per client (weight uint64, paramCount uint32,
    // theta as big-endian float32 words).
    function agg(bytes calldata input) external returns (bytes32) {
        (bool ok, bytes memory out) = PRECOMPILE.staticcall(input);
        require(ok, "precompile call failed");
        lastOutput = out;
        bytes32 root = keccak256(out);
        lastRoot = root;
        emit Aggregated(root, out);
        return root;
    }
}
