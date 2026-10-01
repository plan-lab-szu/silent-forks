// SPDX-License-Identifier: LGPL-3.0-only
pragma solidity ^0.8.20;

/// @title DFRLInputBuilder
/// @notice Shared by DFRLAggregator.sol and DFRLAggregatorCommitOnly.sol so
///         the two contracts cannot silently diverge on how they build the
///         DFRL precompile's calldata ABI -- exactly the kind of drift this
///         whole project exists to eliminate.
///
/// Replaces the original single-line implementation:
///
///     bytes memory input = abi.encodePacked(uint32(n));
///     for (uint256 i = 0; i < n; i++) {
///         input = abi.encodePacked(input, s.weight, uint32(s.tensor.length), s.tensor);
///     }
///
/// which is O(n^2) in memory-copy cost: `abi.encodePacked(input, ...)`
/// allocates a FRESH buffer and copies the ENTIRE growing `input` on every
/// iteration. Measured impact (see HANDBOOK.md's U4 section): at N=5,
/// |Theta|=4096, this dominated aggregateModel's total gas by roughly 64x
/// over the precompile's own cost -- the reported absolute gas numbers were
/// almost entirely an artifact of this loop, not of on-chain aggregation.
///
/// Fix: measure the exact total length first (one pass over storage, no
/// allocation), allocate ONE buffer of that exact size, then fill it in
/// place using MCOPY (EIP-5656, available from Cancun; this chain activates
/// it from genesis, see chain/testbed/genesis.json's cancunTime=0).
library DFRLInputBuilder {
    struct Entry {
        uint64 weight;
        bytes tensor;
    }

    /// @dev Builds the precompile ABI: uint32(n) || (weight:8 || tensorLen:4
    ///      || tensor)*n, all big-endian, matching contracts_dfrl.go's
    ///      decodeDFRLInput exactly (see that function's doc comment for the
    ///      layout this must produce byte-for-byte).
    function build(Entry[] memory entries) internal pure returns (bytes memory input) {
        uint256 n = entries.length;

        // Pass 1: exact total size, no memory allocation.
        uint256 totalLen = 4; // uint32 n
        for (uint256 i = 0; i < n; i++) {
            totalLen += 12 + entries[i].tensor.length; // weight(8) + tensorLen(4) + tensor
        }

        // Pass 2: allocate exactly once, fill via MCOPY. Every write below
        // goes through MCOPY with an EXACT byte count -- never a bare MSTORE
        // into the destination buffer, which would write a full 32-byte word
        // and could spill past the buffer's true end for a small final tensor
        // (verified this matters: a 1-element tensor's canonical encoding is
        // 22 bytes, well under the 32-byte word MSTORE would touch). Header
        // fields are staged in Solidity's reserved scratch space [0x00,0x40)
        // first, where an oversized MSTORE is always safe, then MCOPY'd out
        // with the exact length needed.
        input = new bytes(totalLen);
        uint256 offset = 4;
        assembly {
            mstore(0x00, shl(224, n))
            mcopy(add(input, 32), 0x00, 4)
        }
        for (uint256 i = 0; i < n; i++) {
            uint64 w = entries[i].weight;
            bytes memory tensor = entries[i].tensor;
            uint256 tlen = tensor.length;
            assembly {
                // headerWord: weight (64 bits) || tensorLen (32 bits), 96 bits total.
                let headerWord := or(shl(32, w), tlen)
                mstore(0x00, shl(160, headerWord)) // left-align 12 meaningful bytes in scratch
                let dst := add(add(input, 32), offset)
                mcopy(dst, 0x00, 12)
                mcopy(add(dst, 12), add(tensor, 32), tlen)
            }
            offset += 12 + tlen;
        }
    }
}
