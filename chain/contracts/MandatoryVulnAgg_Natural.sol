// SPDX-License-Identifier: LGPL-3.0-only
pragma solidity ^0.8.20;

/// @title Mandatory vulnerable aggregation (UNPERTURBED natural training input)
/// @notice Same role as MandatoryVulnAgg.sol but the hardcoded input is an
/// unperturbed 16-coordinate window from real federated-MNIST round 0:
/// 10 clients, original training weights, no client-side perturbation.
/// Window coordinates [112, 128) of 7850.
///
/// This closes the narrative gap between the scope-grid proof (natural
/// divergence at the commitment level) and the chain-level payload exchange.
contract MandatoryVulnAgg_Natural {
    bytes32 public lastCommitment;
    uint256 public lastRound;

    address constant VULN_AGG = address(0x103);

    fallback() external {
        bytes memory input = hex"0000000a0000000000001d1700000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc1000000000000000007d900000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000001dcc00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc1000000000000000018ed00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000001e6900000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc100000000000000000c5400000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000000000000000266200000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79d000bcca02003bca20003cd0b600bc240000bc4cf0003bc10000000000000000147a00000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c795400bc7a2400bcca16003bc9d8003cd02600bc242800bc46e8003bc07000000000000000130600000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000000000000000161800000010bafae0003c0b9c0039b2000039100000bbea28003b99e000bc2964003bda30003c79b000bc79cc00bcca08003bca20003cd0ba00bc23fc00bc4ce8003bc10000";

        (bool ok, bytes memory out) = VULN_AGG.staticcall(input);
        require(ok, "vuln agg failed");

        lastCommitment = keccak256(out);
        if (msg.data.length >= 32) {
            lastRound = abi.decode(msg.data, (uint256));
        }
    }
}
