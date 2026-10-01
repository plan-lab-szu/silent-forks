# Consensus-client provenance

This directory is a source snapshot of the go-ethereum fork used by the
artifact. It is based on the `dfrl-consensus` branch at commit
`b2548e9788654759d47a6551347bedc3f66918af` and includes the subsequent working
tree changes used for the reported experiments.

The DFRL-specific implementation is concentrated in:

- `core/state_processor.go` and `core/state_processor_dfrl*_test.go`;
- `core/vm/contracts_dfrl*.go` and `core/vm/instructions_dfrl*.go`;
- `eth/catalyst/api_dfrl*.go`; and
- `params/config.go` and `params/protocol_params.go`.

The snapshot omits nested Git metadata, upstream CI configuration, and
uninitialized upstream test-data submodules. The go-ethereum license terms are
preserved in `COPYING` and `COPYING.LESSER`.
