package dfrlagg

import (
	"crypto/sha3"
	"encoding/binary"
)

// Canonical encoding (§4.2): two logically equal tensors MUST encode to
// identical bytes, and every byte string decodes to at most one tensor.
//
// Layout, all big-endian (network order), no padding, no optional fields:
//
//	rank   uint8
//	dims   uint32 * rank
//	F      uint8
//	length uint64
//	data   int64 * length   (two's complement, fixed width)
//
// Rejecting non-canonical encodings is a security requirement, not fastidiousness:
// if one tensor had several valid encodings, an adversary could submit different
// bytes for the same model and obtain a different state root — a malleability
// attack surface.

const (
	headerFixed = 1 + 1 + 8 // rank + F + length
	// The chain-call ABI prefixes each canonical tensor with a uint32 byte
	// length.  Keeping the codec under the same bound prevents a tensor that is
	// valid in the library from becoming unrepresentable at the EVM boundary.
	maxCanonicalTensorBytes uint64 = 1<<32 - 1
)

// Encode returns the canonical byte string for t. The tensor must already be
// valid; callers on the consensus path validate on decode.
func Encode(t Tensor) ([]byte, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	buf := make([]byte, 0, headerFixed+4*len(t.Dims)+8*len(t.Data))
	buf = append(buf, uint8(len(t.Dims)))
	for _, d := range t.Dims {
		buf = binary.BigEndian.AppendUint32(buf, d)
	}
	buf = append(buf, t.F)
	buf = binary.BigEndian.AppendUint64(buf, uint64(len(t.Data)))
	for _, v := range t.Data {
		buf = binary.BigEndian.AppendUint64(buf, uint64(v))
	}
	return buf, nil
}

// Decode parses a canonical encoding, rejecting anything non-canonical:
// a rank outside [1, MaxRank], a zero dimension, a length inconsistent with
// dims, a truncated body, or trailing bytes. Element ranges are checked too,
// so a decoded tensor is always a legal submission (§2.2).
func Decode(b []byte) (Tensor, error) {
	if len(b) < 1 {
		return Tensor{}, ErrEncoding
	}
	rank := int(b[0])
	if rank < 1 || rank > MaxRank {
		return Tensor{}, ErrEncoding
	}
	if len(b) < 1+4*rank+1+8 {
		return Tensor{}, ErrEncoding
	}
	off := 1
	dims := make([]uint32, rank)
	for i := 0; i < rank; i++ {
		dims[i] = binary.BigEndian.Uint32(b[off:])
		off += 4
	}
	f := b[off]
	off++
	n := binary.BigEndian.Uint64(b[off:])
	off += 8

	want, ok := numel(dims)
	if !ok || n != want {
		return Tensor{}, ErrEncoding
	}
	if uint64(len(b)-off) != 8*n { // no truncation, no trailing bytes
		return Tensor{}, ErrEncoding
	}
	data := make([]int64, n)
	for i := uint64(0); i < n; i++ {
		data[i] = int64(binary.BigEndian.Uint64(b[off:]))
		off += 8
	}
	t := Tensor{Dims: dims, F: f, Data: data}
	if err := t.Validate(); err != nil {
		return Tensor{}, err
	}
	return t, nil
}

// StateRoot is the commitment we compare across validators and across
// languages: SHA3-256 over the canonical encoding.
//
// Note for the paper: this is FIPS-202 SHA3-256, matching Python's
// hashlib.sha3_256 so that the cross-language check compares like with like.
// Ethereum's state trie uses Keccak-256, which differs in padding. The choice
// is immaterial to the divergence argument — any collision-resistant hash
// turns "different bytes" into "different root" — but the distinction must be
// stated correctly rather than blurred into "keccak".
func StateRoot(t Tensor) ([32]byte, error) {
	b, err := Encode(t)
	if err != nil {
		return [32]byte{}, err
	}
	return sha3.Sum256(b), nil
}
