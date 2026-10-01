package dfrlagg

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestCodecRoundTrip(t *testing.T) {
	cases := []Tensor{
		{Dims: []uint32{1}, F: DefaultF, Data: []int64{0}},
		{Dims: []uint32{3}, F: 12, Data: []int64{-1, 0, 1}},
		{Dims: []uint32{2, 3}, F: DefaultF, Data: []int64{1, -2, 3, -4, 5, -6}},
		{Dims: []uint32{2, 2, 2}, F: 24, Data: []int64{
			ParamBound - 1, -(ParamBound - 1), 7, -7, 0, 1, -1, 2}},
	}
	for _, want := range cases {
		b, err := Encode(want)
		if err != nil {
			t.Fatal(err)
		}
		got, err := Decode(b)
		if err != nil {
			t.Fatal(err)
		}
		rb, err := Encode(got)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(b, rb) {
			t.Fatalf("re-encode differs for dims=%v", want.Dims)
		}
	}
}

// TestEncodeIsCanonical: logically equal tensors must encode identically, and
// the encoding must be a pure function of the logical value.
func TestEncodeIsCanonical(t *testing.T) {
	a := Tensor{Dims: []uint32{2, 2}, F: DefaultF, Data: []int64{5, -5, 0, 12345}}
	b := a.Clone()
	ea, err := Encode(a)
	if err != nil {
		t.Fatal(err)
	}
	eb, err := Encode(b)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(ea, eb) {
		t.Fatal("logically equal tensors encoded differently")
	}
	// A different shape over the same elements must NOT collide: shape is part
	// of the commitment.
	c := Tensor{Dims: []uint32{4}, F: DefaultF, Data: []int64{5, -5, 0, 12345}}
	ec, err := Encode(c)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(ea, ec) {
		t.Fatal("reshaped tensor collided with original encoding")
	}
}

// TestDecodeRejectsNonCanonical is the malleability check of §4.2: if one
// tensor had several accepted encodings, an adversary could submit different
// bytes for the same model and obtain a different state root.
func TestDecodeRejectsNonCanonical(t *testing.T) {
	valid, err := Encode(Tensor{Dims: []uint32{2, 2}, F: DefaultF, Data: []int64{1, 2, 3, 4}})
	if err != nil {
		t.Fatal(err)
	}

	mutate := func(f func(b []byte) []byte) []byte { return f(append([]byte(nil), valid...)) }

	bad := map[string][]byte{
		"empty":          {},
		"rank zero":      mutate(func(b []byte) []byte { b[0] = 0; return b }),
		"rank too big":   mutate(func(b []byte) []byte { b[0] = MaxRank + 1; return b }),
		"trailing byte":  mutate(func(b []byte) []byte { return append(b, 0) }),
		"truncated body": mutate(func(b []byte) []byte { return b[:len(b)-1] }),
		"header only":    mutate(func(b []byte) []byte { return b[:5] }),
		"zero dim": mutate(func(b []byte) []byte {
			binary.BigEndian.PutUint32(b[1:], 0)
			return b
		}),
		"length disagrees with dims": mutate(func(b []byte) []byte {
			binary.BigEndian.PutUint64(b[1+4*2+1:], 3)
			return b
		}),
		"element out of range": mutate(func(b []byte) []byte {
			binary.BigEndian.PutUint64(b[len(b)-8:], uint64(ParamBound))
			return b
		}),
	}
	for name, b := range bad {
		if _, err := Decode(b); err == nil {
			t.Errorf("%s: Decode accepted a non-canonical encoding", name)
		}
	}
	if _, err := Decode(valid); err != nil {
		t.Errorf("valid encoding rejected: %v", err)
	}
}

func TestTensorRejectsOverflowingDimensionProduct(t *testing.T) {
	tensor := Tensor{Dims: []uint32{1 << 31, 1 << 31, 4}, F: DefaultF}
	if err := tensor.Validate(); err != ErrEncoding {
		t.Fatalf("overflowing dimension product: got %v, want ErrEncoding", err)
	}
}

func TestStateRootDistinguishes(t *testing.T) {
	a := Tensor{Dims: []uint32{2}, F: DefaultF, Data: []int64{1, 2}}
	b := Tensor{Dims: []uint32{2}, F: DefaultF, Data: []int64{1, 3}}
	c := Tensor{Dims: []uint32{2}, F: DefaultF + 1, Data: []int64{1, 2}}
	ra, _ := StateRoot(a)
	rb, _ := StateRoot(b)
	rc, _ := StateRoot(c)
	if ra == rb || ra == rc {
		t.Fatal("state root failed to distinguish inputs")
	}
}
