# Deterministic Aggregation Specification

Version 0.2, 2026-08-24. This document defines the consensus-relevant behavior
implemented by `dfrl-agg`.

## 1. Fixed-point representation

A client represents each floating-point parameter as
`theta_hat = roundHalfEven(theta * 2^F)`. Quantization occurs before the
consensus boundary. The protocol fixes `F`; the default value is 20.

Every submitted tensor must satisfy these exclusive bounds:

- `abs(theta_hat) < 2^24` for every element;
- `1 <= weight < 2^19`;
- `1 <= number of clients < 2^10`.

Inputs outside these bounds are rejected. They are never clamped or allowed to
wrap. Since `24 + 19 + 10 = 53 < 63`, each weighted coordinate sum fits in a
signed 64-bit integer.

## 2. Weighted average

For client weights `s_n`, define `S = sum(s_n)` and, for every coordinate `j`,
`acc[j] = sum(s_n * theta_hat_n[j])`. Both sums use exact signed 64-bit integer
arithmetic. The output is
`theta_hat_agg[j] = divRoundHalfEven(acc[j], S)`.

`divRoundHalfEven(num, den)`, with `den > 0`, selects the nearest integer
quotient. An exact half is rounded to the even quotient. This is the only
rounding point in aggregation. The rule applies identically to positive and
negative numerators and does not inherit a language's default integer-division
semantics.

Because the accumulator cannot overflow and integer addition is associative
and commutative, the output is independent of client traversal order.

## 3. Coordinate-wise trimmed mean

For every coordinate, sort the submitted integer values, discard the `k`
smallest and `k` largest values, and apply `divRoundHalfEven` to the sum of the
remaining `N - 2k` values. The operation requires `0 <= 2k < N`. Weights are
ignored. Equal values are interchangeable, so sorting stability cannot change
the retained numerical multiset or the output.

## 4. Canonical tensor encoding

The byte representation uses network byte order and contains no padding or
optional fields:

```text
rank    uint8
dims    rank * uint32
F       uint8
length  uint64
data    length * int64 (two's complement)
```

Rank must be in `[1, 8]`; every dimension must be nonzero; `length` must equal
the product of the dimensions; and the body must contain exactly `length`
elements. Decoding rejects truncated input, trailing bytes, invalid shapes,
and out-of-range elements. The portable commitment is FIPS-202 SHA3-256 over
this canonical encoding.

## 5. Input-set and weight rules

Membership must be derived from chain state. For each round, accepted on-chain
submissions form the input set, and a later accepted submission from the same
address replaces the earlier one. The aggregator rejects duplicate addresses
if a caller fails to apply this rule. Weights are positive on-chain integers;
they cannot be derived from validator-local timing or floating-point metrics.

## 6. Agreement claim and boundary

Conforming implementations given the same protocol parameters, chain state,
canonical tensor bytes, integer weights, and operation produce byte-identical
outputs. The specification provides execution determinism. It does not assess
whether client updates are useful or honestly trained, provide model privacy,
or replace the underlying consensus trust assumptions.
