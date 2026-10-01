"""Independent Python reference implementation of DETERMINISM_SPEC.md.

This exists to make the determinism claim falsifiable in the way that matters:
the specification claims that ANY conforming implementation, in any language,
produces identical bytes. A Go module agreeing with itself does not test that.
So this file is written from the specification text, in a different language,
with different integer semantics -- Python's // floors while Go's / truncates,
which is exactly divergence source 4 in section 4 -- and must land on the same
state roots.

Pure standard library on purpose: no numpy anywhere in the integer path, so
that a library version can play no role. (fp_divergence.py, the negative
control, is where numpy versions matter and that is the point of it.)
"""

import hashlib
import struct

# Protocol parameters, section 2.2. Mirror of limits.go.
PARAM_BITS = 24
WEIGHT_BITS = 19
CLIENT_BITS = 10
PARAM_BOUND = 1 << PARAM_BITS
WEIGHT_BOUND = 1 << WEIGHT_BITS
CLIENT_BOUND = 1 << CLIENT_BITS
SUM_BOUND = 1 << (WEIGHT_BITS + CLIENT_BITS)
DEFAULT_F = 20
MAX_RANK = 8
MAX_TENSOR_BYTES = (1 << 32) - 1
MASK64 = (1 << 64) - 1


class SpecError(Exception):
    """Raised where the spec says 'reject' / 'revert'."""


# --- section 3.2: the single rounding point -------------------------------
def div_round_half_even(num: int, den: int) -> int:
    """Divide num by den, nearest, ties to even.

    Deliberately written without // or %: Python floors and Go truncates, and
    the two disagree on negative operands. The spec fixes the behaviour
    explicitly rather than inheriting a language default, so the reference must
    not inherit one either.
    """
    if den <= 0:
        raise SpecError("divisor must be positive")
    neg = num < 0
    a = -num if neg else num
    q, r = divmod(a, den)  # a >= 0, so floor == truncate here
    if 2 * r > den or (2 * r == den and q & 1):
        q += 1
    return -q if neg else q


# --- section 4.2: canonical encoding --------------------------------------
def numel(dims):
    if not dims or len(dims) > MAX_RANK:
        raise SpecError("bad rank")
    max_elements = (MAX_TENSOR_BYTES - (10 + 4 * len(dims))) // 8
    n = 1
    for d in dims:
        if d <= 0 or d >= 1 << 32:
            raise SpecError("bad dimension")
        if n > max_elements // d:
            raise SpecError("tensor exceeds chain ABI length bound")
        n *= d
    return n


def validate(dims, f, data):
    if numel(dims) != len(data):
        raise SpecError("length disagrees with dims")
    if not 0 <= f < 256:
        raise SpecError("bad F")
    for v in data:
        if v <= -PARAM_BOUND or v >= PARAM_BOUND:
            raise SpecError("parameter out of range")


def encode(dims, f, data) -> bytes:
    validate(dims, f, data)
    out = bytearray()
    out.append(len(dims))
    for d in dims:
        out += struct.pack(">I", d)
    out.append(f)
    out += struct.pack(">Q", len(data))
    for v in data:
        out += struct.pack(">q", v)
    return bytes(out)


def decode(b: bytes):
    if len(b) < 1:
        raise SpecError("empty")
    rank = b[0]
    if rank < 1 or rank > MAX_RANK:
        raise SpecError("bad rank")
    if len(b) < 1 + 4 * rank + 1 + 8:
        raise SpecError("truncated header")
    off = 1
    dims = []
    for _ in range(rank):
        dims.append(struct.unpack_from(">I", b, off)[0])
        off += 4
    f = b[off]
    off += 1
    n = struct.unpack_from(">Q", b, off)[0]
    off += 8
    if n != numel(dims):
        raise SpecError("length disagrees with dims")
    if len(b) - off != 8 * n:
        raise SpecError("body length mismatch (truncated or trailing bytes)")
    data = [struct.unpack_from(">q", b, off + 8 * i)[0] for i in range(n)]
    validate(dims, f, data)
    return dims, f, data


def state_root(dims, f, data) -> str:
    """SHA3-256 (FIPS-202) over the canonical encoding.

    Matches Go's crypto/sha3. Ethereum's trie uses Keccak-256, which differs in
    padding; the distinction is immaterial to the argument but must be stated
    accurately in the paper rather than blurred into "keccak".
    """
    return hashlib.sha3_256(encode(dims, f, data)).hexdigest()


# --- section 2.2 / 3.1: aggregation ---------------------------------------
def check_updates(weights, tensors):
    n = len(weights)
    if n < 1 or n >= CLIENT_BOUND:
        raise SpecError("client count out of range")
    if len(tensors) != n:
        raise SpecError("weights/tensors length mismatch")
    dims0, f0, _ = tensors[0]
    total = 0
    for w, (dims, f, data) in zip(weights, tensors):
        if not 1 <= w < WEIGHT_BOUND:
            raise SpecError("weight out of range")
        validate(dims, f, data)
        if list(dims) != list(dims0) or f != f0:
            raise SpecError("shape or F mismatch")
        total += w
    if not 1 <= total < SUM_BOUND:
        raise SpecError("weight sum out of range")
    return total


def fedavg(weights, tensors):
    """Theta[j] = sum_n(s_n * theta_n[j]) / S, exact integers, one rounding."""
    S = check_updates(weights, tensors)
    dims, f, _ = tensors[0]
    p = len(tensors[0][2])
    acc = [0] * p
    for w, (_, _, data) in zip(weights, tensors):
        for j in range(p):
            acc[j] += w * data[j]
    out = [div_round_half_even(acc[j], S) for j in range(p)]
    validate(dims, f, out)
    return list(dims), f, out


def trimmed_mean(_addrs, weights, tensors, k):
    """Coordinate-wise unweighted trimmed mean over integer values."""
    check_updates(weights, tensors)
    n = len(tensors)
    if k < 0 or 2 * k >= n:
        raise SpecError("invalid trim count")
    dims, f, _ = tensors[0]
    p = len(tensors[0][2])
    keep = n - 2 * k
    out = [0] * p
    for j in range(p):
        col = sorted(tensors[i][2][j] for i in range(n))
        out[j] = div_round_half_even(sum(col[k:n - k]), keep)
    validate(dims, f, out)
    return list(dims), f, out


# --- section 2.1: client-side quantization (mirror of quant.go) -----------
# This is the one part of the reference that sits OUTSIDE the consensus
# boundary. Float appears here by necessity; the boundary is drawn immediately
# after. Whatever the client produces IS the definition of the input, so a
# client-side rounding difference is a different legal submission, not a fault.

def quantize(theta, f: int = DEFAULT_F):
    """theta_hat = round_half_even(theta * 2^f), rejecting out-of-range values.

    Accepts any iterable of floats (including a numpy array). Out-of-range
    values raise rather than clamp: silently clamping would aggregate a model
    the client never sent.
    """
    scale = float(1 << f)
    out = []
    for v in theta:
        x = float(v) * scale
        if x != x or x in (float("inf"), float("-inf")):
            raise SpecError("non-finite parameter")
        # round-half-even without relying on any library's rounding mode
        fl = int(x // 1)
        frac = x - fl
        if frac > 0.5 or (frac == 0.5 and fl & 1):
            q = fl + 1
        else:
            q = fl
        if q <= -PARAM_BOUND or q >= PARAM_BOUND:
            raise SpecError("parameter out of range: %r" % (v,))
        out.append(q)
    return out


def dequantize(data, f: int = DEFAULT_F):
    scale = 2.0 ** -f
    return [v * scale for v in data]


def max_abs_float(f: int = DEFAULT_F) -> float:
    """Exclusive bound on |theta| representable at f fractional bits."""
    return float(1 << (PARAM_BITS - f))


def quant_error_bound(f: int = DEFAULT_F) -> float:
    return 2.0 ** -(f + 1)


# --- reproducible input generation (mirror of gen.go) ---------------------
class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


def gen_quantized(seed: int, p: int, bound: int):
    r = SplitMix64(seed)
    span = 2 * bound - 1
    return [r.next() % span - (bound - 1) for _ in range(p)]


def gen_addr(i: int) -> bytes:
    a = bytearray(20)
    a[19] = (i + 1) & 0xFF
    a[18] = ((i + 1) >> 8) & 0xFF
    return bytes(a)


def gen_updates(seed: int, n: int, p: int, weights, bound: int, f: int, dims=None):
    addrs, ws, tensors = [], [], []
    for i in range(n):
        data = gen_quantized(seed + i * 0x1000, p, bound)
        addrs.append(gen_addr(i))
        ws.append(weights[i % len(weights)])
        tensors.append((list(dims) if dims else [p], f, data))
    return addrs, ws, tensors


def gen_float32(seed: int, p: int):
    """float32 values in [-2^-4, 2^-4), every one exactly representable.

    Returns Python floats holding exact float32 values; callers that need a
    numpy array should cast with dtype=np.float32, which is lossless here.
    """
    r = SplitMix64(seed)
    return [(float(r.next() >> 40) - 8388608.0) * (2.0 ** -27) for _ in range(p)]
