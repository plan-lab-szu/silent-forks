"""Datasets and the non-IID client split.

MNIST is the intended workload (D8), but no local copy exists yet and
downloading is a separate decision. So the loader is pluggable and ships with a
deterministic synthetic dataset of the same shape (784 features, 10 classes),
which lets the whole pipeline run and be tested today. Which source was used is
recorded in the manifest -- a synthetic-data result must never be mistaken for
an MNIST result.
"""

import gzip
import os
import struct

import numpy as np

N_FEATURES = 784
N_CLASSES = 10

# Where fetch_mnist.py puts the data. Overridable, but the default means
# --data mnist works with no further arguments once the files are fetched.
DEFAULT_MNIST_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mnist")


def load_synthetic(n_train=12000, n_test=2000, seed=0):
    """A linearly separable 784->10 problem with label noise.

    Not a stand-in for MNIST in any scientific sense; it exists so that
    convergence, quantization and aggregation can be exercised end to end. The
    signal is deliberately easy, so accuracy well above chance is a smoke test,
    not a result.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N_FEATURES, N_CLASSES)).astype(np.float32) * 0.1

    def make(n):
        X = rng.standard_normal((n, N_FEATURES)).astype(np.float32) * 0.3
        logits = X @ W
        y = np.argmax(logits, axis=1).astype(np.int64)
        flip = rng.random(n) < 0.05  # label noise, so 100% accuracy is unreachable
        y[flip] = rng.integers(0, N_CLASSES, size=int(flip.sum()))
        return X, y

    Xtr, ytr = make(n_train)
    Xte, yte = make(n_test)
    return Xtr, ytr, Xte, yte


def _read_idx(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        magic, = struct.unpack(">I", fh.read(4))
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, fh.read(4 * ndim))
        buf = fh.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(dims)


def load_mnist(root):
    """Load MNIST from idx files (raw or .gz) in `root`.

    Expects the canonical names train-images-idx3-ubyte, train-labels-idx1-ubyte,
    t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte, optionally .gz suffixed.
    Raises FileNotFoundError if the data is not there -- it never downloads
    anything on its own.
    """
    names = {
        "Xtr": "train-images-idx3-ubyte",
        "ytr": "train-labels-idx1-ubyte",
        "Xte": "t10k-images-idx3-ubyte",
        "yte": "t10k-labels-idx1-ubyte",
    }
    found = {}
    for key, base in names.items():
        for cand in (base, base + ".gz"):
            p = os.path.join(root, cand)
            if os.path.exists(p):
                found[key] = p
                break
        else:
            raise FileNotFoundError("missing %s (or .gz) under %s" % (base, root))
    Xtr = _read_idx(found["Xtr"]).reshape(-1, N_FEATURES).astype(np.float32) / 255.0
    Xte = _read_idx(found["Xte"]).reshape(-1, N_FEATURES).astype(np.float32) / 255.0
    return Xtr, _read_idx(found["ytr"]).astype(np.int64), Xte, _read_idx(found["yte"]).astype(np.int64)


def load(source="synthetic", root=None, seed=0):
    if source == "mnist":
        root = root or DEFAULT_MNIST_ROOT
        if not os.path.isdir(root):
            raise FileNotFoundError(
                "%s does not exist -- run: python fetch_mnist.py" % root)
        return load_mnist(root)
    if source == "synthetic":
        return load_synthetic(seed=seed)
    raise ValueError("unknown data source " + source)


def dirichlet_split(labels, n_clients, alpha, seed):
    """Label-skewed non-IID partition: each class is spread over clients by a
    Dirichlet(alpha) draw. Small alpha means highly skewed shards.

    This is what answers R2-8 (is bounded heterogeneity a reasonable
    assumption?) without any extra machinery: alpha is a dial, and the resulting
    heterogeneity can be measured directly rather than assumed.

    Returns a list of index arrays, one per client. Every client is guaranteed
    at least one sample, since a weight of zero is not a legal submission
    (DETERMINISM_SPEC section 2.2 requires s_n >= 1).
    """
    rng = np.random.default_rng(seed)
    shards = [[] for _ in range(n_clients)]
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for c, part in enumerate(np.split(idx, cuts)):
            shards[c].extend(part.tolist())
    out = []
    for c in range(n_clients):
        arr = np.array(sorted(shards[c]), dtype=np.int64)
        if arr.size == 0:
            raise ValueError(
                "client %d received no samples (alpha=%g too small for %d clients)"
                % (c, alpha, n_clients))
        out.append(arr)
    return out


def label_distribution(labels, shards):
    """Per-client class histogram, so heterogeneity is reported, not claimed."""
    return [np.bincount(labels[s], minlength=N_CLASSES).tolist() for s in shards]
