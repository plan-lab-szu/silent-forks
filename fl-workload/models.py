"""Models, in numpy, float32 throughout.

No torch, deliberately. Two reasons:

  1. The float pipeline (floatpipe.py) needs explicit control over how sums are
     associated. A framework that chooses its own kernels, thread counts and
     fusion would add exactly the uncontrolled variables the C1 experiment is
     about -- as a study subject that is the point, but not inside the model.
  2. A 784->10 classifier needs no framework. Manual backprop is shorter than
     the code that would pin a framework's nondeterminism.

torch is installed and remains available if a convolutional workload is ever
wanted; nothing here depends on that choice.

Parameters are exposed as one flat float32 vector, because that is what gets
quantized, canonically encoded and aggregated. The flat layout is part of the
protocol surface: two clients must agree on it or their submissions are not
comparable, so it is fixed by construction order and never inferred.
"""

import numpy as np

from data import N_CLASSES, N_FEATURES


def softmax_xent(logits, y):
    """Loss and dlogits for softmax cross-entropy, stable and float32."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z, dtype=np.float32)
    p = e / e.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = float(-np.log(np.maximum(p[np.arange(n), y], np.float32(1e-12))).mean())
    d = p.copy()
    d[np.arange(n), y] -= np.float32(1.0)
    return loss, (d / np.float32(n)).astype(np.float32)


class Model:
    """Common interface. Shapes fixes the flat layout."""

    shapes: list

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.params = []
        for shape, scale in self._init_spec():
            if scale == 0.0:
                self.params.append(np.zeros(shape, dtype=np.float32))
            else:
                self.params.append(
                    (rng.standard_normal(shape) * scale).astype(np.float32))
        self.shapes = [p.shape for p in self.params]

    @property
    def n_params(self):
        return int(sum(int(np.prod(s)) for s in self.shapes))

    def flat(self):
        return np.concatenate([p.ravel() for p in self.params]).astype(np.float32)

    def load_flat(self, v):
        v = np.asarray(v, dtype=np.float32)
        if v.size != self.n_params:
            raise ValueError("expected %d parameters, got %d" % (self.n_params, v.size))
        off = 0
        for i, shape in enumerate(self.shapes):
            n = int(np.prod(shape))
            self.params[i] = v[off:off + n].reshape(shape).astype(np.float32)
            off += n

    def flat_grad(self, grads):
        return np.concatenate([g.ravel() for g in grads]).astype(np.float32)

    def accuracy(self, X, y):
        return float((self.forward(X).argmax(axis=1) == y).mean())

    def loss(self, X, y):
        return softmax_xent(self.forward(X), y)[0]


class LogReg(Model):
    """Multinomial logistic regression: 784*10 + 10 = 7,850 parameters.

    The small end of the |Theta| range. At 7,850 int64 parameters the canonical
    encoding is ~63 kB, which is the size that makes the calldata and block gas
    limit discussion concrete (R1-6c, U4).
    """

    def _init_spec(self):
        return [((N_FEATURES, N_CLASSES), 0.01), ((N_CLASSES,), 0.0)]

    def forward(self, X):
        W, b = self.params
        return (X @ W + b).astype(np.float32)

    def loss_and_grad(self, X, y):
        W, b = self.params
        loss, d = softmax_xent(self.forward(X), y)
        return loss, self.flat_grad([(X.T @ d).astype(np.float32), d.sum(axis=0)])


class MLP(Model):
    """One hidden layer, tanh. Parameter count = 795*hidden + 10.

    The width is the point: |Theta| becomes a free variable, which is what the
    X8 gas surface needs. In the DCA workload the parameter count was pinned by
    the network architecture, so it could not be swept independently.
    """

    def __init__(self, hidden=314, seed=0):
        self.hidden = int(hidden)
        super().__init__(seed=seed)

    def _init_spec(self):
        h = self.hidden
        return [((N_FEATURES, h), (1.0 / N_FEATURES) ** 0.5), ((h,), 0.0),
                ((h, N_CLASSES), (1.0 / h) ** 0.5), ((N_CLASSES,), 0.0)]

    def forward(self, X):
        W1, b1, W2, b2 = self.params
        return (np.tanh(X @ W1 + b1, dtype=np.float32) @ W2 + b2).astype(np.float32)

    def loss_and_grad(self, X, y):
        W1, b1, W2, b2 = self.params
        h = np.tanh(X @ W1 + b1, dtype=np.float32)
        loss, d = softmax_xent((h @ W2 + b2).astype(np.float32), y)
        dh = ((d @ W2.T) * (np.float32(1.0) - h * h)).astype(np.float32)
        return loss, self.flat_grad([
            (X.T @ dh).astype(np.float32), dh.sum(axis=0),
            (h.T @ d).astype(np.float32), d.sum(axis=0)])


def params_for_width(hidden):
    return 795 * int(hidden) + 10


def width_for_params(target):
    """Largest hidden width whose parameter count does not exceed target.

    Used by the X8 sweep: pick a target |Theta|, get a model that hits it.
    """
    return max(1, (int(target) - 10) // 795)


def build(kind, hidden=314, seed=0):
    if kind == "logreg":
        return LogReg(seed=seed)
    if kind == "mlp":
        return MLP(hidden=hidden, seed=seed)
    raise ValueError("unknown model " + kind)
