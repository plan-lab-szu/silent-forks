"""Client-side local training.

The client is the last place floating point is allowed to matter (spec section
1). It trains in float32, quantizes, and submits integers. Nondeterminism in
this stage is harmless -- whatever the client produces IS the definition of the
input -- so the seeding here is for reproducibility of the experiment, not for
consensus.
"""

import numpy as np


class Client:
    def __init__(self, cid, X, y, model, lr=0.5, batch=64, local_epochs=1):
        self.cid = cid
        self.X = X
        self.y = y
        self.model = model
        self.lr = np.float32(lr)
        self.batch = int(batch)
        self.local_epochs = int(local_epochs)

    @property
    def n_samples(self):
        """The weight s_n. Section 4.3 requires it to be an on-chain integer, so
        it must be something the chain can see and check -- a sample count is,
        a measured wall-clock time or a float loss is not.
        """
        return int(len(self.y))

    def train(self, global_flat, seed):
        """One local round starting from the global model. Returns flat float32."""
        self.model.load_flat(global_flat)
        rng = np.random.default_rng(seed)
        n = len(self.y)
        last_loss = float("nan")
        for _ in range(self.local_epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch):
                idx = order[start:start + self.batch]
                if idx.size == 0:
                    continue
                last_loss, grad = self.model.loss_and_grad(self.X[idx], self.y[idx])
                self.model.load_flat(self.model.flat() - self.lr * grad)
        return self.model.flat(), last_loss
