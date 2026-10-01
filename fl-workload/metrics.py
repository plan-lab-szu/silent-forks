"""Evaluation and per-round telemetry.

Two things are recorded every round, both from the first day rather than added
later:

  1. Learning quality (test accuracy and loss), so the plain-FL workload can be
     compared against the centralized arm -- that is U6/X4's claim that the
     chain buys verifiability, not accuracy.
  2. The state root of the aggregate, so divergence is a byte-level fact per
     round rather than something inferred from a learning curve. A learning
     curve cannot reveal a fork; only the root can.
"""

import json

import numpy as np


class RoundLog:
    """Append-only JSONL. Figures are generated from this file, never from memory."""

    def __init__(self, path):
        self.path = path
        self.rows = []
        self._fh = open(path, "w", encoding="utf-8")

    def add(self, **row):
        self.rows.append(row)
        self._fh.write(json.dumps(row, sort_keys=True) + "\n")
        self._fh.flush()
        return row

    def close(self):
        self._fh.close()


def quant_error(original, restored):
    """Absolute error introduced by the quantize/aggregate/dequantize path."""
    d = np.abs(np.asarray(original, dtype=np.float64) - np.asarray(restored, dtype=np.float64))
    return {"max": float(d.max()), "mean": float(d.mean()), "rms": float(np.sqrt((d * d).mean()))}


def heterogeneity(label_hist):
    """A measured proxy for client heterogeneity, so R2-8 is answered with a
    number rather than an assumption. Total variation distance between each
    client's label distribution and the pooled one, averaged.
    """
    h = np.asarray(label_hist, dtype=np.float64)
    per_client = h / np.maximum(h.sum(axis=1, keepdims=True), 1.0)
    pooled = h.sum(axis=0) / max(h.sum(), 1.0)
    tv = 0.5 * np.abs(per_client - pooled).sum(axis=1)
    return {"tv_mean": float(tv.mean()), "tv_max": float(tv.max())}


def summarize(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    return {"final": float(a[-1]), "best": float(a.max()), "mean": float(a.mean())}
