"""Bridge from the workload to dfrl-agg.

The training loop calls the SAME code the chain will call. That is the whole
argument behind U6 (on-chain equals off-chain) and X6 (validators agree byte for
byte): they are properties of the build, not results to be reconciled later. So
aggregation here goes through the Go module's CLI by default, and the pure
Python reference in dfrl-agg/pyref is used as an independent cross-check rather
than as the primary path.

If the Go toolchain is unavailable the bridge falls back to the Python
reference and says so, because a run that silently changed its aggregator would
be uncomparable to the others.
"""

import os
import subprocess
import sys
import tempfile

AGG_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dfrl-agg"))
sys.path.insert(0, os.path.join(AGG_ROOT, "pyref"))

import dfrl_ref as R  # noqa: E402

DEFAULT_F = R.DEFAULT_F


class BridgeError(RuntimeError):
    pass


class Aggregator:
    """Integer aggregation with an optional per-round cross-check.

    backend: "go" (default), "python", or "both".
    crosscheck_every: with backend="go", also run the Python reference every N
        rounds and require byte equality. 0 disables. This is cheap for the
        7,850-parameter model and worth keeping on: it turns the cross-language
        claim from a one-off vector check into a continuous invariant.
    """

    def __init__(self, backend="go", crosscheck_every=0, scratch=None):
        self.backend = backend
        self.crosscheck_every = int(crosscheck_every)
        self._binary = None
        self._scratch = scratch or tempfile.mkdtemp(prefix="dfrl-agg-")
        self.backend_detail = backend
        if backend in ("go", "both"):
            try:
                self._binary = self._build()
            except BridgeError as exc:
                if backend == "both":
                    raise
                self.backend = "python"
                self.backend_detail = "python (go unavailable: %s)" % exc

    def _build(self):
        out = os.path.join(self._scratch, "dfrl-agg" + (".exe" if os.name == "nt" else ""))
        try:
            proc = subprocess.run(
                ["go", "build", "-o", out, "./cmd/dfrl-agg"],
                cwd=AGG_ROOT, capture_output=True, text=True, timeout=300)
        except OSError as exc:
            raise BridgeError(str(exc))
        if proc.returncode != 0:
            raise BridgeError(proc.stderr.strip()[:200])
        return out

    # --- quantization (client side, off the consensus path) ---------------
    @staticmethod
    def quantize(vec, f=DEFAULT_F):
        """float32 vector -> int64 list, rejecting out-of-range values.

        A rejection here is real information, not an inconvenience: it means the
        model left the representable range and the run's F or learning rate is
        wrong. Never clamp.
        """
        return R.quantize(vec, f)

    @staticmethod
    def dequantize(data, f=DEFAULT_F):
        import numpy as np
        return np.asarray(R.dequantize(data, f), dtype=np.float32)

    # --- aggregation (consensus path) -------------------------------------
    def fedavg(self, weights, tensors, dims, f=DEFAULT_F, round_index=0):
        """Weighted average. Returns (data, root_hex, detail).

        weights: integer s_n per client, already validated against the spec range
        tensors: list of int64 lists, one per client
        """
        if self.backend == "python":
            return self._fedavg_python(weights, tensors, dims, f)

        data, root, _ = self._fedavg_go(weights, tensors, dims, f)
        detail = "go"
        want_check = self.backend == "both" or (
            self.crosscheck_every and round_index % self.crosscheck_every == 0)
        if want_check:
            pdata, proot, _ = self._fedavg_python(weights, tensors, dims, f)
            if proot != root or pdata != data:
                raise BridgeError(
                    "cross-language mismatch at round %d: go=%s python=%s"
                    % (round_index, root[:16], proot[:16]))
            detail = "go+python-verified"
        return data, root, detail

    def _fedavg_python(self, weights, tensors, dims, f):
        tlist = [(list(dims), f, t) for t in tensors]
        d, ff, out = R.fedavg(list(weights), tlist)
        return out, R.state_root(d, ff, out), "python"

    def _fedavg_go(self, weights, tensors, dims, f):
        paths = []
        for i, t in enumerate(tensors):
            p = os.path.join(self._scratch, "in%03d.bin" % i)
            with open(p, "wb") as fh:
                fh.write(R.encode(list(dims), f, t))
            paths.append(p)
        outp = os.path.join(self._scratch, "out.bin")
        cmd = [self._binary, "fedavg", "-w", ",".join(str(int(w)) for w in weights),
               "-o", outp] + paths
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise BridgeError("dfrl-agg fedavg failed: " + proc.stderr.strip()[:300])
        root = proc.stdout.strip()
        with open(outp, "rb") as fh:
            _, _, data = R.decode(fh.read())
        return data, root, "go"

    # --- robust aggregation (Phase 4 / U3) --------------------------------
    def trimmed_mean(self, tensors, dims, k, f=DEFAULT_F, round_index=0):
        """Coordinate-wise trimmed mean, k smallest/largest discarded per
        coordinate. Weights are not accepted -- dfrl-agg/trimmed.go: weighting
        would reintroduce the influence trimming exists to bound. Returns the
        same (data, root_hex, detail) shape fedavg() does.
        """
        if self.backend == "python":
            return self._trimmed_python(tensors, dims, k, f)

        data, root, _ = self._trimmed_go(tensors, dims, k, f)
        detail = "go"
        want_check = self.backend == "both" or (
            self.crosscheck_every and round_index % self.crosscheck_every == 0)
        if want_check:
            pdata, proot, _ = self._trimmed_python(tensors, dims, k, f)
            if proot != root or pdata != data:
                raise BridgeError(
                    "cross-language trimmed-mean mismatch at round %d: go=%s python=%s"
                    % (round_index, root[:16], proot[:16]))
            detail = "go+python-verified"
        return data, root, detail

    @staticmethod
    def _synth_addr(i):
        # Mirrors cmd/dfrl-agg/main.go's synthetic identities. TrimmedMean is
        # numerically unweighted; identities are retained only for the shared
        # update-validation interface.
        a = bytearray(20)
        a[19] = (i + 1) & 0xFF
        a[18] = ((i + 1) >> 8) & 0xFF
        return bytes(a)

    def _trimmed_python(self, tensors, dims, k, f):
        addrs = [self._synth_addr(i) for i in range(len(tensors))]
        weights = [1] * len(tensors)
        tlist = [(list(dims), f, t) for t in tensors]
        d, ff, out = R.trimmed_mean(addrs, weights, tlist, k)
        return out, R.state_root(d, ff, out), "python"

    def _trimmed_go(self, tensors, dims, k, f):
        paths = []
        for i, t in enumerate(tensors):
            p = os.path.join(self._scratch, "tin%03d.bin" % i)
            with open(p, "wb") as fh:
                fh.write(R.encode(list(dims), f, t))
            paths.append(p)
        outp = os.path.join(self._scratch, "tout.bin")
        cmd = [self._binary, "trimmed", "-k", str(int(k)), "-o", outp] + paths
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise BridgeError("dfrl-agg trimmed failed: " + proc.stderr.strip()[:300])
        root = proc.stdout.strip()
        with open(outp, "rb") as fh:
            _, _, data = R.decode(fh.read())
        return data, root, "go"

    def state_root(self, data, dims, f=DEFAULT_F):
        return R.state_root(list(dims), f, data)

    def keep_submissions(self, dest, round_index, weights, tensors, dims, f=DEFAULT_F):
        """Persist one round's submissions in canonical form.

        This is the raw material for the replay audit (C3 / X7): a third party
        recomputes the aggregate from these bytes and compares. Off by default
        only because of disk cost, never because it is optional to the argument.
        """
        d = os.path.join(dest, "round%04d" % round_index)
        os.makedirs(d, exist_ok=True)
        for i, (w, t) in enumerate(zip(weights, tensors)):
            with open(os.path.join(d, "client%03d_w%d.bin" % (i, int(w))), "wb") as fh:
                fh.write(R.encode(list(dims), f, t))
        return d
