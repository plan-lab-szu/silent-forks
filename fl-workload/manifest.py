"""Run manifests: every run records what produced it.

R1-5b asks for >=10 seeds with mean +/- 95% CI and paired tests. That is
impossible to assemble after the fact unless every run drops a manifest at the
moment it happens, so this module exists before any experiment does. All figures
must be generated from files on disk, never from something still in memory.

Deliberately not recorded: wall-clock time as an identifier. Run identity comes
from the configuration, so re-running the same configuration overwrites rather
than accumulating near-duplicates.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys

RUNS_DIR = os.environ.get("DFRL_RUNS", "runs")


def git_commit(path="."):
    """Best-effort commit id. Returns None when the tree is not in git yet.

    The repository is deliberately not under git at the time of writing, so this
    must degrade rather than fail; a run without a commit id is still valid, it
    just carries a weaker provenance claim and says so.
    """
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", path, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def library_versions():
    """Versions of everything that could plausibly change a float result.

    This is not bookkeeping for its own sake: the C1 result is that a library
    version alone can change the aggregate. A run that does not record its
    library versions cannot be compared to another run.
    """
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "torch", "scipy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def run_id(config: dict) -> str:
    """Stable identifier: a short hash of the configuration."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class Manifest:
    def __init__(self, config: dict, timestamp: str | None = None, root=RUNS_DIR):
        self.config = dict(config)
        self.id = run_id(self.config)
        self.dir = os.path.join(root, self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.data = {
            "run_id": self.id,
            "config": self.config,
            "git_commit": git_commit(os.path.join(os.path.dirname(__file__), "..")),
            "libraries": library_versions(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "timestamp": timestamp,
            "results": {},
        }

    def note(self, key, value):
        self.data["results"][key] = value

    def path(self, *parts):
        p = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def save(self):
        with open(os.path.join(self.dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return self.dir
