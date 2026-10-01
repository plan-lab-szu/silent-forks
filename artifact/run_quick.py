"""Portable quick check for the Silent Forks artifact.

Run from any directory:

    python artifact/run_quick.py [--with-geth]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(
    label: str,
    command: list[str],
    cwd: Path,
    *,
    summary_tail: int | None = None,
) -> None:
    print(f"\n== {label} ==", flush=True)
    print(f"$ {' '.join(command)}", flush=True)
    if summary_tail is None:
        subprocess.run(command, cwd=cwd, check=True)
        return

    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    lines = completed.stdout.rstrip().splitlines()
    if completed.returncode != 0:
        print(completed.stdout, end="")
        completed.check_returncode()
    for line in lines[-summary_tail:]:
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run portable artifact checks")
    parser.add_argument(
        "--with-geth",
        action="store_true",
        help="also run integration tests in the companion chain/geth checkout",
    )
    args = parser.parse_args()

    agg = ROOT / "dfrl-agg"
    run("Go unit and property tests", ["go", "test", "./..."], agg)
    run(
        "Generate fixed-seed randomized conformance cases",
        ["go", "test", "-run", "TestGenerateDifferentialVectors", "./..."],
        agg,
    )
    run(
        "Curated Go/Python conformance",
        [sys.executable, "pyref/check_vectors.py"],
        agg,
    )
    run(
        "Randomized Go/Python conformance",
        [sys.executable, "pyref/check_vectors.py", "vectors/differential.json"],
        agg,
        summary_tail=4,
    )

    if args.with_geth:
        geth = ROOT / "chain" / "geth"
        if not (geth / "go.mod").is_file():
            raise SystemExit(
                "chain/geth is missing; clone the companion dfrl-consensus branch "
                "as described in ARTIFACTS.md"
            )
        run(
            "Mandatory height-scheduled state transition",
            [
                "go",
                "test",
                "./core",
                "-run",
                "TestDFRLAggregation(IsMandatory|ReachesStateRoot)$",
            ],
            geth,
        )
        run(
            "EVM aggregation entry points and contract integration",
            ["go", "test", "./core/vm", "-run", "DFRL|X4RealMNIST"],
            geth,
        )
        run(
            "Engine API rejection of a divergent post-state root",
            [
                "go",
                "test",
                "./eth/catalyst",
                "-run",
                "TestDFRLDivergentStateRootIsINVALID$",
            ],
            geth,
        )

    print("\nAll requested artifact checks passed.", flush=True)


if __name__ == "__main__":
    main()
