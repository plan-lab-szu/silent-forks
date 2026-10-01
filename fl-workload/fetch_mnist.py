"""Fetch MNIST into data/mnist/, verifying what arrived.

  python fetch_mnist.py [--dest data/mnist]

Kept in the repository rather than done by hand: reproducing the dataset
acquisition is part of the artifact release (R1-Qe), and a dataset that arrived
by an unrecorded route weakens every number computed from it.

Source is the OSSCI (PyTorch) S3 mirror. Files stay gzipped -- data.py reads
.gz directly -- so the on-disk cost is ~11.6 MB rather than ~55 MB.

Three independent checks, because a corrupt or substituted dataset would be
invisible in an accuracy curve:
  1. byte length against the published values
  2. MD5 against the published values (reported as a comparison, not assumed)
  3. structure: idx magic number consistent with the rank, declared count,
     body length, and value range (0..255 for pixels, 0..9 for labels)
The SHA-256 of whatever actually arrived is recorded in SOURCE.json either way,
so a later run can prove it used the same bytes.
"""

import argparse
import gzip
import hashlib
import json
import os
import struct

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"

# Published values for the canonical MNIST distribution. Treated as a claim to
# be checked against the bytes, not as ground truth about them.
FILES = [
    ("train-images-idx3-ubyte.gz", 9912422, "f68b3c2dcbeaaa9fbdd348bbdeb94873", 60000),
    ("train-labels-idx1-ubyte.gz", 28881, "d53e105ee54ea40749a09fcbcd1e9432", 60000),
    ("t10k-images-idx3-ubyte.gz", 1648877, "9fb629c4189551a2d022fa330f9573f3", 10000),
    ("t10k-labels-idx1-ubyte.gz", 4542, "ec29112dd5afa0611ce80d1b7f02629c", 10000),
]

IDX_MAGIC = {1: 2049, 3: 2051}  # rank -> expected magic for uint8 idx files


def download(name, dest):
    import requests
    path = os.path.join(dest, name)
    if os.path.exists(path):
        return path, "cached"
    resp = requests.get(BASE + name, timeout=120, stream=True)
    resp.raise_for_status()
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        for chunk in resp.iter_content(1 << 16):
            fh.write(chunk)
    os.replace(tmp, path)
    return path, "downloaded"


def digest(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def inspect(path):
    with gzip.open(path, "rb") as fh:
        magic, = struct.unpack(">I", fh.read(4))
        rank = magic & 0xFF
        dims = struct.unpack(">" + "I" * rank, fh.read(4 * rank))
        body = fh.read()
    return magic, rank, dims, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "mnist"))
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    record = {"source": BASE, "files": {}}
    problems = []
    print("fetching into %s" % args.dest)
    for name, want_size, want_md5, want_n in FILES:
        path, how = download(name, args.dest)
        size = os.path.getsize(path)
        md5, sha = digest(path)
        magic, rank, dims, body = inspect(path)

        elems = 1
        for d in dims:
            elems *= d
        vmax = max(body)
        checks = {
            "size": size == want_size,
            "md5": md5 == want_md5,
            "magic": IDX_MAGIC.get(rank) == magic,
            "count": bool(dims) and dims[0] == want_n,
            "body_len": len(body) == elems,
            "value_range": vmax <= (255 if rank == 3 else 9),
        }
        bad = [k for k, ok in checks.items() if not ok]
        print("  %-28s %-11s %9d B  magic %d dims %s  max %d  %s"
              % (name, how, size, magic, dims, vmax,
                 "ok" if not bad else "FAILED: " + ",".join(bad)))
        if bad:
            problems.append(name)
        record["files"][name] = {
            "size": size, "md5": md5, "sha256": sha,
            "published_md5": want_md5, "md5_matches_published": checks["md5"],
            "idx_magic": magic, "idx_dims": list(dims), "max_value": vmax,
        }

    with open(os.path.join(args.dest, "SOURCE.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")

    if problems:
        print("\nPROBLEMS with: %s -- do not use this data" % ", ".join(problems))
        return 1
    print("\nall four files verified: size + published md5 + idx structure")
    print("wrote %s" % os.path.join(args.dest, "SOURCE.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
