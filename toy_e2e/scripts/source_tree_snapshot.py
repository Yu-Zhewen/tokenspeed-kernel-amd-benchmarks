#!/usr/bin/env python3
"""Create a reproducible fingerprint for a dirty Git worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(("git", "-C", str(root), *args))


def snapshot(root: Path) -> dict[str, object]:
    root = root.resolve()
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).decode().splitlines()
    paths = _git(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ).split(b"\0")

    digest = hashlib.sha256()
    file_count = 0
    for encoded_path in sorted(path for path in paths if path):
        relative = os.fsdecode(encoded_path)
        path = root / relative
        digest.update(encoded_path)
        digest.update(b"\0")
        if path.is_symlink():
            content = os.fsencode(os.readlink(path))
            kind = b"symlink"
        else:
            content = path.read_bytes()
            kind = b"file"
        digest.update(kind)
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        file_count += 1

    return {
        "format": "tokenspeed_source_tree_snapshot_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "revision": revision,
        "dirty": bool(status),
        "status": status,
        "file_count": file_count,
        "worktree_sha256": digest.hexdigest(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--field",
        choices=("revision", "dirty", "worktree_sha256"),
        help="print only one field",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = snapshot(args.root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.field is not None:
        value = result[args.field]
        print(str(value).lower() if isinstance(value, bool) else value)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
