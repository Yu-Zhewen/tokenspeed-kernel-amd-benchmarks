# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Download and stream-repack a rank-local, reduced Kimi-K3 checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_e2e.scripts.analyze_checkpoint import (
    DEFAULT_REPO_ID,
    _fetch_json,
    _fetch_safetensors_header,
    _headers,
    _resolve_revision,
    _resolve_url,
    select_required_keys,
)

_COPY_BYTES = 16 << 20
_REPORT_BYTES = 1 << 30
_METADATA_FILES = (
    "LICENSE",
    "README.md",
    "config.json",
    "configuration_kimi_k3.py",
    "encoding_k3.py",
    "generation_config.json",
    "kimi_k3_processor.py",
    "kimi_k3_vision_processing.py",
    "media_utils.py",
    "modeling_kimi_k3.py",
    "modeling_kimi_linear.py",
    "preprocessor_config.json",
    "tiktoken.model",
    "tokenization_kimi.py",
    "tokenizer_config.json",
)


def _read_local_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as source:
        encoded_length = source.read(8)
        if len(encoded_length) != 8:
            raise RuntimeError(f"invalid safetensors prefix in {path}")
        header_length = int.from_bytes(encoded_length, "little")
        encoded = source.read(header_length)
    if len(encoded) != header_length:
        raise RuntimeError(f"short safetensors header in {path}")
    return json.loads(encoded), 8 + header_length


def _encode_header(header: dict[str, Any]) -> bytes:
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    padding = (-len(encoded)) % 8
    return encoded + b" " * padding


def _copy_range(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    source_offset: int,
    size: int,
) -> None:
    source.seek(source_offset)
    remaining = size
    while remaining:
        chunk = source.read(min(_COPY_BYTES, remaining))
        if not chunk:
            raise RuntimeError("source shard ended before the selected tensor")
        destination.write(chunk)
        remaining -= len(chunk)


def repack_safetensors(
    source_path: Path,
    destination_path: Path,
    selected_keys: set[str],
) -> tuple[int, int]:
    """Copy selected tensors byte-for-byte into a new safetensors file."""
    source_header, source_data_start = _read_local_header(source_path)
    unknown = selected_keys - source_header.keys()
    if unknown:
        raise KeyError(f"{len(unknown)} selected keys absent from {source_path.name}")

    entries = sorted(
        ((key, source_header[key]) for key in selected_keys if key != "__metadata__"),
        key=lambda item: int(item[1]["data_offsets"][0]),
    )
    output_header: dict[str, Any] = {}
    if "__metadata__" in source_header:
        output_header["__metadata__"] = source_header["__metadata__"]

    output_offset = 0
    for key, entry in entries:
        start, end = (int(value) for value in entry["data_offsets"])
        size = end - start
        rewritten = dict(entry)
        rewritten["data_offsets"] = [output_offset, output_offset + size]
        output_header[key] = rewritten
        output_offset += size

    encoded_header = _encode_header(output_header)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as source, temporary.open("wb") as destination:
        destination.write(len(encoded_header).to_bytes(8, "little"))
        destination.write(encoded_header)
        for _, entry in entries:
            start, end = (int(value) for value in entry["data_offsets"])
            _copy_range(
                source,
                destination,
                source_offset=source_data_start + start,
                size=end - start,
            )
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(destination_path)
    return len(entries), output_offset


def _download(
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None = None,
) -> None:
    if destination.exists():
        if expected_bytes is None or destination.stat().st_size == expected_bytes:
            return
        raise RuntimeError(f"existing file has unexpected size: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = _headers()
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()
    next_report = ((offset // _REPORT_BYTES) + 1) * _REPORT_BYTES

    with urllib.request.urlopen(request, timeout=300) as response:
        status = getattr(response, "status", None)
        if offset and status != 206:
            raise RuntimeError(f"server refused resume request for {destination.name}")
        mode = "ab" if offset else "wb"
        with partial.open(mode) as output:
            total = offset
            while True:
                chunk = response.read(_COPY_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                total += len(chunk)
                if total >= next_report:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    rate = (total - offset) / elapsed / (1 << 20)
                    print(
                        f"{destination.name}: {total / (1 << 30):.1f} GiB "
                        f"({rate:.1f} MiB/s)",
                        flush=True,
                    )
                    next_report += _REPORT_BYTES
            output.flush()
            os.fsync(output.fileno())

    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"downloaded {partial.stat().st_size} bytes for {destination.name}; "
            f"expected {expected_bytes}"
        )
    partial.replace(destination)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _reduced_config(config: dict[str, Any], num_layers: int) -> dict[str, Any]:
    reduced = copy.deepcopy(config)
    text_config = reduced.get("text_config", reduced)
    text_config["num_hidden_layers"] = num_layers
    linear = text_config["linear_attn_config"]
    linear["kda_layers"] = [
        layer for layer in linear["kda_layers"] if layer <= num_layers
    ]
    linear["full_attn_layers"] = [
        layer for layer in linear["full_attn_layers"] if layer <= num_layers
    ]
    return reduced


def repack_checkpoint(
    *,
    repo_id: str,
    revision: str,
    num_layers: int,
    tp_size: int,
    tp_rank: int,
    ep_size: int,
    ep_rank: int,
    staging_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Download required shards sequentially and build a filtered checkpoint."""
    resolved_revision = _resolve_revision(repo_id, revision)
    config = _fetch_json(_resolve_url(repo_id, resolved_revision, "config.json"))
    index = _fetch_json(
        _resolve_url(
            repo_id,
            resolved_revision,
            "model.safetensors.index.json",
        )
    )
    weight_map = index["weight_map"]
    text_config = config.get("text_config", config)
    selected, expert_range = select_required_keys(
        weight_map,
        num_layers=num_layers,
        num_experts=int(text_config["num_experts"]),
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    keys_by_shard: dict[str, set[str]] = {}
    for key in selected:
        keys_by_shard.setdefault(weight_map[key], set()).add(key)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    repacked_bytes = 0
    repacked_tensors = 0
    shard_manifest = []
    for filename in sorted(keys_by_shard):
        _, source_bytes = _fetch_safetensors_header(
            repo_id,
            resolved_revision,
            filename,
        )
        source_path = staging_dir / filename
        destination_path = output_dir / filename
        print(f"Downloading {filename} ({source_bytes / (1 << 30):.2f} GiB)")
        _download(
            _resolve_url(repo_id, resolved_revision, filename),
            source_path,
            expected_bytes=source_bytes,
        )
        print(f"Repacking {filename}")
        tensor_count, tensor_bytes = repack_safetensors(
            source_path,
            destination_path,
            keys_by_shard[filename],
        )
        source_path.unlink()
        repacked_tensors += tensor_count
        repacked_bytes += tensor_bytes
        shard_manifest.append(
            {
                "filename": filename,
                "source_bytes": source_bytes,
                "repacked_bytes": tensor_bytes,
                "tensors": tensor_count,
            }
        )

    filtered_index = {
        "metadata": {"total_size": repacked_bytes},
        "weight_map": {key: weight_map[key] for key in sorted(selected)},
    }
    _write_json(output_dir / "model.safetensors.index.json", filtered_index)

    for filename in _METADATA_FILES:
        destination = output_dir / filename
        _download(
            _resolve_url(repo_id, resolved_revision, filename),
            destination,
        )
    (output_dir / "config.json").replace(output_dir / "config.original.json")
    _write_json(output_dir / "config.json", _reduced_config(config, num_layers))

    manifest = {
        "repo_id": repo_id,
        "resolved_revision": resolved_revision,
        "num_layers": num_layers,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "ep_size": ep_size,
        "ep_rank": ep_rank,
        "expert_range": list(expert_range),
        "repacked_tensors": repacked_tensors,
        "repacked_tensor_bytes": repacked_bytes,
        "shards": shard_manifest,
        "global_tp_tensors_are_unsharded": True,
    }
    _write_json(output_dir / "reduced_checkpoint_manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = repack_checkpoint(
        repo_id=args.repo_id,
        revision=args.revision,
        num_layers=args.num_layers,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        ep_size=args.ep_size,
        ep_rank=args.ep_rank,
        staging_dir=args.staging_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
