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
"""Estimate a reduced Kimi-K3 checkpoint using indexes and shard headers only."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_REPO_ID = "moonshotai/Kimi-K3"
_LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.block_sparse_moe\.experts\.(\d+)\.")
_CONTENT_RANGE_RE = re.compile(r"^bytes \d+-\d+/(\d+)$")
_MAX_HEADER_BYTES = 128 << 20


@dataclass(frozen=True)
class ShardEstimate:
    """Storage estimate for one source safetensors shard."""

    filename: str
    source_bytes: int
    required_bytes: int
    required_tensors: int


def _headers() -> dict[str, str]:
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "tokenspeed-kimi-k3-checkpoint-analyzer/1.0",
    }
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def _resolve_revision(repo_id: str, revision: str) -> str:
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    info = _fetch_json(
        f"https://huggingface.co/api/models/{encoded_repo}/revision/{encoded_revision}"
    )
    return str(info["sha"])


def _resolve_url(repo_id: str, revision: str, filename: str) -> str:
    return (
        f"https://huggingface.co/{repo_id}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/"
        f"{urllib.parse.quote(filename, safe='/')}"
    )


def _fetch_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    headers = _headers()
    headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        status = getattr(response, "status", None)
        content_range = response.headers.get("Content-Range", "")
        match = _CONTENT_RANGE_RE.match(content_range)
        if status != 206 or match is None:
            raise RuntimeError(
                "server did not honor the metadata-only range request; "
                f"refusing to read tensor payload from {url}"
            )
        payload = response.read(end - start + 1)
    if len(payload) != end - start + 1:
        raise RuntimeError(f"short range response from {url}")
    return payload, int(match.group(1))


def _fetch_safetensors_header(
    repo_id: str,
    revision: str,
    filename: str,
) -> tuple[dict[str, Any], int]:
    url = _resolve_url(repo_id, revision, filename)
    prefix, source_bytes = _fetch_range(url, 0, 7)
    header_bytes = int.from_bytes(prefix, byteorder="little", signed=False)
    if header_bytes <= 0 or header_bytes > _MAX_HEADER_BYTES:
        raise RuntimeError(
            f"unsafe safetensors header size {header_bytes} for {filename}"
        )
    encoded, repeated_source_bytes = _fetch_range(url, 8, 7 + header_bytes)
    if repeated_source_bytes != source_bytes:
        raise RuntimeError(f"source size changed while reading {filename}")
    return json.loads(encoded), source_bytes


def select_required_keys(
    weight_map: dict[str, str],
    *,
    num_layers: int,
    num_experts: int,
    ep_size: int,
    ep_rank: int,
) -> tuple[set[str], tuple[int, int]]:
    """Select language-model globals, early layers, and one EP rank's experts."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if ep_size <= 0 or not 0 <= ep_rank < ep_size:
        raise ValueError("ep_rank must be in [0, ep_size)")
    if num_experts % ep_size:
        raise ValueError("num_experts must divide evenly across ep_size")

    local_experts = num_experts // ep_size
    first_expert = ep_rank * local_experts
    last_expert = first_expert + local_experts
    selected: set[str] = set()

    for key in weight_map:
        if not key.startswith("language_model."):
            continue
        layer_match = _LAYER_RE.match(key)
        if layer_match is None:
            selected.add(key)
            continue
        if int(layer_match.group(1)) >= num_layers:
            continue
        expert_match = _EXPERT_RE.search(key)
        if expert_match is None:
            selected.add(key)
            continue
        expert_id = int(expert_match.group(1))
        if first_expert <= expert_id < last_expert:
            selected.add(key)

    return selected, (first_expert, last_expert)


def _tensor_bytes(entry: dict[str, Any]) -> int:
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("invalid safetensors data_offsets")
    return int(offsets[1]) - int(offsets[0])


def analyze(
    *,
    repo_id: str,
    revision: str,
    num_layers: int,
    tp_size: int,
    tp_rank: int,
    ep_size: int,
    ep_rank: int,
) -> dict[str, Any]:
    """Analyze reduced-checkpoint storage without reading any tensor payload."""
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("tp_rank must be in [0, tp_size)")

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
    num_experts = int(text_config["num_experts"])
    selected, expert_range = select_required_keys(
        weight_map,
        num_layers=num_layers,
        num_experts=num_experts,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )

    keys_by_shard: dict[str, list[str]] = {}
    for key in selected:
        keys_by_shard.setdefault(weight_map[key], []).append(key)

    estimates: list[ShardEstimate] = []
    missing: list[str] = []
    for filename in sorted(keys_by_shard):
        header, source_bytes = _fetch_safetensors_header(
            repo_id,
            resolved_revision,
            filename,
        )
        required_bytes = 0
        found = 0
        for key in keys_by_shard[filename]:
            entry = header.get(key)
            if entry is None:
                missing.append(key)
                continue
            required_bytes += _tensor_bytes(entry)
            found += 1
        estimates.append(
            ShardEstimate(
                filename=filename,
                source_bytes=source_bytes,
                required_bytes=required_bytes,
                required_tensors=found,
            )
        )

    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "num_layers": num_layers,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "ep_size": ep_size,
        "ep_rank": ep_rank,
        "num_experts": num_experts,
        "expert_range": expert_range,
        "full_checkpoint_bytes": int(index.get("metadata", {}).get("total_size", 0)),
        "total_index_tensors": len(weight_map),
        "required_tensors": len(selected),
        "source_download_bytes": sum(item.source_bytes for item in estimates),
        "repacked_tensor_bytes": sum(item.required_bytes for item in estimates),
        "shards": estimates,
        "missing": sorted(missing),
    }


def _gib(value: int) -> str:
    return f"{value / (1 << 30):.2f} GiB"


def render_markdown(result: dict[str, Any]) -> str:
    """Render a human-readable storage and shard estimate."""
    first_expert, last_expert = result["expert_range"]
    lines = [
        "# Kimi-K3 reduced-checkpoint estimate",
        "",
        f"- Repository: `{result['repo_id']}`",
        f"- Revision: `{result['resolved_revision']}`",
        f"- Layers retained: `0..{result['num_layers'] - 1}`",
        f"- Logical TP rank: `{result['tp_rank']}/{result['tp_size']}`",
        f"- Logical EP rank: `{result['ep_rank']}/{result['ep_size']}`",
        f"- Experts retained: `{first_expert}..{last_expert - 1}`",
        (
            f"- Required tensors: `{result['required_tensors']:,}` "
            f"of `{result['total_index_tensors']:,}`"
        ),
        f"- Full checkpoint: `{_gib(result['full_checkpoint_bytes'])}`",
        f"- Source shard download: `{_gib(result['source_download_bytes'])}`",
        f"- Repacked tensor payload: `{_gib(result['repacked_tensor_bytes'])}`",
        "",
        "## Required source shards",
        "",
        "| Shard | Source size | Required tensors | Repacked payload |",
        "|---|---:|---:|---:|",
    ]
    for shard in result["shards"]:
        lines.append(
            f"| `{shard.filename}` | {_gib(shard.source_bytes)} | "
            f"{shard.required_tensors:,} | {_gib(shard.required_bytes)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The source-download estimate assumes whole Hugging Face shard files.",
            (
                "- The repacked estimate keeps global TP tensors; the standard "
                "loader will slice them for the selected logical TP rank."
            ),
            "- Only the selected EP rank's expert tensors are retained.",
            (
                "- Vision tensors and language-model layers after the retained "
                "prefix are excluded."
            ),
            (
                "- This command reads JSON indexes and safetensors headers only. "
                "It does not read tensor payload bytes."
            ),
        ]
    )
    if result["missing"]:
        lines.extend(
            [
                "",
                "## Missing header entries",
                "",
                *[f"- `{key}`" for key in result["missing"]],
            ]
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = analyze(
        repo_id=args.repo_id,
        revision=args.revision,
        num_layers=args.num_layers,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        ep_size=args.ep_size,
        ep_rank=args.ep_rank,
    )
    if args.as_json:
        serializable = dict(result)
        serializable["shards"] = [shard.__dict__ for shard in serializable["shards"]]
        print(json.dumps(serializable, indent=2, sort_keys=True))
    else:
        print(render_markdown(result))
    return 1 if result["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
