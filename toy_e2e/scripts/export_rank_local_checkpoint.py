#!/usr/bin/env python3
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
"""Export Kimi-K3 TP8/EP1 rank 0 before GPU-specific weight preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_e2e.logical_rank import (  # noqa: E402
    load_logical_rank,
    logical_rank_runtime,
    model_summary,
)
from toy_e2e.rank_checkpoint import (  # noqa: E402
    MANIFEST_NAME,
    ExportOptions,
    make_exporting_loader,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part-gib", type=float, default=2.0)
    parser.add_argument("--tokenspeed-revision")
    parser.add_argument("--source-revision")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "export requires exactly one visible GPU; set ROCR_VISIBLE_DEVICES "
            "and HIP_VISIBLE_DEVICES to one device"
        )
    if args.part_gib <= 0:
        raise ValueError("--part-gib must be positive")

    options = ExportOptions(
        output_dir=args.output,
        source_dir=args.source,
        max_part_bytes=int(args.part_gib * (1 << 30)),
        overwrite=args.overwrite,
        tokenspeed_revision=args.tokenspeed_revision,
        source_revision=args.source_revision,
    )
    loader = make_exporting_loader(options)

    torch.cuda.set_device(0)
    with logical_rank_runtime():
        server_args, _model_config, runner = load_logical_rank(
            args.source,
            load_format=loader,
        )
        summary = model_summary(server_args, runner)

    manifest = json.loads(
        (args.output / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    result = {
        "status": "passed",
        "checkpoint": str(args.output),
        "tensor_gib": manifest["tensor_bytes"] / (1 << 30),
        "part_count": len(manifest["parts"]),
        "model": summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    del runner
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
