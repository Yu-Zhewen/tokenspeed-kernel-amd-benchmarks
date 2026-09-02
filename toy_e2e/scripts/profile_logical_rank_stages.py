#!/usr/bin/env python3
"""Capture stage-separated GPU traces for the one-GPU logical-rank workload."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
import triton

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_e2e.benchmark_logical_rank import _create_cache, _run_workload  # noqa: E402
from toy_e2e.logical_rank import (  # noqa: E402
    load_logical_rank,
    logical_rank_runtime,
    model_summary,
)
from toy_e2e.rank_checkpoint import RawRankStateLoader  # noqa: E402
from toy_e2e.workload import (  # noqa: E402
    DEFAULT_PROMPT_SEED,
    DEFAULT_SYNTHETIC_VOCAB_SIZE,
)


class StageTrace:
    """Start the GPU profiler immediately before one target forward stage."""

    def __init__(
        self,
        *,
        phase: str,
        output: Path,
        max_steps: int | None,
    ) -> None:
        self.phase = phase
        self.output = output
        self.max_steps = max_steps
        self.steps = 0
        self._profiler: Any | None = None

    def before_forward(self, phase: str) -> None:
        if phase != self.phase or self._profiler is not None:
            return
        if self.max_steps is not None and self.steps >= self.max_steps:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        )
        self._profiler.start()

    def after_forward(self, phase: str) -> None:
        if phase != self.phase or self._profiler is None:
            return
        self.steps += 1
        if self.max_steps is not None and self.steps >= self.max_steps:
            self.close()

    def close(self) -> None:
        if self._profiler is None:
            return
        torch.cuda.synchronize()
        profiler = self._profiler
        self._profiler = None
        profiler.stop()
        profiler.export_chrome_trace(str(self.output))


def _profile_stage(
    *,
    runner: Any,
    backend: Any,
    pool: Any,
    logical_backend: Any,
    concurrency: int,
    prompt_tokens: int,
    chunked_prefill_size: int,
    prompt_seed: int,
    synthetic_vocabulary_size: int,
    phase: str,
    steps: int,
    output: Path,
) -> dict[str, Any]:
    output_tokens = 1 if phase == "prefill" else steps + 1
    trace = StageTrace(
        phase=phase,
        output=output,
        max_steps=steps,
    )
    try:
        workload = _run_workload(
            runner=runner,
            backend=backend,
            pool=pool,
            logical_backend=logical_backend,
            concurrency=concurrency,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            chunked_prefill_size=chunked_prefill_size,
            prompt_seed=prompt_seed,
            synthetic_vocabulary_size=synthetic_vocabulary_size,
            before_forward=trace.before_forward,
            after_forward=trace.after_forward,
        )
    finally:
        trace.close()
    if trace.steps != steps:
        raise RuntimeError(
            f"captured {trace.steps} {phase} forwards at C{concurrency}; "
            f"expected {steps}"
        )
    if not output.is_file():
        raise RuntimeError(f"profiler did not create {output}")
    return {
        "phase": phase,
        "forward_count": trace.steps,
        "trace": str(output),
        "model_ms": workload["model_ms"].get(phase, {"count": 0}),
        "collectives": workload["collectives"].get(phase, {}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "logical-rank profiling requires exactly one visible GPU; set "
            "ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES"
        )
    if args.prompt_tokens <= 0 or args.decode_steps <= 0:
        raise ValueError("token and step counts must be positive")
    if args.chunked_prefill_size <= 0 or args.cache_gib <= 0:
        raise ValueError("prefill size and cache GiB must be positive")
    if args.synthetic_vocabulary_size <= 0:
        raise ValueError("synthetic vocabulary size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    architecture = torch.cuda.get_device_properties(0).gcnArchName
    if not architecture.startswith(args.expected_arch):
        raise RuntimeError(
            f"expected {args.expected_arch}, detected {architecture}"
        )
    concurrencies = tuple(dict.fromkeys(args.concurrency))
    load_format: str | type = {
        "raw-rank-state": RawRankStateLoader,
        "dummy": "dummy",
    }[args.load_format]
    load_started = time.perf_counter()
    with logical_rank_runtime() as logical_backend:
        server_args, model_config, runner = load_logical_rank(
            args.checkpoint,
            load_format=load_format,
            max_model_len=args.prompt_tokens + args.decode_steps + 1,
            max_num_seqs=max(concurrencies),
            chunked_prefill_size=args.chunked_prefill_size,
        )
        if args.synthetic_vocabulary_size > model_config.vocab_size:
            raise ValueError(
                "--synthetic-vocabulary-size exceeds model vocabulary: "
                f"{args.synthetic_vocabulary_size} > {model_config.vocab_size}"
            )
        load_wall_s = time.perf_counter() - load_started
        loaded_model = model_summary(server_args, runner)
        backend, pool, _cache_storage = _create_cache(
            server_args,
            model_config,
            int(args.cache_gib * (1 << 30)),
        )
        runs = []
        for concurrency in concurrencies:
            print(f"Warming C{concurrency}", flush=True)
            _run_workload(
                runner=runner,
                backend=backend,
                pool=pool,
                logical_backend=logical_backend,
                concurrency=concurrency,
                prompt_tokens=args.prompt_tokens,
                output_tokens=2,
                chunked_prefill_size=args.chunked_prefill_size,
                prompt_seed=args.prompt_seed,
                synthetic_vocabulary_size=args.synthetic_vocabulary_size,
            )
            prefill_steps = math.ceil(
                concurrency * args.prompt_tokens / args.chunked_prefill_size
            )
            print(
                f"Capturing C{concurrency} prefill ({prefill_steps} forwards)",
                flush=True,
            )
            prefill_path = (
                args.output_dir
                / f"c{concurrency}"
                / "prefill"
                / f"toy-c{concurrency}-TP0-EXTEND.trace.json"
            )
            prefill = _profile_stage(
                runner=runner,
                backend=backend,
                pool=pool,
                logical_backend=logical_backend,
                concurrency=concurrency,
                prompt_tokens=args.prompt_tokens,
                chunked_prefill_size=args.chunked_prefill_size,
                prompt_seed=args.prompt_seed,
                synthetic_vocabulary_size=args.synthetic_vocabulary_size,
                phase="prefill",
                steps=prefill_steps,
                output=prefill_path,
            )
            print(
                f"Capturing C{concurrency} decode ({args.decode_steps} forwards)",
                flush=True,
            )
            decode_path = (
                args.output_dir
                / f"c{concurrency}"
                / "decode"
                / f"toy-c{concurrency}-TP0-DECODE.trace.json"
            )
            decode = _profile_stage(
                runner=runner,
                backend=backend,
                pool=pool,
                logical_backend=logical_backend,
                concurrency=concurrency,
                prompt_tokens=args.prompt_tokens,
                chunked_prefill_size=args.chunked_prefill_size,
                prompt_seed=args.prompt_seed,
                synthetic_vocabulary_size=args.synthetic_vocabulary_size,
                phase="decode",
                steps=args.decode_steps,
                output=decode_path,
            )
            runs.append(
                {
                    "concurrency": concurrency,
                    "prefill": prefill,
                    "decode": decode,
                }
            )

    result = {
        "format": "tokenspeed_logical_rank_profile_v2",
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "architecture": architecture,
            "gpu_count": 1,
        },
        "software": {
            "tokenspeed_revision": args.tokenspeed_revision,
            "model_revision": args.model_revision,
            "pytorch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": transformers.__version__,
            "triton": triton.__version__,
            "profile_backend": os.environ.get(
                "TOKENSPEED_KERNEL_PROFILE_BACKEND", "torch-default"
            ),
            "container_image": args.container_image,
            "os": platform.platform(),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "load_format": args.load_format,
            "load_wall_s": load_wall_s,
        },
        "topology": {
            "physical_ranks": 1,
            "logical_tp_size": 8,
            "logical_tp_rank": 0,
            "expert_parallel": 1,
            "collectives": "local shape/traffic substitutes",
        },
        "workload": {
            "prompt_tokens": args.prompt_tokens,
            "concurrencies": list(concurrencies),
            "chunked_prefill_size": args.chunked_prefill_size,
            "decode_profile_steps": args.decode_steps,
            "prompt_source": "deterministic varied synthetic token IDs",
            "prompt_seed": args.prompt_seed,
            "synthetic_vocabulary_size": args.synthetic_vocabulary_size,
            "decode_input": "deterministic rank-local token ID 1",
        },
        "model": loaded_model,
        "runs": runs,
    }
    manifest = args.output_dir / "profile_manifest.json"
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    manifest.write_text(encoded, encoding="utf-8")
    print(encoded, end="", flush=True)
    del runner
    torch.cuda.empty_cache()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-arch",
        choices=("gfx950", "gfx1250"),
        required=True,
    )
    parser.add_argument("--tokenspeed-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--container-image",
        default="zhewenyu/kimi-k3-e2e:tokenspeed-0b1061eb",
    )
    parser.add_argument(
        "--load-format",
        choices=("raw-rank-state", "dummy"),
        default="raw-rank-state",
    )
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=(1, 16),
        nargs="+",
        default=[1, 16],
    )
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument("--cache-gib", type=float, default=32.0)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--prompt-seed", type=int, default=DEFAULT_PROMPT_SEED)
    parser.add_argument(
        "--synthetic-vocabulary-size",
        type=int,
        default=DEFAULT_SYNTHETIC_VOCAB_SIZE,
    )
    return parser.parse_args()


def main() -> int:
    run(_parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
