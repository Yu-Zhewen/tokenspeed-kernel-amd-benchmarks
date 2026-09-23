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
from tokenspeed_kernel.profiling import start_shape_capture, stop_shape_capture

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
        shapes_output: Path | None = None,
        capture_trace: bool = True,
    ) -> None:
        self.phase = phase
        self.output = output
        self.shapes_output = shapes_output or output.with_suffix(".shapes.json")
        self.max_steps = max_steps
        self.capture_trace = capture_trace
        self.steps = 0
        self._profiler: Any | None = None
        self._active = False
        self._shape_capture_active = False

    def before_forward(self, phase: str) -> None:
        if phase != self.phase or self._active:
            return
        if self.max_steps is not None and self.steps >= self.max_steps:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.shapes_output.parent.mkdir(parents=True, exist_ok=True)
        start_shape_capture()
        self._shape_capture_active = True
        self._active = True
        try:
            if self.capture_trace:
                self._profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=(
                        os.environ.get("TOKENSPEED_TORCH_PROFILER_RECORD_SHAPES")
                        == "1"
                    ),
                    with_stack=False,
                )
                self._profiler.start()
        except BaseException:
            self._active = False
            self._stop_shape_capture()
            raise

    def after_forward(self, phase: str) -> None:
        if phase != self.phase or not self._active:
            return
        self.steps += 1
        if self.max_steps is not None and self.steps >= self.max_steps:
            self.close()

    def close(self) -> None:
        if not self._active:
            return
        self._active = False
        profiler = self._profiler
        self._profiler = None
        try:
            if profiler is not None:
                torch.cuda.synchronize()
                profiler.stop()
                profiler.export_chrome_trace(str(self.output))
        finally:
            self._stop_shape_capture()

    def _stop_shape_capture(self) -> None:
        if not self._shape_capture_active:
            return
        self._shape_capture_active = False
        stop_shape_capture(self.shapes_output)


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
    capture_trace: bool = True,
) -> dict[str, Any]:
    output_tokens = 1 if phase == "prefill" else steps + 1
    trace = StageTrace(
        phase=phase,
        output=output,
        max_steps=steps,
        capture_trace=capture_trace,
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
    if capture_trace and not output.is_file():
        raise RuntimeError(f"profiler did not create {output}")
    return {
        "phase": phase,
        "forward_count": trace.steps,
        "trace": str(output) if capture_trace else None,
        "shapes": str(trace.shapes_output),
        "model_ms": workload["model_ms"].get(phase, {"count": 0}),
        "collectives": workload["collectives"].get(phase, {}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "logical-rank profiling requires exactly one visible GPU; set "
            "ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES"
        )
    if args.prompt_tokens <= 0 or args.decode_steps < 0:
        raise ValueError("token count must be positive and step count non-negative")
    if args.output_tokens <= 0:
        raise ValueError("--output-tokens must be positive")
    if args.decode_steps > args.output_tokens:
        raise ValueError(
            "--decode-steps cannot exceed --output-tokens; the KV allocation "
            "is sized from --output-tokens"
        )
    if args.chunked_prefill_size <= 0 or args.cache_gib <= 0:
        raise ValueError("prefill size and cache GiB must be positive")
    if args.synthetic_vocabulary_size <= 0:
        raise ValueError("synthetic vocabulary size must be positive")
    if args.profile_kimi3_moe:
        from tokenspeed.runtime.layers.moe import latent

        enable_profiling = getattr(latent, "enable_kimi3_moe_profiling", None)
        if enable_profiling is not None:
            enable_profiling()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    architecture = torch.cuda.get_device_properties(0).gcnArchName
    if not architecture.startswith(args.expected_arch):
        raise RuntimeError(f"expected {args.expected_arch}, detected {architecture}")
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
            max_model_len=args.prompt_tokens + args.output_tokens,
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
        backend.init_cuda_graph_state(
            max(concurrencies),
            cache_group_specs=tuple(pool.arena.cache_group_specs),
            cache_group_page_counts=pool.arena.cache_group_page_counts,
            max_tokens_per_req=1,
            overlap_schedule_depth=0,
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
                capture_trace=not args.shapes_only,
            )
            if args.decode_steps:
                print(
                    f"Capturing C{concurrency} decode "
                    f"({args.decode_steps} forwards)",
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
                    capture_trace=not args.shapes_only,
                )
            else:
                print(
                    f"Skipping C{concurrency} decode capture",
                    flush=True,
                )
                decode = {
                    "phase": "decode",
                    "forward_count": 0,
                    "trace": None,
                    "shapes": None,
                    "model_ms": {"count": 0},
                    "collectives": {},
                }
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
            "tokenspeed_worktree_sha256": os.environ.get(
                "TOKENSPEED_WORKTREE_SHA256", "unavailable"
            ),
            "tokenspeed_worktree_dirty": os.environ.get(
                "TOKENSPEED_WORKTREE_DIRTY", "unavailable"
            ),
            "benchmarks_worktree_sha256": os.environ.get(
                "BENCHMARKS_WORKTREE_SHA256", "unavailable"
            ),
            "benchmarks_worktree_dirty": os.environ.get(
                "BENCHMARKS_WORKTREE_DIRTY", "unavailable"
            ),
            "model_revision": args.model_revision,
            "pytorch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": transformers.__version__,
            "triton": triton.__version__,
            "profile_backend": os.environ.get(
                "TOKENSPEED_KERNEL_PROFILE_BACKEND", "torch-default"
            )
            if not args.shapes_only
            else "shape-capture-only",
            "kimi3_moe_scopes": args.profile_kimi3_moe,
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
    # Sizes the KV allocation, matching benchmark_logical_rank.py. Keeping
    # this separate from --decode-steps means the number of forwards we
    # capture cannot change the memory geometry being profiled.
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument(
        "--shapes-only",
        action="store_true",
        help="capture exact selected-kernel shapes without GPU trace files",
    )
    parser.add_argument(
        "--profile-kimi3-moe",
        action="store_true",
        help="record named semantic scopes for the Kimi-K3 MoE flow",
    )
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
