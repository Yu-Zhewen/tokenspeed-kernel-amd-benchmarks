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
"""Benchmark a portable Kimi-K3 TP8/EP1 rank-0 checkpoint on one GPU."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toy_e2e.logical_rank import (  # noqa: E402
    load_logical_rank,
    logical_rank_runtime,
    model_summary,
)
from toy_e2e.rank_checkpoint import RawRankStateLoader  # noqa: E402


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _collective_summary(events: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "input_bytes": 0, "output_bytes": 0}
    )
    for event in events:
        operation = event["operation"]
        totals[operation]["calls"] += 1
        totals[operation]["input_bytes"] += int(event["input_bytes"])
        totals[operation]["output_bytes"] += int(event["output_bytes"])
    return dict(totals)


def _hotspot_summary(
    module_ms: dict[tuple[str, int, str, str], list[float]],
    model_ms: dict[str, list[float]],
    *,
    top_k: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for phase, phase_model_ms in model_ms.items():
        if not phase_model_ms:
            continue
        model_p50_ms = float(statistics.median(phase_model_ms))
        modules = []
        for (module_phase, layer, category, module_type), values in module_ms.items():
            if module_phase != phase or not values:
                continue
            timing = _summary(values)
            p50_ms = float(timing["p50"])
            modules.append(
                {
                    "layer": layer,
                    "category": category,
                    "module_type": module_type,
                    "timing_ms": timing,
                    "share_of_model_p50_pct": (
                        100.0 * p50_ms / model_p50_ms if model_p50_ms else 0.0
                    ),
                }
            )
        modules.sort(
            key=lambda item: (-float(item["timing_ms"]["p50"]), int(item["layer"]))
        )
        result[phase] = {
            "model_p50_ms": model_p50_ms,
            "top_modules": modules[:top_k],
        }
    return result


class LayerBreakdown:
    """Low-volume CUDA-event instrumentation for sampled forwards."""

    def __init__(self, runner) -> None:
        model = runner.model.language_model.model
        self.enabled = False
        self._starts: dict[int, list[torch.cuda.Event]] = defaultdict(list)
        self._samples: list[
            tuple[str, int, str, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._metadata: dict[int, tuple[str, int, str]] = {}
        self._hooks = []
        for layer_index, layer in enumerate(model.layers):
            attention = layer.self_attn
            attention_category = (
                "kda_attention"
                if "KDA" in type(attention).__name__
                else "mla_attention"
            )
            self._register(attention, attention_category, layer_index)
            if hasattr(layer, "block_sparse_moe"):
                self._register(layer.block_sparse_moe, "moe", layer_index)
            elif hasattr(layer, "mlp"):
                self._register(layer.mlp, "dense_ffn", layer_index)

    def _register(self, module, category: str, layer_index: int) -> None:
        key = id(module)
        self._metadata[key] = (category, layer_index, type(module).__name__)

        def before(current, _args):
            if not self.enabled:
                return
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._starts[id(current)].append(event)

        def after(current, _args, _output):
            if not self.enabled:
                return
            start = self._starts[id(current)].pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            category, layer, module_type = self._metadata[id(current)]
            self._samples.append((category, layer, module_type, start, end))

        self._hooks.append(module.register_forward_pre_hook(before))
        self._hooks.append(module.register_forward_hook(after))

    def begin(self) -> None:
        self._starts.clear()
        self._samples.clear()
        self.enabled = True

    def end(self) -> tuple[dict[str, float], list[dict[str, Any]]]:
        self.enabled = False
        categories: dict[str, float] = defaultdict(float)
        modules = []
        for category, layer, module_type, start, end in self._samples:
            elapsed_ms = float(start.elapsed_time(end))
            categories[category] += elapsed_ms
            modules.append(
                {
                    "category": category,
                    "layer": layer,
                    "module_type": module_type,
                    "elapsed_ms": elapsed_ms,
                }
            )
        return dict(categories), modules

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()


@dataclass
class PreparedForward:
    ctx: Any
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    extend_prefix_lens: torch.Tensor


def _execute_forward(runner, prepared: PreparedForward):
    from tokenspeed.runtime.execution.breakable_cuda_graph import active_forward

    with torch.inference_mode(), active_forward(prepared.ctx):
        return runner.forward(
            ctx=prepared.ctx,
            input_ids=prepared.input_ids,
            positions=prepared.positions,
            out_cache_loc=prepared.out_cache_loc,
            req_pool_indices=prepared.req_pool_indices,
            seq_lens=prepared.seq_lens,
            extend_prefix_lens=prepared.extend_prefix_lens,
        )


@contextmanager
def _tokenspeed_graph_phase(*, capture: bool):
    from tokenspeed.runtime.execution import cuda_graph_wrapper

    previous_graph_phase = cuda_graph_wrapper._is_cuda_graph_phase
    previous_capture_mode = cuda_graph_wrapper._is_capture_mode
    cuda_graph_wrapper._is_cuda_graph_phase = True
    cuda_graph_wrapper._is_capture_mode = capture
    try:
        yield
    finally:
        cuda_graph_wrapper._is_capture_mode = previous_capture_mode
        cuda_graph_wrapper._is_cuda_graph_phase = previous_graph_phase


def _create_cache(server_args, model_config, cache_bytes: int):
    from tokenspeed.runtime.layers.attention import registry as attention_registry

    total_gib = max(1, torch.cuda.get_device_properties(0).total_memory // (1 << 30))
    with mock.patch.object(
        attention_registry,
        "profile_available_cache_memory_bytes",
        return_value=cache_bytes,
    ):
        backend, pool, _, _, storage = attention_registry.create_attn_components(
            server_args,
            model_config,
            gpu_id=0,
            rank=0,
            gpu_memory=total_gib,
        )
    return backend, pool, storage


def _new_scheduler(pool, *, concurrency: int, chunked_prefill_size: int):
    from tokenspeed.runtime.engine.scheduler_utils import (
        aligned_max_scheduled_tokens,
        make_config,
        pool_to_cache_groups,
        scheduler_cache_geometry_from_pool,
    )
    from tokenspeed_scheduler import Scheduler

    geometry = scheduler_cache_geometry_from_pool(pool)
    cache_groups = pool_to_cache_groups(pool)
    max_scheduled_tokens = aligned_max_scheduled_tokens(
        chunked_prefill_size, cache_groups
    )
    config = make_config(
        num_device_pages=geometry.num_device_pages,
        max_scheduled_tokens=max_scheduled_tokens,
        max_batch_size=concurrency,
        prefix_granularity=geometry.prefix_granularity,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role="null",
        disable_prefix_cache=True,
        cache_groups=cache_groups,
    )
    scheduler = Scheduler(config)
    bind_scheduler = getattr(pool, "bind_paged_cache_scheduler", None)
    if callable(bind_scheduler):
        bind_scheduler(scheduler)
    return scheduler, geometry, max_scheduled_tokens


def _prepare_forward(
    backend,
    pool,
    forward_op,
    cached_lengths: dict[str, int],
) -> PreparedForward:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
        CacheBatchMetadata,
    )

    request_ids = list(forward_op.request_ids)
    input_lengths = [int(value) for value in forward_op.input_lengths]
    num_extends = int(forward_op.num_extends())
    bs = len(request_ids)
    num_tokens = sum(input_lengths)
    mode = ForwardMode.EXTEND if num_extends else ForwardMode.DECODE

    starts = [cached_lengths[request_id] for request_id in request_ids]
    ends = [start + length for start, length in zip(starts, input_lengths)]
    for index in range(num_extends):
        expected = int(forward_op.extend_prefix_lens[index])
        if starts[index] != expected:
            raise RuntimeError(
                f"scheduler prefix mismatch for {request_ids[index]}: "
                f"{expected} != {starts[index]}"
            )

    req_pool_indices = torch.tensor(
        forward_op.request_pool_indices,
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor(ends, dtype=torch.int32, device="cuda")
    extend_seq_lens_cpu = torch.tensor(
        input_lengths[:num_extends], dtype=torch.int32
    )
    extend_seq_lens = extend_seq_lens_cpu.to("cuda")
    extend_prefix_lens_cpu = torch.tensor(
        list(forward_op.extend_prefix_lens), dtype=torch.int32
    )
    extend_prefix_lens = extend_prefix_lens_cpu.to("cuda")
    positions = torch.cat(
        [
            torch.arange(start, end, dtype=torch.int64, device="cuda")
            for start, end in zip(starts, ends)
        ]
    )
    # The scheduler carries prompt chunks in input_ids. Decode rows normally
    # read the prior sampled token from ModelExecutor.runtime_states, which this
    # standalone harness intentionally replaces with a deterministic token.
    prompt_input_ids = list(forward_op.input_ids)
    decode_token_count = sum(input_lengths[num_extends:])
    input_ids = torch.tensor(
        [*prompt_input_ids, *([1] * decode_token_count)],
        dtype=torch.int64,
        device="cuda",
    )

    cache_metadata = CacheBatchMetadata.from_forward_op(
        forward_op,
        device="cuda",
        contract=pool.arena.runtime_contract,
        num_requests=bs,
    )
    page_table = cache_metadata.require_full_attention_table(
        active_forward_op=forward_op
    )
    block_granularity = cache_metadata.block_granularity
    rows = torch.repeat_interleave(
        torch.arange(bs, dtype=torch.int64, device="cuda"),
        torch.tensor(input_lengths, dtype=torch.int64, device="cuda"),
    )
    logical_pages = torch.div(
        positions,
        block_granularity,
        rounding_mode="floor",
    )
    physical_pages = page_table[rows, logical_pages].to(torch.int64)
    out_cache_loc = (
        physical_pages * block_granularity
        + torch.remainder(positions, block_granularity)
    )
    block_tables = dict(cache_metadata.tables(active_forward_op=forward_op))
    backend.init_forward_metadata(
        bs=bs,
        num_extends=num_extends,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        page_table=page_table,
        forward_mode=mode,
        extend_with_prefix=any(starts[:num_extends]),
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_prefix_lens=extend_prefix_lens,
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        positions=positions,
        out_cache_loc=out_cache_loc,
        global_num_tokens=[num_tokens] * 8,
        all_decode_or_idle=not num_extends,
        num_tokens=num_tokens,
        cache_metadata=cache_metadata,
        forward_batch=forward_op,
        block_tables=block_tables,
    )
    gather_ids = torch.cumsum(
        torch.tensor(input_lengths, dtype=torch.int64, device="cuda"), dim=0
    ) - 1
    ctx = ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=pool,
        bs=bs,
        num_extends=num_extends,
        input_num_tokens=num_tokens,
        forward_mode=mode,
        global_num_tokens=[num_tokens] * 8,
        global_bs=[bs] * 8,
        all_decode_or_idle=not num_extends,
        all_extend=num_extends == bs,
        gather_ids=gather_ids,
    )
    return PreparedForward(
        ctx=ctx,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_prefix_lens=extend_prefix_lens,
    )


def _run_workload(
    *,
    runner,
    backend,
    pool,
    logical_backend,
    concurrency: int,
    prompt_tokens: int,
    output_tokens: int,
    chunked_prefill_size: int,
    breakdown: LayerBreakdown | None = None,
    hotspot_top_k: int = 10,
    before_forward: Callable[[str], None] | None = None,
    after_forward: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    from tokenspeed.runtime.engine.scheduler_utils import (
        advance_scheduler,
        make_extend_result_event,
        make_finish_event,
        make_spec,
    )

    pool.clear_kv_buffers()
    scheduler, geometry, effective_chunk = _new_scheduler(
        pool,
        concurrency=concurrency,
        chunked_prefill_size=chunked_prefill_size,
    )
    requests = [f"request-{index}" for index in range(concurrency)]
    scheduler.submit_requests(
        [
            make_spec(
                request_id,
                [1] * prompt_tokens,
                max_new_tokens=output_tokens,
            )
            for request_id in requests
        ]
    )
    cached_lengths = {request_id: 0 for request_id in requests}
    generated = {request_id: 0 for request_id in requests}
    first_token_ms: dict[str, float] = {}
    model_ms: dict[str, list[float]] = defaultdict(list)
    step_wall_ms: dict[str, list[float]] = defaultdict(list)
    component_ms: dict[str, list[float]] = defaultdict(list)
    module_ms: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    collective_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    started = time.perf_counter()

    max_steps = prompt_tokens * concurrency + output_tokens + concurrency + 100
    for _ in range(max_steps):
        if all(count >= output_tokens for count in generated.values()):
            break
        step_started = time.perf_counter()
        plan = scheduler.next_execution_plan()
        if not plan.forward:
            raise RuntimeError("scheduler produced no forward before requests finished")
        forward_op = plan.forward[0]
        if not forward_op.request_ids:
            raise RuntimeError("scheduler produced an empty forward before completion")
        pages_to_zero = dict(plan.pages_to_zero)
        zero_new_blocks = getattr(pool, "zero_new_blocks", None)
        if callable(zero_new_blocks):
            zero_new_blocks(pages_to_zero)
        else:
            pool.zero_new_pages(pages_to_zero)
        prepared = _prepare_forward(backend, pool, forward_op, cached_lengths)
        phase = "prefill" if prepared.ctx.num_extends else "decode"
        logical_backend.snapshot(reset=True)

        if before_forward is not None:
            before_forward(phase)
        if breakdown is not None:
            breakdown.begin()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _execute_forward(runner, prepared)
        end_event.record()
        torch.cuda.synchronize()
        elapsed = float(start_event.elapsed_time(end_event))
        model_ms[phase].append(elapsed)
        if breakdown is not None:
            category_totals, module_samples = breakdown.end()
            for category, value in category_totals.items():
                component_ms[f"{phase}.{category}"].append(value)
            for sample in module_samples:
                key = (
                    phase,
                    int(sample["layer"]),
                    str(sample["category"]),
                    str(sample["module_type"]),
                )
                module_ms[key].append(float(sample["elapsed_ms"]))
        if after_forward is not None:
            after_forward(phase)
        collective_events[phase].extend(logical_backend.snapshot(reset=True))

        events = []
        num_extends = int(forward_op.num_extends())
        for index, (request_id, length) in enumerate(
            zip(forward_op.request_ids, forward_op.input_lengths)
        ):
            cached_lengths[request_id] += int(length)
            final_prefill = (
                index < num_extends
                and int(forward_op.extend_prefix_lens[index]) + int(length)
                == int(forward_op.prefill_lengths[index])
            )
            produces_token = final_prefill or index >= num_extends
            tokens = [1] if produces_token else []
            if produces_token:
                generated[request_id] += 1
                first_token_ms.setdefault(
                    request_id, (time.perf_counter() - started) * 1e3
                )
            events.append(make_extend_result_event(request_id, tokens))
            if generated[request_id] >= output_tokens:
                events.append(make_finish_event(request_id))
        advance_scheduler(scheduler, events)
        step_wall_ms[phase].append((time.perf_counter() - step_started) * 1e3)
    else:
        raise RuntimeError("scheduler workload exceeded its safety step limit")

    wall_ms = (time.perf_counter() - started) * 1e3
    collective_summaries = {
        phase: _collective_summary(events)
        for phase, events in collective_events.items()
    }

    return {
        "concurrency": concurrency,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "effective_chunked_prefill_size": effective_chunk,
        "cache_token_capacity": geometry.token_capacity,
        "wall_ms": wall_ms,
        "first_token_ms": _summary(list(first_token_ms.values())),
        "model_ms": {phase: _summary(values) for phase, values in model_ms.items()},
        "step_wall_ms": {
            phase: _summary(values) for phase, values in step_wall_ms.items()
        },
        "per_user_decode_tps": (
            1e3 / statistics.fmean(step_wall_ms["decode"])
            if step_wall_ms["decode"]
            else 0.0
        ),
        "aggregate_output_tps": concurrency * output_tokens * 1e3 / wall_ms,
        "component_ms": {
            category: _summary(values) for category, values in component_ms.items()
        },
        "hotspots": _hotspot_summary(
            module_ms,
            model_ms,
            top_k=hotspot_top_k,
        ),
        "collectives": collective_summaries,
    }


def _prepare_full_decode_batch(
    *,
    runner,
    backend,
    pool,
    concurrency: int,
    prompt_tokens: int,
    chunked_prefill_size: int,
) -> tuple[PreparedForward, int]:
    from tokenspeed.runtime.engine.scheduler_utils import (
        advance_scheduler,
        make_extend_result_event,
        make_spec,
    )

    pool.clear_kv_buffers()
    scheduler, _geometry, effective_chunk = _new_scheduler(
        pool,
        concurrency=concurrency,
        chunked_prefill_size=chunked_prefill_size,
    )
    request_ids = [f"graph-request-{index}" for index in range(concurrency)]
    request_set = set(request_ids)
    scheduler.submit_requests(
        [
            make_spec(
                request_id,
                [1] * prompt_tokens,
                max_new_tokens=1024,
            )
            for request_id in request_ids
        ]
    )
    cached_lengths = {request_id: 0 for request_id in request_ids}

    max_steps = prompt_tokens * concurrency + concurrency + 100
    for _ in range(max_steps):
        plan = scheduler.next_execution_plan()
        if len(plan.forward) != 1:
            raise RuntimeError(
                "graph preparation requires exactly one forward operation per plan"
            )
        forward_op = plan.forward[0]
        pages_to_zero = dict(plan.pages_to_zero)
        zero_new_blocks = getattr(pool, "zero_new_blocks", None)
        if callable(zero_new_blocks):
            zero_new_blocks(pages_to_zero)
        else:
            pool.zero_new_pages(pages_to_zero)
        prepared = _prepare_forward(backend, pool, forward_op, cached_lengths)
        if (
            prepared.ctx.num_extends == 0
            and len(forward_op.request_ids) == concurrency
            and set(forward_op.request_ids) == request_set
        ):
            return prepared, effective_chunk

        _execute_forward(runner, prepared)
        torch.cuda.synchronize()
        events = []
        num_extends = int(forward_op.num_extends())
        for index, (request_id, length) in enumerate(
            zip(forward_op.request_ids, forward_op.input_lengths)
        ):
            cached_lengths[request_id] += int(length)
            final_prefill = (
                index < num_extends
                and int(forward_op.extend_prefix_lens[index]) + int(length)
                == int(forward_op.prefill_lengths[index])
            )
            produces_token = final_prefill or index >= num_extends
            events.append(
                make_extend_result_event(request_id, [1] if produces_token else [])
            )
        advance_scheduler(scheduler, events)
    raise RuntimeError("could not prepare a full decode batch for graph capture")


def _benchmark_decode_graph(
    *,
    runner,
    backend,
    pool,
    logical_backend,
    concurrency: int,
    prompt_tokens: int,
    chunked_prefill_size: int,
    replays: int,
) -> dict[str, Any]:
    prepared, effective_chunk = _prepare_full_decode_batch(
        runner=runner,
        backend=backend,
        pool=pool,
        concurrency=concurrency,
        prompt_tokens=prompt_tokens,
        chunked_prefill_size=chunked_prefill_size,
    )
    sequence_lengths = [int(value) for value in prepared.seq_lens.cpu().tolist()]
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), _tokenspeed_graph_phase(capture=False):
        for _ in range(4):
            _execute_forward(runner, prepared)
    torch.cuda.synchronize()

    logical_backend.snapshot(reset=True)
    graph = torch.cuda.CUDAGraph()
    capture_started = time.perf_counter()
    with _tokenspeed_graph_phase(capture=True):
        with torch.cuda.graph(graph, stream=capture_stream):
            captured_output = _execute_forward(runner, prepared)
    torch.cuda.synchronize()
    capture_wall_ms = (time.perf_counter() - capture_started) * 1e3
    collectives = _collective_summary(logical_backend.snapshot(reset=True))

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    starts: list[torch.cuda.Event] = []
    ends: list[torch.cuda.Event] = []
    wall_started = time.perf_counter()
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        starts.append(start)
        ends.append(end)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_started) * 1e3
    model_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    mean_model_ms = statistics.fmean(model_ms)
    result = {
        "scope": "static first full decode batch after prefill",
        "concurrency": concurrency,
        "prompt_tokens": prompt_tokens,
        "sequence_lengths": sequence_lengths,
        "effective_chunked_prefill_size": effective_chunk,
        "replays": replays,
        "capture_wall_ms": capture_wall_ms,
        "replay_wall_ms": wall_ms,
        "model_ms": _summary(model_ms),
        "per_user_decode_tps": 1e3 / mean_model_ms,
        "aggregate_decode_tps": concurrency * 1e3 / mean_model_ms,
        "collectives_per_replay": collectives,
    }
    del captured_output, graph, capture_stream
    torch.cuda.empty_cache()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected-arch",
        choices=("gfx950", "gfx1250"),
        required=True,
    )
    parser.add_argument("--tokenspeed-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--container-image", default="unavailable")
    parser.add_argument(
        "--load-format",
        choices=("raw-rank-state", "dummy", "safetensors"),
        default="raw-rank-state",
    )
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=(1, 16),
        nargs="+",
        required=True,
        help="one or more serving batch sizes (for example: 1 16)",
    )
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument("--cache-gib", type=float, default=32.0)
    parser.add_argument("--warmup-output-tokens", type=int, default=2)
    parser.add_argument("--profile-output-tokens", type=int, default=8)
    parser.add_argument(
        "--hotspot-top-k",
        type=int,
        default=10,
        help="slowest layer components retained per profiled phase",
    )
    parser.add_argument(
        "--decode-graph-replays",
        type=int,
        default=20,
        help="steady-state decode graph replays per batch size; zero disables",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "logical-rank benchmark requires exactly one visible GPU; set "
            "ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES"
        )
    if min(
        args.prompt_tokens,
        args.output_tokens,
        args.warmup_output_tokens,
        args.profile_output_tokens,
    ) <= 0:
        raise ValueError("token counts must be positive")
    if args.cache_gib <= 0:
        raise ValueError("--cache-gib must be positive")
    if args.decode_graph_replays < 0:
        raise ValueError("--decode-graph-replays cannot be negative")
    if args.hotspot_top_k <= 0:
        raise ValueError("--hotspot-top-k must be positive")

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
        "safetensors": "safetensors",
    }[args.load_format]
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    with logical_rank_runtime() as logical_backend:
        server_args, model_config, runner = load_logical_rank(
            args.checkpoint,
            load_format=load_format,
            max_model_len=args.prompt_tokens + args.output_tokens,
            max_num_seqs=max(concurrencies),
            chunked_prefill_size=args.chunked_prefill_size,
        )
        load_wall_s = time.perf_counter() - load_started
        loaded_model_summary = model_summary(server_args, runner)
        load_peak_gib = torch.cuda.max_memory_allocated() / (1 << 30)
        print(
            f"Loaded {args.load_format} in {load_wall_s:.2f}s "
            f"(peak {load_peak_gib:.2f} GiB)",
            flush=True,
        )
        backend, pool, cache_storage = _create_cache(
            server_args,
            model_config,
            int(args.cache_gib * (1 << 30)),
        )
        breakdown = LayerBreakdown(runner)
        runs = []
        try:
            for concurrency in concurrencies:
                print(f"Starting concurrency {concurrency} warmup", flush=True)
                warmup = _run_workload(
                    runner=runner,
                    backend=backend,
                    pool=pool,
                    logical_backend=logical_backend,
                    concurrency=concurrency,
                    prompt_tokens=args.prompt_tokens,
                    output_tokens=args.warmup_output_tokens,
                    chunked_prefill_size=args.chunked_prefill_size,
                )
                print(f"Profiling concurrency {concurrency} breakdown", flush=True)
                profile = _run_workload(
                    runner=runner,
                    backend=backend,
                    pool=pool,
                    logical_backend=logical_backend,
                    concurrency=concurrency,
                    prompt_tokens=args.prompt_tokens,
                    output_tokens=args.profile_output_tokens,
                    chunked_prefill_size=args.chunked_prefill_size,
                    breakdown=breakdown,
                    hotspot_top_k=args.hotspot_top_k,
                )
                if args.decode_graph_replays:
                    print(
                        f"Capturing concurrency {concurrency} decode graph",
                        flush=True,
                    )
                graph_decode = (
                    _benchmark_decode_graph(
                        runner=runner,
                        backend=backend,
                        pool=pool,
                        logical_backend=logical_backend,
                        concurrency=concurrency,
                        prompt_tokens=args.prompt_tokens,
                        chunked_prefill_size=args.chunked_prefill_size,
                        replays=args.decode_graph_replays,
                    )
                    if args.decode_graph_replays
                    else None
                )
                print(
                    f"Running concurrency {concurrency} "
                    f"{args.prompt_tokens}/{args.output_tokens} workload",
                    flush=True,
                )
                benchmark = _run_workload(
                    runner=runner,
                    backend=backend,
                    pool=pool,
                    logical_backend=logical_backend,
                    concurrency=concurrency,
                    prompt_tokens=args.prompt_tokens,
                    output_tokens=args.output_tokens,
                    chunked_prefill_size=args.chunked_prefill_size,
                )
                runs.append(
                    {
                        "concurrency": concurrency,
                        "warmup": warmup,
                        "profile": profile,
                        "graph_decode": graph_decode,
                        "benchmark": benchmark,
                    }
                )
                print(f"Completed concurrency {concurrency}", flush=True)
        finally:
            breakdown.close()
        import transformers
        import triton

        device = torch.cuda.get_device_name(0)
        result = {
            "format": "tokenspeed_logical_rank_benchmark_v1",
            "status": "passed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": "toy one-GPU Kimi-K3 TP8/EP1 logical rank 0",
            "device": device,
            "architecture": architecture,
            "hardware": {
                "device": device,
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
                "container_image": args.container_image,
                "os": platform.platform(),
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
                "output_tokens": args.output_tokens,
                "concurrencies": list(concurrencies),
                "chunked_prefill_size": args.chunked_prefill_size,
                "measurement": (
                    "full eager scheduler workload plus static decode graph replay"
                ),
            },
            "checkpoint": str(args.checkpoint),
            "load_format": args.load_format,
            "load_wall_s": load_wall_s,
            "load_peak_gib": load_peak_gib,
            "model": loaded_model_summary,
            "cache": cache_storage,
            "runtime_peak_gib": torch.cuda.max_memory_allocated() / (1 << 30),
            "runs": runs,
        }

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    del runner
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
