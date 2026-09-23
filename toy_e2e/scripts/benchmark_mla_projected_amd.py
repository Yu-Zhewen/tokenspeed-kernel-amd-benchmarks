#!/usr/bin/env python3
"""Benchmark projected Kimi-K3 MLA paths on gfx950 or gfx1250."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from unittest import mock

import torch
from tokenspeed_kernel import mla_decode_with_kvcache, mla_project_value
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul
from tokenspeed_kernel.profiling import ShapeCapture

_PROJECTED_DECODE_ATOL = 0.25


def _measure_us(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) * 1_000.0))
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def _selected_kernel(fn: Callable[[], torch.Tensor]) -> str:
    capture = ShapeCapture.get()
    previous = capture.enabled
    capture.clear()
    capture.enabled = True
    try:
        fn()
        records = list(capture._records)
    finally:
        capture.enabled = previous
        capture.clear()
    if not records:
        raise RuntimeError("shape capture did not record a selected MLA kernel")
    return records[-1].kernel_name


def _overrides(arch: str) -> dict[str, str]:
    if arch.startswith("gfx950"):
        return {
            "decode": "gluon_mla_decode_fp8xfp8_gfx950_bh16bn128",
            "decode_project": "gluon_mla_decode_projected_value_gfx950",
            "project": "gluon_mla_project_value_gfx950",
        }
    if arch.startswith("gfx1250"):
        return {
            "decode": "gluon_mla_decode_gfx1250",
            "decode_project": "gluon_mla_decode_projected_value_gfx1250",
            "project": "gluon_mla_project_value_gfx1250",
        }
    raise RuntimeError(f"expected gfx950 or gfx1250, detected {arch}")


@contextmanager
def _matched_split_override(
    arch: str,
    forced_splits: int | None,
) -> Iterator[None]:
    if forced_splits is None:
        yield
        return

    if arch.startswith("gfx1250"):
        module_name = "tokenspeed_kernel_amd.ops.gfx1250.attention.mla.decode"
        selector_name = "_select_num_kv_splits"
    elif arch.startswith("gfx950"):
        if forced_splits != 16:
            raise ValueError(
                "gfx950 only supports --force-matched-splits 16: the projected "
                "path has no safe configurable split override"
            )
        module_name = "tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode"
        selector_name = "_select_num_kv_splits_bh16bn128_fp8"
    else:
        raise RuntimeError(f"expected gfx950 or gfx1250, detected {arch}")

    decode_module = importlib.import_module(module_name)
    with mock.patch.object(decode_module, selector_name, return_value=forced_splits):
        yield


@contextmanager
def _projected_split_override(
    arch: str,
    forced_splits: int | None,
) -> Iterator[None]:
    if forced_splits is None:
        yield
        return

    if arch.startswith("gfx1250"):
        module_name = "tokenspeed_kernel_amd.ops.gfx1250.attention.mla.decode"
        decode_module = importlib.import_module(module_name)
        original_selector = decode_module._select_num_kv_splits

        def select_num_kv_splits(*args, **kwargs) -> int:
            if kwargs.get("split_cap", 64) != 64:
                return forced_splits
            return original_selector(*args, **kwargs)

        patch = mock.patch.object(
            decode_module,
            "_select_num_kv_splits",
            new=select_num_kv_splits,
        )
    elif arch.startswith("gfx950"):
        module_name = "tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode"
        decode_module = importlib.import_module(module_name)
        selector_name = "_select_projected_value_num_kv_splits"
        if not hasattr(decode_module, selector_name):
            raise RuntimeError(
                "gfx950 --force-projected-splits requires experimental source "
                f"to expose {selector_name}"
            )
        patch = mock.patch.object(
            decode_module,
            selector_name,
            return_value=forced_splits,
        )
    else:
        raise RuntimeError(f"expected gfx950 or gfx1250, detected {arch}")

    with patch:
        yield


def _validate_forced_splits(
    forced_splits: int | None,
    *,
    option: str,
    seq_lens: list[int],
    page_size: int,
) -> None:
    if forced_splits is None:
        return
    if forced_splits <= 0:
        raise ValueError(f"{option} must be positive")
    infeasible = [
        seq_len
        for seq_len in seq_lens
        if forced_splits > math.ceil(seq_len / page_size)
    ]
    if infeasible:
        raise ValueError(
            f"{option} {forced_splits} exceeds the available "
            f"{page_size}-token pages for sequence lengths {infeasible}"
        )


def _run_case(
    *,
    arch: str,
    batch: int,
    seq_len: int,
    seed: int,
    forced_splits: int | None,
    forced_projected_splits: int | None,
    warmup: int,
    repeats: int,
    direct_candidates: bool,
    decode_atol: float,
) -> dict:
    print(
        f"CASE start batch={batch} sequence={seq_len} seed={seed} "
        f"forced_splits={forced_splits} "
        f"forced_projected_splits={forced_projected_splits}",
        flush=True,
    )
    torch.manual_seed(seed)
    overrides = _overrides(arch)
    if direct_candidates:
        if arch.startswith("gfx950"):
            from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
                gluon_mla_decode_projected_value_gfx950 as decode_candidate,
            )
            from tokenspeed_kernel_amd.ops.gfx950.attention.mla.project_value import (
                gluon_mla_project_value_gfx950 as project_candidate,
            )
        else:
            from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.decode import (
                gluon_mla_decode_projected_value_gfx1250 as decode_candidate,
            )
            from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.project_value import (
                gluon_mla_project_value_gfx1250 as project_candidate,
            )
    heads = 12
    latent = 512
    value = 128
    rope = 64
    page_size = 64
    pages_per_sequence = math.ceil(seq_len / page_size)
    num_pages = batch * pages_per_sequence

    q = torch.randn(
        batch,
        1,
        heads,
        latent + rope,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    kv_cache = torch.randn(
        num_pages,
        page_size,
        1,
        latent + rope,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    page_table = torch.arange(
        num_pages,
        device="cuda",
        dtype=torch.int32,
    ).reshape(batch, pages_per_sequence)
    cache_seqlens = torch.tensor(
        [seq_len - (index % 4) * page_size for index in range(batch)],
        device="cuda",
        dtype=torch.int32,
    )
    weight = torch.randn(
        heads,
        latent,
        value,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate_storage = torch.randn(
        batch,
        3648,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = gate_storage[:, -(heads * value) :]
    attention = torch.randn(
        batch,
        heads,
        latent,
        device="cuda",
        dtype=torch.bfloat16,
    )
    project_fused_out = torch.empty_like(gate)
    project_fallback_out = torch.empty_like(gate)
    latent_out = torch.empty(
        batch,
        1,
        heads,
        latent,
        device="cuda",
        dtype=torch.bfloat16,
    )
    decode_fused_out = torch.empty_like(gate)
    decode_composed_out = torch.empty_like(gate)

    decode_args = {
        "q": q,
        "kv_cache": kv_cache,
        "page_table": page_table,
        "cache_seqlens": cache_seqlens,
        "max_seqlen_k": seq_len,
        "qk_nope_head_dim": 128,
        "kv_lora_rank": latent,
        "qk_rope_head_dim": rope,
        "softmax_scale": 1.0 / math.sqrt(192),
    }

    def project_fused() -> torch.Tensor:
        if direct_candidates:
            return project_candidate(
                attention,
                weight,
                gate=gate,
                out=project_fused_out,
            )
        return mla_project_value(
            attention,
            weight,
            gate=gate,
            out=project_fused_out,
            override=overrides["project"],
        )

    def project_fallback() -> torch.Tensor:
        projected = torch.bmm(attention.transpose(0, 1).contiguous(), weight)
        project_fallback_out.view(batch, heads, value).copy_(
            projected.transpose(0, 1)
        )
        sigmoid_mul(project_fallback_out, gate)
        return project_fallback_out

    def decode_composed() -> torch.Tensor:
        mla_decode_with_kvcache(
            **decode_args,
            out=latent_out,
            override=overrides["decode"],
        )
        return mla_project_value(
            latent_out.reshape(batch, heads, latent),
            weight,
            gate=gate,
            out=decode_composed_out,
            override=overrides["project"],
        )

    def decode_fused() -> torch.Tensor:
        if direct_candidates:
            return decode_candidate(
                **decode_args,
                value_weight=weight,
                gate=gate,
                out=decode_fused_out,
            )
        return mla_decode_with_kvcache(
            **decode_args,
            value_weight=weight,
            gate=gate,
            out=decode_fused_out,
            override=overrides["decode_project"],
        )

    project_fused()
    project_fallback()
    decode_composed()
    decode_fused()
    torch.cuda.synchronize()
    decode_abs_error = (decode_fused_out.float() - decode_composed_out.float()).abs()
    relative_error_floor = 1.0e-6
    decode_rel_error = decode_abs_error / decode_composed_out.float().abs().clamp_min(
        relative_error_floor
    )
    production_close = torch.isclose(
        decode_fused_out,
        decode_composed_out,
        atol=_PROJECTED_DECODE_ATOL,
        rtol=0.05,
    )
    exact_equal = torch.eq(decode_fused_out, decode_composed_out)
    production_mismatches = int((~production_close).sum().item())
    max_absolute_error = float(decode_abs_error.max().item())
    max_relative_error = float(decode_rel_error.max().item())
    exact_equality_count = int(exact_equal.sum().item())
    print(
        f"CASE correctness batch={batch} sequence={seq_len} seed={seed} "
        f"production_mismatches={production_mismatches} "
        f"max_abs={max_absolute_error:.7g} max_rel={max_relative_error:.7g} "
        f"exact_equal={exact_equality_count}/{decode_fused_out.numel()} "
        f"diagnostic_atol={decode_atol}",
        flush=True,
    )
    torch.testing.assert_close(
        project_fused_out,
        project_fallback_out,
        atol=0.125,
        rtol=0.05,
    )
    torch.testing.assert_close(
        decode_fused_out,
        decode_composed_out,
        atol=decode_atol,
        rtol=0.05,
    )

    project_fused_timing = _measure_us(
        project_fused,
        warmup=warmup,
        repeats=repeats,
    )
    project_fallback_timing = _measure_us(
        project_fallback,
        warmup=warmup,
        repeats=repeats,
    )
    decode_fused_timing = _measure_us(
        decode_fused,
        warmup=warmup,
        repeats=repeats,
    )
    decode_composed_timing = _measure_us(
        decode_composed,
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "batch_size": batch,
        "sequence_length": seq_len,
        "seed": seed,
        "forced_split_count": forced_splits,
        "forced_projected_split_count": forced_projected_splits,
        "shape": {
            "q": list(q.shape),
            "kv_cache": list(kv_cache.shape),
            "value_weight": list(weight.shape),
            "gate": list(gate.shape),
            "gate_stride": list(gate.stride()),
        },
        "selection": {
            "project_value": (
                project_candidate.__name__
                if direct_candidates
                else _selected_kernel(project_fused)
            ),
            "decode_projected_value": (
                decode_candidate.__name__
                if direct_candidates
                else _selected_kernel(decode_fused)
            ),
        },
        "project_value": {
            "fused": project_fused_timing,
            "fallback": project_fallback_timing,
            "speedup": (
                project_fallback_timing["median_us"]
                / project_fused_timing["median_us"]
            ),
        },
        "decode_projected_value": {
            "correctness": {
                "diagnostic_atol": decode_atol,
                "diagnostic_rtol": 0.05,
                "production_atol": _PROJECTED_DECODE_ATOL,
                "production_rtol": 0.05,
                "production_mismatches": production_mismatches,
                "max_absolute_error": max_absolute_error,
                "max_relative_error": max_relative_error,
                "relative_error_denominator_floor": relative_error_floor,
                "exact_equality_count": exact_equality_count,
                "element_count": decode_fused_out.numel(),
            },
            "fused": decode_fused_timing,
            "composed": decode_composed_timing,
            "speedup": (
                decode_composed_timing["median_us"]
                / decode_fused_timing["median_us"]
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--seq-len", type=int, nargs="+", default=[4161])
    parser.add_argument(
        "--direct-candidates",
        action="store_true",
        help="measure candidate kernels even outside registered batch traits",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--decode-atol", type=float, default=_PROJECTED_DECODE_ATOL)
    parser.add_argument(
        "--seed",
        type=int,
        help="seed all random tensors (default preserves the existing per-batch seed)",
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--force-matched-splits",
        type=int,
        help="diagnostically force matching standalone and projected split counts",
    )
    split_group.add_argument(
        "--force-projected-splits",
        type=int,
        help="force only projected decode splits for performance tuning",
    )
    parser.add_argument("--tokenspeed-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    page_size = 64
    _validate_forced_splits(
        args.force_matched_splits,
        option="--force-matched-splits",
        seq_lens=args.seq_len,
        page_size=page_size,
    )
    _validate_forced_splits(
        args.force_projected_splits,
        option="--force-projected-splits",
        seq_lens=args.seq_len,
        page_size=page_size,
    )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("projected MLA benchmark requires exactly one GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    with (
        _matched_split_override(arch, args.force_matched_splits),
        _projected_split_override(arch, args.force_projected_splits),
    ):
        cases = [
            _run_case(
                arch=arch,
                batch=batch,
                seq_len=seq_len,
                seed=args.seed if args.seed is not None else 89 + batch,
                forced_splits=args.force_matched_splits,
                forced_projected_splits=args.force_projected_splits,
                warmup=args.warmup,
                repeats=args.repeats,
                direct_candidates=args.direct_candidates,
                decode_atol=args.decode_atol,
            )
            for batch in args.batch
            for seq_len in args.seq_len
        ]
    result = {
        "format": "tokenspeed_kimi_k3_mla_projected_amd_v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "architecture": arch,
        "tokenspeed_revision": args.tokenspeed_revision,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "forced_split_count": args.force_matched_splits,
        "forced_projected_split_count": args.force_projected_splits,
        "decode_atol": args.decode_atol,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
