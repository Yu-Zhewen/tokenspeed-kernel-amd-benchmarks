#!/usr/bin/env python3
"""Sweep wave32/wave64 AttnRes warp counts at Kimi-K3 exact shapes."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch


def _measure_us(fn, warmup: int, repeats: int) -> dict:
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


def _kernel_for_arch(arch: str):
    if arch.startswith("gfx950"):
        from tokenspeed_kernel_amd.ops.gfx950.attention.kda.attn_res import (
            _attn_res_rmsnorm_kernel,
        )

        return _attn_res_rmsnorm_kernel, False
    if arch.startswith("gfx1250"):
        from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.attn_res import (
            _attn_res_rmsnorm_kernel,
        )

        # The current gfx1250 kernel fixes load width and score mode internally.
        return _attn_res_rmsnorm_kernel, False
    raise RuntimeError(f"expected gfx950 or gfx1250, detected {arch}")


def _run_case(
    kernel,
    *,
    tokens: int,
    valid_blocks: int,
    has_delta: bool,
    num_warps: int,
    load_elems: int,
    has_load_elems_parameter: bool,
    score_mode: str,
    warmup: int,
    repeats: int,
) -> dict:
    hidden = 7168
    allocated_blocks = 8
    generator = torch.Generator(device="cuda").manual_seed(
        1_000 + tokens + valid_blocks
    )
    layer = torch.randn(
        tokens,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    delta = torch.zeros_like(layer)
    blocks_storage = torch.randn(
        allocated_blocks,
        tokens,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    blocks = blocks_storage.transpose(0, 1)
    res_weight = torch.randn(
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    score_weight = torch.randn(
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    output_weight = torch.randn(
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    output = torch.empty_like(layer)

    def launch(target: torch.Tensor, use_score_mode: str) -> torch.Tensor:
        launch_options = {
            "H": hidden,
            "N": valid_blocks + 1,
            "BLOCK_WRITE_IDX": 0,
            "HAS_DELTA": has_delta,
            "WRITE_BLOCK": False,
            "SCORE_EPS": 1e-6,
            "OUTPUT_EPS": 1e-6,
            "NUM_WARPS": num_warps,
            "num_warps": num_warps,
        }
        if has_load_elems_parameter:
            launch_options["LOAD_ELEMS"] = load_elems
            launch_options["FP64_SCORE"] = use_score_mode == "fp64"
            launch_options["MIXED_SCORE"] = use_score_mode == "mixed"
        kernel[(tokens,)](
            layer,
            delta,
            blocks,
            res_weight,
            score_weight,
            output_weight,
            target,
            stride_layer_t=layer.stride(0),
            stride_delta_t=delta.stride(0),
            stride_block_t=blocks.stride(0),
            stride_block_n=blocks.stride(1),
            stride_output_t=target.stride(0),
            **launch_options,
        )
        return target

    def run() -> torch.Tensor:
        return launch(output, score_mode)

    numerics = {
        "within_established_tolerance": True,
        "mismatched_elements": 0,
        "max_abs_difference": 0.0,
    }
    if has_load_elems_parameter and score_mode != "fp64":
        reference = torch.empty_like(output)
        launch(reference, "fp64")
        run()
        torch.cuda.synchronize()
        close = torch.isclose(output, reference, rtol=5e-3, atol=1.6e-2)
        numerics = {
            "within_established_tolerance": bool(close.all().item()),
            "mismatched_elements": int((~close).sum().item()),
            "max_abs_difference": float(
                (output.float() - reference.float()).abs().max().item()
            ),
        }

    timing = _measure_us(run, warmup, repeats)
    return {
        "tokens": tokens,
        "hidden_size": hidden,
        "valid_blocks": valid_blocks,
        "num_sources": valid_blocks + 1,
        "has_delta": has_delta,
        "num_warps": num_warps,
        "load_elems": load_elems,
        "score_dtype": score_mode,
        "block_strides": list(blocks.stride()),
        "numerics": numerics,
        "timing": timing,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 4096])
    parser.add_argument("--valid-blocks", type=int, nargs="+", default=[0, 3, 7])
    parser.add_argument("--num-warps", type=int, nargs="+", default=[4])
    parser.add_argument("--load-elems", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tokenspeed-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("AttnRes benchmark requires exactly one GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    kernel, has_load_elems_parameter = _kernel_for_arch(arch)
    load_elems_values = args.load_elems if has_load_elems_parameter else [2]
    score_modes = ("fp64", "mixed", "fp32") if has_load_elems_parameter else ("fp64",)
    cases = []
    for tokens in args.tokens:
        for valid_blocks in args.valid_blocks:
            for has_delta in (False, True):
                for num_warps in args.num_warps:
                    for load_elems in load_elems_values:
                        for score_mode in score_modes:
                            case = _run_case(
                                kernel,
                                tokens=tokens,
                                valid_blocks=valid_blocks,
                                has_delta=has_delta,
                                num_warps=num_warps,
                                load_elems=load_elems,
                                has_load_elems_parameter=has_load_elems_parameter,
                                score_mode=score_mode,
                                warmup=args.warmup,
                                repeats=args.repeats,
                            )
                            print(json.dumps(case, sort_keys=True), flush=True)
                            cases.append(case)
        torch.cuda.empty_cache()
    result = {
        "format": "tokenspeed_kimi_k3_attn_res_warps_amd_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "architecture": arch,
        "tokenspeed_revision": args.tokenspeed_revision,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
