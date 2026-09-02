#!/usr/bin/env python3
"""Capture stage-separated TokenSpeed traces with an exact token-ID workload."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _post(url: str, payload: dict[str, Any], timeout_s: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc


def _prompt(length: int, seed: int, vocabulary_size: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(vocabulary_size) for _ in range(length)]


def _generate(
    *,
    service_url: str,
    model: str,
    prompt: list[int],
    output_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    result = _post(
        f"{service_url.rstrip('/')}/generate",
        {
            "model": model,
            "input_ids": prompt,
            "sampling_params": {
                "max_new_tokens": output_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout_s,
    )
    if not isinstance(result, list) or len(result) != 1:
        raise RuntimeError(f"unexpected /generate response: {result!r}")
    return result[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = {path.name for path in args.output_dir.iterdir()}
    output_tokens = args.profile_steps + 2 if args.capture == "decode" else 1
    profile_request = {
        "output_dir": str(args.output_dir),
        "activities": [args.activity],
        "profile_by_stage": args.capture == "decode",
        "with_stack": False,
        "record_shapes": False,
        "profile_id": args.profile_id,
    }
    if args.capture == "decode":
        profile_request["num_steps"] = args.profile_steps
    profile_response = _post(
        f"{args.control_url.rstrip('/')}/start_profile",
        profile_request,
        args.timeout_s,
    )

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                _generate,
                service_url=args.service_url,
                model=args.model,
                prompt=_prompt(
                    args.prompt_tokens,
                    args.seed + request_index,
                    args.vocabulary_size,
                ),
                output_tokens=output_tokens,
                timeout_s=args.timeout_s,
            )
            for request_index in range(args.concurrency)
        ]
        responses = [future.result() for future in futures]
    stop_response = None
    if args.capture == "prefill":
        stop_response = _post(
            f"{args.control_url.rstrip('/')}/stop_profile",
            {},
            args.timeout_s,
        )
    wall_s = time.perf_counter() - started

    request_results = []
    for request_index, response in enumerate(responses):
        meta = response["meta_info"]
        if meta["prompt_tokens"] != args.prompt_tokens:
            raise RuntimeError(
                f"request {request_index} reported {meta['prompt_tokens']} "
                f"prompt tokens; expected {args.prompt_tokens}"
            )
        if meta["completion_tokens"] != output_tokens:
            raise RuntimeError(
                f"request {request_index} reported {meta['completion_tokens']} "
                f"completion tokens; expected {output_tokens}"
            )
        request_results.append(
            {
                "request_index": request_index,
                "prompt_tokens": meta["prompt_tokens"],
                "completion_tokens": meta["completion_tokens"],
                "server_reported_e2e_latency_s": meta["e2e_latency"],
                "finish_reason": meta["finish_reason"],
            }
        )

    new_traces = sorted(
        path.name
        for path in args.output_dir.iterdir()
        if path.name not in existing_files and ".trace.json" in path.name
    )
    discarded_traces = []
    if args.capture == "decode":
        # Stage profiling is armed from the first EXTEND predicate, after the
        # first prefill forward has already run. Its EXTEND files therefore
        # contain only transition bookkeeping and must not be reported as a
        # prefill profile. Capture prefill separately with --capture prefill.
        discarded_traces = [
            trace for trace in new_traces if "-EXTEND.trace.json" in trace
        ]
        for trace in discarded_traces:
            (args.output_dir / trace).unlink()
        traces = [
            trace for trace in new_traces if "-DECODE.trace.json" in trace
        ]
    else:
        traces = new_traces
    expected_trace_count = args.tp_size
    if len(traces) != expected_trace_count:
        raise RuntimeError(
            f"found {len(traces)} traces, expected {expected_trace_count}: {traces}"
        )

    result = {
        "schema_version": 1,
        "profile_request": profile_request,
        "profile_response": profile_response,
        "stop_profile_response": stop_response,
        "workload": {
            "model": args.model,
            "prompt_tokens": args.prompt_tokens,
            "completion_tokens": output_tokens,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "vocabulary_size": args.vocabulary_size,
            "sampling": "greedy",
            "ignore_eos": True,
            "wall_s": wall_s,
        },
        "requests": request_results,
        "traces": traces,
        "discarded_transition_traces": discarded_traces,
    }
    manifest = args.output_dir / "profile_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://127.0.0.1:21000")
    parser.add_argument("--control-url", default="http://127.0.0.1:21001")
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument(
        "--capture",
        choices=("decode", "prefill"),
        default="decode",
        help="capture first-N decode batches, or one immediately armed prefill",
    )
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--profile-steps", type=int, default=64)
    parser.add_argument("--activity", choices=("GPU", "PROTON"), default="GPU")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--vocabulary-size", type=int, default=160000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-s", type=float, default=900)
    return parser.parse_args()


def main() -> None:
    result = run(_parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
