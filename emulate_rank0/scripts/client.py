#!/usr/bin/env python3
"""Drive `tokenspeed serve` with synthetic token-id prompts.

Request i of a run uses random.Random(seed + i) over [0, vocab), the same
prompts as toy_e2e/workload.py. Each concurrency level runs closed-loop waves
of equal-length requests with ignore_eos, so every request decodes exactly
--output-tokens tokens. `profile` opens the prefill and decode windows that
serve_bench.sh captures with rocprofv3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import aiohttp


def synthetic_prompt(length: int, seed: int, request_index: int, vocab: int):
    generator = random.Random(seed + request_index)
    return [generator.randrange(vocab) for _ in range(length)]


def summary(values):
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


async def complete(session, args, prompt, max_tokens, stream):
    body = {
        "model": args.model,
        "input_ids": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": stream,
    }
    url = f"{args.base_url}/generate"
    started = time.perf_counter()
    first = None
    meta = {}
    chunks = 0
    async with session.post(url, json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        if not stream:
            payload = await resp.json()
            if isinstance(payload, list):
                payload = payload[0]
            meta = payload.get("meta_info") or {}
            return {
                "latency_s": time.perf_counter() - started,
                "usage": {
                    "prompt_tokens": meta.get("prompt_tokens"),
                    "completion_tokens": meta.get("completion_tokens"),
                },
                "finish_reason": meta.get("finish_reason"),
            }
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            meta = obj.get("meta_info") or meta
            if obj.get("text") or obj.get("output_ids"):
                first = time.perf_counter() if first is None else first
                chunks += 1
    return {
        "started": started,
        "first": first,
        "finished": time.perf_counter(),
        "chunks": chunks,
        "usage": {
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
        },
    }


async def smoke(args):
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for index in range(2):
            prompt = synthetic_prompt(128, args.seed, index, args.vocab)
            result = await complete(session, args, prompt, 8, stream=False)
            print(f"smoke request {index}: {json.dumps(result)}", flush=True)
            usage = result["usage"] or {}
            if usage.get("prompt_tokens") != 128 or usage.get("completion_tokens") != 8:
                raise RuntimeError(f"unexpected usage {usage}")


async def run_phase(session, args, concurrency, offset, waves):
    results = []
    for wave in range(waves):
        base = offset + wave * concurrency
        prompts = [
            synthetic_prompt(args.prompt_tokens, args.seed, base + i, args.vocab)
            for i in range(concurrency)
        ]
        wave_started = time.perf_counter()
        wave_results = await asyncio.gather(
            *(
                complete(session, args, prompt, args.output_tokens, stream=True)
                for prompt in prompts
            )
        )
        wave_ms = (time.perf_counter() - wave_started) * 1e3
        for result in wave_results:
            usage = result["usage"] or {}
            if usage.get("completion_tokens") != args.output_tokens:
                raise RuntimeError(f"short completion: {usage}")
            result["wave_ms"] = wave_ms
        results.extend(wave_results)
    return results


async def bench(args):
    timeout = aiohttp.ClientTimeout(total=3600)
    report = {
        "config": vars(args),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phases": [],
    }
    offset = 0
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for concurrency in args.concurrency:
            await run_phase(session, args, concurrency, offset, args.warmup_waves)
            offset += concurrency * args.warmup_waves
            measured = await run_phase(
                session, args, concurrency, offset, args.measurement_waves
            )
            offset += concurrency * args.measurement_waves
            ttft = [(r["first"] - r["started"]) * 1e3 for r in measured]
            tpot = [
                (r["finished"] - r["first"]) * 1e3 / (args.output_tokens - 1)
                for r in measured
            ]
            wave_ms = sorted({r["wave_ms"] for r in measured})
            output_tokens = concurrency * args.output_tokens
            phase = {
                "concurrency": concurrency,
                "requests": len(measured),
                "time_to_first_token_ms": summary(ttft),
                "time_per_output_token_ms": summary(tpot),
                "output_tokens_per_s": summary(
                    [output_tokens * 1e3 / ms for ms in wave_ms]
                ),
                "wave_ms": summary(wave_ms),
                "stream_chunks": summary([r["chunks"] for r in measured]),
            }
            print(json.dumps(phase), flush=True)
            report["phases"].append(phase)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)


async def start_profile(session, args, output_dir, num_steps=None):
    body = {
        "output_dir": output_dir,
        "activities": ["CUDA_PROFILER"],
        "with_stack": False,
        "record_shapes": False,
    }
    if num_steps is not None:
        body["num_steps"] = num_steps
    for _ in range(300):
        async with session.post(f"{args.base_url}/start_profile", json=body) as resp:
            text = await resp.text()
            if resp.status == 200:
                return
            if resp.status < 500:
                raise RuntimeError(f"start_profile: {resp.status} {text}")
        await asyncio.sleep(1)
    raise RuntimeError(f"start_profile kept failing: {resp.status} {text}")


async def stop_profile(session, args):
    async with session.post(f"{args.base_url}/stop_profile", json={}) as resp:
        if resp.status != 200:
            raise RuntimeError(f"stop_profile: {resp.status} {await resp.text()}")


async def profile(args):
    """Open one rocprofv3 window per stage and concurrency.

    The window's output directory names the roctx range, which
    rocprof_to_traces.py uses to split kernels into stages.
    """
    timeout = aiohttp.ClientTimeout(total=3600)
    root = Path(args.profile_dir)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for concurrency in args.concurrency:
            prompts = [
                synthetic_prompt(args.prompt_tokens, args.seed, i, args.vocab)
                for i in range(concurrency)
            ]
            # Warm this batch shape before either capture.
            await asyncio.gather(
                *(complete(session, args, p, 2, stream=False) for p in prompts)
            )

            # One output token each: the window holds exactly this wave's
            # prefill forwards, however arrival splits the first chunk.
            prefill_dir = root / f"c{concurrency}" / "prefill"
            prefill_dir.mkdir(parents=True, exist_ok=True)
            await start_profile(session, args, str(prefill_dir))
            await asyncio.gather(
                *(complete(session, args, p, 1, stream=False) for p in prompts)
            )
            await stop_profile(session, args)

            decode_dir = root / f"c{concurrency}" / "decode"
            decode_dir.mkdir(parents=True, exist_ok=True)
            started = [asyncio.Event() for _ in prompts]

            async def decode_request(prompt, event):
                body = {
                    "model": args.model,
                    "input_ids": prompt,
                    "sampling_params": {
                        "max_new_tokens": args.decode_steps * 8,
                        "temperature": 0.0,
                        "ignore_eos": True,
                    },
                    "stream": True,
                }
                async with session.post(f"{args.base_url}/generate", json=body) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
                    async for raw in resp.content:
                        if raw.startswith(b"data:"):
                            event.set()

            tasks = [
                asyncio.create_task(decode_request(p, e))
                for p, e in zip(prompts, started)
            ]
            # Profile only once every request is decoding, so the window
            # holds no prefill forwards.
            await asyncio.gather(*(e.wait() for e in started))
            await start_profile(session, args, str(decode_dir), args.decode_steps)
            await asyncio.gather(*tasks)
            print(f"profiled C{concurrency}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "bench", "profile"))
    parser.add_argument("--profile-dir", default="profile")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=50_000)
    parser.add_argument("--output-tokens", type=int, default=500)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--warmup-waves", type=int, default=1)
    parser.add_argument("--measurement-waves", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vocab", type=int, required=True)
    parser.add_argument("--output", default="serve_bench.json")
    args = parser.parse_args()
    run = {"smoke": smoke, "bench": bench, "profile": profile}[args.mode]
    asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
