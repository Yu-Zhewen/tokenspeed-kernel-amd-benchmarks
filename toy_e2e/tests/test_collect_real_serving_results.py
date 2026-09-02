import json

from toy_e2e.scripts.collect_real_serving_results import collect


def _write_run(root, concurrency):
    run = root / f"c{concurrency}" / "kimi-k3" / f"parallel_{concurrency}"
    run.mkdir(parents=True)
    payloads = {
        "benchmark_summary.json": {
            "Concurrency": concurrency,
            "Total Requests": concurrency,
            "Success Requests": concurrency,
            "Avg Input Tokens": 4096.0,
            "Avg Output Tokens": 1024.0,
            "Avg Latency (s)": 13.0,
            "TTFT (ms)": 400.0,
            "Output Throughput (tok/s)": 80.0 * concurrency,
        },
        "benchmark_percentile.json": [
            {
                "Percentiles": percentile,
                "TTFT (ms)": 400.0,
                "TPOT (ms)": 12.0,
            }
            for percentile in ("50%", "90%")
        ],
        "workload_throughput.json": {
            "rows": [
                {
                    "metric": "Completion tok/s",
                    "overall": 80.0 * concurrency,
                    "steady_state": 90.0 * concurrency,
                }
            ]
        },
        "benchmark_args.json": {"warmup_num": concurrency},
    }
    for name, payload in payloads.items():
        (run / name).write_text(json.dumps(payload))


def test_collect_real_serving_results_normalizes_evalscope_outputs(tmp_path):
    _write_run(tmp_path, 1)
    _write_run(tmp_path, 16)

    result = collect(
        tmp_path,
        gpu_count=8,
        device="MI355X",
        architecture="gfx950",
        tokenspeed_revision="abc",
        model_revision="def",
    )

    assert result["status"] == "passed"
    assert result["topology"]["physical_ranks"] == 8
    assert [run["concurrency"] for run in result["runs"]] == [1, 16]
    assert result["runs"][0]["derived"]["overall_output_tps_per_gpu"] == 10.0
    assert result["runs"][1]["derived"]["steady_output_tps_per_gpu"] == 180.0
