import csv
import json

from toy_e2e.scripts.summarize_gpu_hotspots import _category, _write_csvs, summarize


def _write_trace(path, durations):
    path.write_text(
        json.dumps(
            {
                "displayTimeUnit": "ms",
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": name,
                        "dur": duration,
                    }
                    for name, duration in durations
                ],
            }
        )
    )


def test_summarize_gpu_hotspots_aggregates_across_tp_ranks(tmp_path):
    trace_dir = tmp_path / "c1" / "decode"
    trace_dir.mkdir(parents=True)
    rank0 = trace_dir / "real-c1-TP0-DECODE.trace.json"
    rank1 = trace_dir / "real-c1-TP1-DECODE.trace.json"
    _write_trace(rank0, [("moe.fused[gluon]", 4000), ("mla.decode[gluon]", 1000)])
    _write_trace(rank1, [("moe.fused[gluon]", 6000), ("mla.decode[gluon]", 2000)])

    result = summarize([rank0, rank1], top_k=1)
    profile = result["profiles"][0]

    assert profile["source"] == "torch"
    assert profile["setting"] == "c1"
    assert profile["profile_id"] == "real-c1"
    assert profile["stage"] == "DECODE"
    assert profile["rank_count"] == 2
    assert profile["rank_kernel_ms"] == {
        "total": 13.0,
        "min": 5.0,
        "mean": 6.5,
        "max": 8.0,
        "imbalance_pct": 100.0 * 3.0 / 6.5,
    }
    assert profile["top_categories"][0]["name"] == "moe"
    assert profile["top_categories"][0]["total_ms"] == 10.0
    assert profile["top_categories"][0]["gpu_time_pct"] == 100.0 * 10.0 / 13.0
    assert profile["top_kernels"][0]["calls"] == 2
    assert profile["top_kernels"][0]["avg_us"] == 5000.0

    csv_dir = tmp_path / "csv"
    _write_csvs(result, csv_dir)
    with (csv_dir / "c1_decode.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["kernel_name"] == "moe.fused[gluon]"
    assert rows[0]["calls"] == "2"
    assert rows[0]["gpu_time_pct"] == "76.923077"


def test_category_recognizes_generated_gemm_names():
    assert _category("_rowcta_gemv_add3_kernel") == "gemm_or_quant"
    assert _category("Cijk_Alik_Bljk_generated_solution") == "gemm_or_quant"
    assert _category("_fp8_quantize_kernel") == "gemm_or_quant"


def test_category_keeps_moe_latent_projections_out_of_attention():
    assert _category("_packed_input_projections_kernel") == "moe"
    assert _category("_packed_projection_gemm_kernel") == "moe"
    assert _category("_latent_input_decode_kernel") == "moe"


def test_summarize_reports_cross_stream_kernel_overlap(tmp_path):
    trace_dir = tmp_path / "c16" / "prefill"
    trace_dir.mkdir(parents=True)
    trace = trace_dir / "real-c16-TP0-EXTEND.trace.json"
    trace.write_text(
        json.dumps(
            {
                "displayTimeUnit": "ms",
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "kernel_a",
                        "ts": 1000,
                        "dur": 2000,
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "kernel_b",
                        "ts": 2000,
                        "dur": 2000,
                    },
                ],
            }
        )
    )

    profile = summarize([trace], top_k=5)["profiles"][0]

    assert profile["rank_kernel_ms"]["mean"] == 4.0
    assert profile["rank_kernel_union_ms"]["mean"] == 3.0
    assert profile["rank_kernel_overlap_ms"]["mean"] == 1.0
    assert profile["rank_kernel_overlap_ms"]["mean_pct_of_kernel_sum"] == 25.0
