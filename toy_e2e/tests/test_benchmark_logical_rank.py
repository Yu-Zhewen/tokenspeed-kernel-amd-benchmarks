import pytest

from toy_e2e.benchmark_logical_rank import _hotspot_summary, _summary


def test_summary_reports_common_p90_metric():
    result = _summary([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result["p50"] == 3.0
    assert result["p90"] == pytest.approx(4.6)


def test_hotspot_summary_sorts_modules_and_reports_model_share():
    result = _hotspot_summary(
        {
            ("decode", 4, "moe", "MoeBlock"): [4.0, 6.0],
            ("decode", 2, "mla_attention", "MlaAttention"): [1.0, 3.0],
            ("prefill", 1, "kda_attention", "KdaAttention"): [8.0],
        },
        {
            "decode": [10.0, 14.0],
            "prefill": [40.0],
        },
        top_k=1,
    )

    assert result["decode"]["model_p50_ms"] == 12.0
    top = result["decode"]["top_modules"][0]
    assert top["layer"] == 4
    assert top["category"] == "moe"
    assert top["module_type"] == "MoeBlock"
    assert top["timing_ms"]["p50"] == 5.0
    assert top["timing_ms"]["p95"] == pytest.approx(5.9)
    assert top["share_of_model_p50_pct"] == pytest.approx(100.0 * 5.0 / 12.0)
    assert result["prefill"]["top_modules"][0]["category"] == "kda_attention"
