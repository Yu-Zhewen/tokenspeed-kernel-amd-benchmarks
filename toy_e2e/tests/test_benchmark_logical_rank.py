import ast
import inspect

import pytest

from toy_e2e.benchmark_logical_rank import (
    _context_checkpoints,
    _context_sample_summary,
    _hotspot_summary,
    _run_rolling_phase,
    _summary,
)


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


def test_context_checkpoints_cover_rolling_decode_without_false_5120_forward():
    checkpoints = _context_checkpoints(4096, 1024)

    assert checkpoints == [
        4097,
        4224,
        4352,
        4480,
        4608,
        4736,
        4864,
        4992,
        5119,
    ]
    assert 5120 not in checkpoints


def test_context_sample_summary_reports_resulting_context():
    result = _context_sample_summary(
        {
            4097: [10.0, 12.0, 14.0],
            5119: [20.0, 22.0, 24.0],
        },
        [4097, 4608, 5119],
    )

    assert [sample["decode_input_tokens"] for sample in result] == [4097, 5119]
    assert result[-1]["resulting_context_tokens"] == 5120
    assert result[0]["step_wall_ms"]["p50"] == 12.0


def test_rolling_driver_uses_depth_one_without_full_device_sync():
    source = inspect.getsource(_run_rolling_phase)
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
    }

    assert "effective_depth = 1 if forward_op is not None else 0" in source
    assert "torch.cuda.synchronize" not in calls
