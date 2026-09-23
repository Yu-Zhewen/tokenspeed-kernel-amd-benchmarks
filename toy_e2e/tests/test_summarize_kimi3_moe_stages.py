import json

import pytest

from toy_e2e.scripts.summarize_kimi3_moe_stages import summarize_trace


def test_summarize_attributes_kernels_to_innermost_moe_scopes(tmp_path):
    trace = tmp_path / "c16" / "prefill" / "toy-c16-TP0-EXTEND.trace.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": "tokenspeed::kimi3_moe.router",
                        "pid": 1,
                        "tid": 1,
                        "ts": 10,
                        "dur": 20,
                        "args": {"External id": 1},
                    },
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::mm",
                        "pid": 1,
                        "tid": 1,
                        "ts": 12,
                        "dur": 2,
                        "args": {"External id": 10},
                    },
                    {
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": "tokenspeed::kimi3_moe.routed_experts",
                        "pid": 1,
                        "tid": 1,
                        "ts": 40,
                        "dur": 20,
                        "args": {"External id": 2},
                    },
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "moe",
                        "pid": 1,
                        "tid": 1,
                        "ts": 42,
                        "dur": 2,
                        "args": {"External id": 20},
                    },
                    {
                        "ph": "X",
                        "cat": "cuda_runtime",
                        "name": "hipModuleLaunchKernel",
                        "pid": 1,
                        "tid": 1,
                        "ts": 45,
                        "dur": 1,
                        "args": {
                            "correlation": 42,
                            "kernel": "expert_b",
                        },
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "router_gemm",
                        "pid": 2,
                        "tid": 0,
                        "ts": 100,
                        "dur": 10,
                        "args": {"External id": 10, "stream": 0},
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "expert_a",
                        "pid": 2,
                        "tid": 0,
                        "ts": 200,
                        "dur": 20,
                        "args": {"External id": 20, "stream": 0},
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "expert_b",
                        "pid": 2,
                        "tid": 1,
                        "ts": 210,
                        "dur": 20,
                        "args": {"correlation": 42, "stream": 1},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_trace(trace, moe_layers=1)
    stages = {stage["stage"]: stage for stage in result["stages"]}

    assert result["inferred_forwards"] == 1
    assert stages["router"]["kernel_sum_ms"] == pytest.approx(0.01)
    assert stages["routed_experts"]["kernel_sum_ms"] == pytest.approx(0.04)
    assert stages["routed_experts"]["kernel_union_ms"] == pytest.approx(0.03)
    assert stages["routed_experts"]["kernel_overlap_ms"] == pytest.approx(0.01)
    assert stages["routed_experts"]["streams_ms"] == {"0": 0.02, "1": 0.02}
