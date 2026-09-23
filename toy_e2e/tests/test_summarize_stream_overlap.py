import json

import pytest

from toy_e2e.scripts.summarize_stream_overlap import summarize_trace


def test_summarize_separates_cross_and_same_stream_intervals(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "a",
                        "ts": 0,
                        "dur": 20,
                        "args": {"stream": 0},
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "b",
                        "ts": 5,
                        "dur": 10,
                        "args": {"stream": 0},
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "c",
                        "ts": 10,
                        "dur": 20,
                        "args": {"stream": 1},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_trace(trace, top_k=10)

    assert result["same_stream_pairwise_overlap_ms"] == pytest.approx(0.01)
    assert result["cross_stream_pairwise_overlap_ms"] == pytest.approx(0.015)
    assert result["stream_pairs"] == [
        {"pair": ["0", "1"], "overlap_ms": pytest.approx(0.015)}
    ]
