"""Cover the cross-architecture generator on synthetic inputs.

The generator only reads JSON, so this needs no GPU. It checks the parts that
would silently produce a misleading document: the bucket map, the direction of
every ratio, and that the per-stage CSVs keep every symbol.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "generate_arch_comparison.py"
)


def _performance(arch: str, *, decode_tps: float, ttft: float) -> dict:
    def run(concurrency):
        return {
            "benchmark": {
                "concurrency": concurrency,
                "time_to_first_token_ms": {"p50": ttft},
                "step_wall_ms": {"prefill": {"p50": ttft / 4}},
                "steady_decode_step_ms": {"p50": 1000.0 / decode_tps},
                "per_user_decode_tps": decode_tps,
                "aggregate_output_tps": decode_tps * concurrency,
            }
        }

    return {
        "device": f"{arch} device",
        "hardware": {"device": f"{arch} device", "architecture": arch},
        "model": {
            "model_type": "KimiK3ForConditionalGeneration",
            "num_layers": 93,
            "attn_tp_size": 8,
            "moe_tp_size": 8,
            "moe_ep_size": 1,
        },
        "software": {
            "model_revision": "modelsha",
            "container_image": f"image-{arch}@sha256:abc",
            "hip": "7.2",
            "pytorch": "2.13",
            "triton": "3.7",
        },
        "workload": {
            "prompt_tokens": 50000,
            "output_tokens": 1024,
            "concurrencies": [1, 16],
            "chunked_prefill_size": 8192,
            "warmup_waves": 1,
            "measurement_waves": 3,
        },
        "runs": [run(1), run(16)],
    }


def _hotspots(kernels_by_stage: dict) -> dict:
    return {
        "profiles": [
            {
                "stage": stage,
                "setting": setting,
                "all_kernels": [
                    {"name": n, "total_ms": ms, "calls": c}
                    for n, ms, c in kernels
                ],
            }
            for (stage, setting), kernels in kernels_by_stage.items()
        ]
    }


@pytest.fixture
def generated(tmp_path):
    stages = [("EXTEND", "c16"), ("EXTEND", "c1"), ("DECODE", "c16"), ("DECODE", "c1")]
    # gfx950 splits the MoE into staged kernels plus a reduce; gfx1250 uses two
    # matmuls plus a separate weighted reduce. Same job, different split, and
    # the reason the generator buckets instead of pairing names.
    hs950 = _hotspots({
        s: [
            ("gluon_mxfp4_moe_stage1_kernel.kd", 300.0, 10),
            ("gluon_mxfp4_moe_stage2_reduce_kernel.kd", 100.0, 10),
            ("gluon_mm_a16w16_prefill_gfx950.kd", 200.0, 20),
            ("gluon_kda_paged_prefill_state_scan_gfx950.kd", 100.0, 5),
            ("some_unmapped_kernel.kd", 7.0, 1),
        ]
        for s in stages
    })
    hs1250 = _hotspots({
        s: [
            ("_matmul.kd", 200.0, 20),
            ("_weighted_topk_reduce_gfx1250_kernel.kd", 50.0, 10),
            ("_wmma_tdm_dense_largem_kernel.kd", 100.0, 20),
            ("gluon_kda_paged_prefill_state_scan_gfx1250.kd", 100.0, 5),
            ("some_unmapped_kernel.kd", 3.0, 1),
        ]
        for s in stages
    })

    paths = {}
    for name, doc in (
        ("p950", _performance("gfx950", decode_tps=80.0, ttft=3000.0)),
        ("p1250", _performance("gfx1250", decode_tps=100.0, ttft=2000.0)),
        ("h950", hs950),
        ("h1250", hs1250),
    ):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(doc))

    out = tmp_path / "entry"
    subprocess.run(
        [
            sys.executable, str(SCRIPT), "--revision", "a" * 40,
            "--gfx950-performance", str(paths["p950"]),
            "--gfx950-hotspots", str(paths["h950"]),
            "--gfx1250-performance", str(paths["p1250"]),
            "--gfx1250-hotspots", str(paths["h1250"]),
            "--output-dir", str(out), "--top-kernels", "3",
        ],
        check=True,
        capture_output=True,
    )
    return out, (out / "README.md").read_text()


def test_records_provenance(generated):
    _, text = generated
    assert "a" * 40 in text
    assert "modelsha" in text
    assert "50,000 / 1,024" in text


def test_ratios_favour_the_faster_architecture(generated):
    _, text = generated
    # gfx1250 is given 1.5x the decode throughput and 1.5x lower TTFT, so both
    # a latency row and a throughput row must report the same 1.50x advantage.
    assert text.count("| 1.50x |") >= 2


def test_buckets_the_moe_as_one_job(generated):
    _, text = generated
    # 400 ms of gfx950 MoE against 250 ms of gfx1250 MoE is 1.60x. Pairing by
    # name would instead compare the gfx1250 reduce against nothing.
    moe_row = next(line for line in text.splitlines() if line.startswith("| MoE |"))
    assert "1.60x" in moe_row


def test_flat_bucket_reports_no_advantage(generated):
    _, text = generated
    scan = next(
        line for line in text.splitlines() if line.startswith("| KDA state scan |")
    )
    assert "1.00x" in scan


def test_unmapped_kernels_fall_into_other(generated):
    _, text = generated
    assert any(line.startswith("| other |") for line in text.splitlines())


def test_kernel_tables_are_capped_but_csvs_are_complete(generated):
    out, text = generated
    # --top-kernels 3 was requested, so rank 4 must not appear in markdown.
    assert "| 3 | " in text
    assert "| 4 | " not in text
    for arch, expected in (("gfx950", 5), ("gfx1250", 5)):
        for stage in ("c1-prefill", "c16-prefill", "c1-decode", "c16-decode"):
            path = out / "kernel-tables" / arch / f"{stage}.csv"
            rows = list(csv.DictReader(path.open()))
            assert len(rows) == expected
            assert {"category", "kernel_name", "calls", "total_ms"} <= set(rows[0])
