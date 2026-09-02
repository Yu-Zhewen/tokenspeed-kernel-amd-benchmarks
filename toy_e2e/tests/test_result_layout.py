import csv
import json
import re
from pathlib import Path

import pytest


RESULTS = Path(__file__).resolve().parents[1] / "results"
TOY_ROOT = RESULTS.parent
APPROVED_DIRECTORIES = {
    "gfx950_toy_1gpu_0b1061eb",
    "gfx950_real_8gpu_0b1061eb",
    "gfx1250_toy_1gpu_pending",
}
REQUIRED_HEADINGS = [
    "## Status",
    "## Software and hardware setup",
    "## Workload and topology",
    "## Correctness",
    "## Unprofiled performance",
    "## Stage hotspot summary",
    "## Top exact kernels",
    "## Incomplete or failed work",
    "## Exact commands",
    "## Raw artifacts",
    "## Conclusions and limitations",
]
LOCAL_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
EXACT_KERNEL_ROW = re.compile(
    r"^\| (?P<concurrency>1|16) "
    r"\| (?P<stage>prefill|decode) "
    r"\| (?P<order>\d+) "
    r"\| `(?P<name>.+)` "
    r"\| (?P<calls>[\d,]+) "
    r"\| (?P<total_ms>[\d,.]+) ms "
    r"\| (?P<gpu_time_pct>[\d.]+)% "
    r"\| (?P<avg_us>[\d,.]+) µs \|$"
)


def test_result_index_contains_only_approved_targets():
    directories = {path.name for path in RESULTS.iterdir() if path.is_dir()}

    assert directories == APPROVED_DIRECTORIES


def test_result_readmes_share_the_template_headings():
    template = (TOY_ROOT / "RESULT_TEMPLATE.md").read_text().splitlines()
    assert [line for line in template if line.startswith("## ")] == REQUIRED_HEADINGS

    for directory in sorted(APPROVED_DIRECTORIES):
        lines = (RESULTS / directory / "README.md").read_text().splitlines()
        headings = [line for line in lines if line.startswith("## ")]
        assert headings == REQUIRED_HEADINGS


def test_toy_markdown_local_links_resolve():
    for markdown in TOY_ROOT.rglob("*.md"):
        for target in LOCAL_LINK.findall(markdown.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            assert (markdown.parent / path).exists(), f"{markdown}: {target}"


@pytest.mark.parametrize(
    ("directory", "rank_count"),
    [
        ("gfx950_toy_1gpu_0b1061eb", 1),
        ("gfx950_real_8gpu_0b1061eb", 8),
    ],
)
def test_collected_results_have_complete_common_profile_contract(
    directory, rank_count
):
    root = RESULTS / directory
    result = json.loads((root / "result.json").read_text())
    hotspots = json.loads((root / "hotspots" / "hotspots.json").read_text())

    assert result["status"] == "passed"
    assert result["hardware"]["architecture"].startswith("gfx950")
    assert result["hardware"]["gpu_count"] == rank_count
    profiles = hotspots["profiles"]
    assert len(profiles) == 4
    assert {
        (profile["setting"], profile["stage"], profile["rank_count"])
        for profile in profiles
    } == {
        ("c1", "EXTEND", rank_count),
        ("c1", "DECODE", rank_count),
        ("c16", "EXTEND", rank_count),
        ("c16", "DECODE", rank_count),
    }
    assert {path.name for path in (root / "hotspots" / "csv").iterdir()} == {
        "c1_extend.csv",
        "c1_decode.csv",
        "c16_extend.csv",
        "c16_decode.csv",
    }


@pytest.mark.parametrize(
    "directory",
    [
        "gfx950_toy_1gpu_0b1061eb",
        "gfx950_real_8gpu_0b1061eb",
    ],
)
def test_collected_readmes_show_exact_top_ten_kernels(directory):
    root = RESULTS / directory
    rows = {}
    for line in (root / "README.md").read_text().splitlines():
        match = EXACT_KERNEL_ROW.match(line)
        if match is None:
            continue
        key = (match["concurrency"], match["stage"])
        rows.setdefault(key, []).append(match.groupdict())

    for concurrency in ("1", "16"):
        for stage, csv_stage in (("prefill", "extend"), ("decode", "decode")):
            key = (concurrency, stage)
            reported = rows[key]
            with (
                root / "hotspots" / "csv" / f"c{concurrency}_{csv_stage}.csv"
            ).open(newline="") as handle:
                expected = list(csv.DictReader(handle))[:10]

            assert [int(row["order"]) for row in reported] == list(range(1, 11))
            assert [row["name"] for row in reported] == [
                row["kernel_name"] for row in expected
            ]
            assert [int(row["calls"].replace(",", "")) for row in reported] == [
                int(row["calls"]) for row in expected
            ]
            for field in ("total_ms", "gpu_time_pct", "avg_us"):
                assert [
                    float(row[field].replace(",", "")) for row in reported
                ] == pytest.approx(
                    [float(row[field]) for row in expected],
                    abs=0.005,
                )
