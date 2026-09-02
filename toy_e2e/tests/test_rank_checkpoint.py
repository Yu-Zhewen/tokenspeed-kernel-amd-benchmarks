# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import torch

from toy_e2e.rank_checkpoint import (
    MANIFEST_NAME,
    ExportOptions,
    RawRankStateLoader,
    export_raw_rank_state,
    process_weights_after_loading,
)


class _TinyModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4, 4), float(value)))
        self.register_buffer("scale", torch.full((4,), float(value)))


class _PostLoadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def post_load_weights(self) -> None:
        self.events.append("post_load")

    def process_weights_after_loading(self, module) -> None:
        assert module is self
        self.events.append("process")


def _source_dir(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["TinyModel"],
                "num_hidden_layers": 2,
                "num_experts": 8,
            }
        ),
        encoding="utf-8",
    )
    return source


def test_export_raw_rank_state_writes_bounded_parts_and_manifest(tmp_path):
    output = tmp_path / "output"
    manifest = export_raw_rank_state(
        _TinyModel(3),
        ExportOptions(
            output_dir=output,
            source_dir=_source_dir(tmp_path),
            max_part_bytes=64,
        ),
    )

    assert manifest["format"] == "tokenspeed_raw_rank_state_v1"
    assert manifest["complete"] is True
    assert manifest["tp_size"] == 8
    assert manifest["ep_size"] == 1
    assert manifest["tensor_count"] == 2
    assert len(manifest["parts"]) == 2
    assert sum(part["bytes"] for part in manifest["parts"]) == 80
    assert (output / MANIFEST_NAME).is_file()
    assert all((output / part["filename"]).is_file() for part in manifest["parts"])


def test_raw_rank_state_loader_restores_before_processing(tmp_path):
    output = tmp_path / "output"
    export_raw_rank_state(
        _TinyModel(7),
        ExportOptions(
            output_dir=output,
            source_dir=_source_dir(tmp_path),
            max_part_bytes=64,
        ),
    )
    initialized = _TinyModel(0)
    loader = RawRankStateLoader(SimpleNamespace())
    model_config = SimpleNamespace(model_path=str(output), dtype=torch.float32)
    device_config = SimpleNamespace(device="cpu")

    with (
        mock.patch(
            "tokenspeed.runtime.model_loader.loader._initialize_model",
            return_value=initialized,
        ),
        mock.patch(
            "toy_e2e.rank_checkpoint.process_weights_after_loading"
        ) as process,
    ):
        loaded = loader.load_model(
            model_config=model_config,
            device_config=device_config,
        )

    assert loaded is initialized
    torch.testing.assert_close(loaded.weight, torch.full((4, 4), 7.0))
    torch.testing.assert_close(loaded.scale, torch.full((4,), 7.0))
    process.assert_called_once_with(initialized, torch.device("cpu"))


def test_post_load_derived_weights_are_rebuilt_before_processing():
    model = _PostLoadModel()

    process_weights_after_loading(model, torch.device("cpu"))

    assert model.events == ["post_load", "process"]
