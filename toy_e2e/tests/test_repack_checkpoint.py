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

from pathlib import Path

from toy_e2e.scripts.repack_checkpoint import (
    _encode_header,
    _read_local_header,
    _reduced_config,
    repack_safetensors,
)


def _write_fixture(path: Path) -> None:
    header = {
        "__metadata__": {"format": "pt"},
        "tensor_a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        "tensor_b": {"dtype": "U8", "shape": [6], "data_offsets": [4, 10]},
    }
    encoded = _encode_header(header)
    with path.open("wb") as output:
        output.write(len(encoded).to_bytes(8, "little"))
        output.write(encoded)
        output.write(b"abcd123456")


def test_repack_safetensors_copies_selected_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "destination.safetensors"
    _write_fixture(source)

    tensor_count, tensor_bytes = repack_safetensors(
        source,
        destination,
        {"tensor_b"},
    )

    assert tensor_count == 1
    assert tensor_bytes == 6
    header, data_start = _read_local_header(destination)
    assert set(header) == {"__metadata__", "tensor_b"}
    assert header["tensor_b"]["data_offsets"] == [0, 6]
    with destination.open("rb") as repacked:
        repacked.seek(data_start)
        assert repacked.read() == b"123456"


def test_reduced_config_keeps_first_attention_cycle() -> None:
    config = {
        "text_config": {
            "num_hidden_layers": 93,
            "linear_attn_config": {
                "kda_layers": [1, 2, 3, 5, 6, 7],
                "full_attn_layers": [4, 8],
            },
        }
    }

    reduced = _reduced_config(config, 4)

    assert reduced["text_config"]["num_hidden_layers"] == 4
    assert reduced["text_config"]["linear_attn_config"] == {
        "kda_layers": [1, 2, 3],
        "full_attn_layers": [4],
    }
    assert config["text_config"]["num_hidden_layers"] == 93
