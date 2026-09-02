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

import torch

from toy_e2e.logical_rank import LogicalRankCommBackend


def test_logical_collectives_preserve_tp8_shapes_and_record_volume():
    backend = LogicalRankCommBackend()
    group = tuple(range(8))
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    reduced = backend.all_reduce(source, group)
    gathered = backend.all_gather(source, group, dim=0)
    scattered = backend.reduce_scatter(gathered, group)

    assert reduced is source
    assert gathered.shape == (16, 3)
    torch.testing.assert_close(scattered, source)

    events = backend.snapshot(reset=True)
    assert [event["operation"] for event in events] == [
        "all_reduce",
        "all_gather",
        "reduce_scatter",
    ]
    assert events[1]["group_size"] == 8
    assert events[1]["input_bytes"] == source.numel() * source.element_size()
    assert events[1]["output_bytes"] == gathered.numel() * gathered.element_size()
    assert backend.snapshot() == []


def test_logical_all_gather_into_tensor_fills_caller_buffer():
    backend = LogicalRankCommBackend()
    group = tuple(range(8))
    source = torch.arange(4, dtype=torch.int32).reshape(1, 4)
    output = torch.empty(8, 1, 4, dtype=torch.int32)

    backend.all_gather_into_tensor(output, source, group)

    torch.testing.assert_close(output, source.expand_as(output))
    event = backend.snapshot()[0]
    assert event["operation"] == "all_gather_into_tensor"
    assert event["output_bytes"] == output.numel() * output.element_size()
