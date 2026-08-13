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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from toy_e2e.run_kimi_k3_logical_rank import (
    LogicalRankCommBackend,
    _validate_cache_group_tables,
    cache_parent_page,
    logicalize_topk,
)


def test_logicalize_topk_masks_remote_experts() -> None:
    ids = torch.tensor([[4, 87, 150, 301], [111, 112, 0, 895]])
    weights = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.8, 0.7, 0.6, 0.5]])

    local_ids, local_weights, is_local = logicalize_topk(
        ids,
        weights,
        expert_start=0,
        num_local_experts=112,
    )

    assert local_ids.tolist() == [[4, 87, 0, 0], [111, 0, 0, 0]]
    torch.testing.assert_close(
        local_weights,
        torch.tensor([[0.4, 0.3, 0.0, 0.0], [0.8, 0.0, 0.6, 0.0]]),
    )
    assert is_local.tolist() == [
        [True, True, False, False],
        [True, False, True, False],
    ]


def test_logicalize_topk_offsets_nonzero_rank() -> None:
    ids = torch.tensor([[111, 112, 150, 223, 224]])
    weights = torch.ones_like(ids, dtype=torch.float32)

    local_ids, local_weights, _ = logicalize_topk(
        ids,
        weights,
        expert_start=112,
        num_local_experts=112,
    )

    assert local_ids.tolist() == [[0, 0, 38, 111, 0]]
    assert local_weights.tolist() == [[0.0, 1.0, 1.0, 1.0, 0.0]]


def test_logical_collectives_preserve_local_partials_and_expected_shapes() -> None:
    backend = LogicalRankCommBackend()
    value = torch.arange(6).reshape(2, 3)
    group = tuple(range(8))

    assert backend.all_reduce(value, group) is value
    gathered = backend.all_gather(value, group, dim=0)
    assert gathered.shape == (16, 3)
    for rank_copy in gathered.chunk(8):
        assert torch.equal(rank_copy, value)

    scattered = backend.reduce_scatter(gathered, group)
    assert torch.equal(scattered, value)


def _cache_pool():
    groups = (
        SimpleNamespace(group_id="full_attention", cache_blocks_per_lcm_block=12),
        SimpleNamespace(group_id="linear_attention_0", cache_blocks_per_lcm_block=1),
    )
    return SimpleNamespace(plan=SimpleNamespace(groups=groups))


def _forward_op(full_page: int, state_page: int):
    arrays = {
        "full_attention": np.array([[full_page]], dtype=np.int32),
        "linear_attention_0": np.array([[state_page]], dtype=np.int32),
    }
    return SimpleNamespace(block_tables_arrays=lambda: arrays)


def test_child_pages_map_to_shared_lcm_parents() -> None:
    assert cache_parent_page(child_page=1, cache_blocks_per_parent=12) == 1
    assert cache_parent_page(child_page=12, cache_blocks_per_parent=12) == 1
    assert cache_parent_page(child_page=13, cache_blocks_per_parent=12) == 2


def test_cache_groups_reject_aliasing_lcm_parent() -> None:
    with pytest.raises(ValueError, match="alias LCM parent 1"):
        _validate_cache_group_tables(_cache_pool(), _forward_op(12, 1))


def test_cache_groups_accept_scheduler_allocated_parents() -> None:
    tables = _validate_cache_group_tables(_cache_pool(), _forward_op(1, 2))

    assert tables == {
        "full_attention": [[1]],
        "linear_attention_0": [[2]],
    }
