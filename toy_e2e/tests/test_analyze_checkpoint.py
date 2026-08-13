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

from toy_e2e.scripts.analyze_checkpoint import select_required_keys


def test_selects_early_layers_globals_and_local_experts() -> None:
    weight_map = {
        "language_model.model.embed_tokens.weight": "global",
        "language_model.model.norm.weight": "global",
        "language_model.model.layers.0.self_attn.q_proj.weight": "layer0",
        "language_model.model.layers.1.self_attn.q_proj.weight": "layer1",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight": (
            "layer1"
        ),
        "language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight": (
            "layer1"
        ),
        "language_model.model.layers.1.block_sparse_moe.experts.2.w1.weight": (
            "layer1"
        ),
        "language_model.model.layers.2.self_attn.q_proj.weight": "layer2",
        "vision_tower.layers.0.weight": "vision",
    }

    selected, expert_range = select_required_keys(
        weight_map,
        num_layers=2,
        num_experts=4,
        ep_size=2,
        ep_rank=0,
    )

    assert expert_range == (0, 2)
    assert selected == {
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.1.self_attn.q_proj.weight",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight",
        "language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight",
    }


def test_selects_requested_ep_rank() -> None:
    weight_map = {
        f"language_model.model.layers.1.block_sparse_moe.experts.{expert}.w1.weight": (
            "layer1"
        )
        for expert in range(8)
    }

    selected, expert_range = select_required_keys(
        weight_map,
        num_layers=2,
        num_experts=8,
        ep_size=4,
        ep_rank=2,
    )

    assert expert_range == (4, 6)
    assert selected == {
        "language_model.model.layers.1.block_sparse_moe.experts.4.w1.weight",
        "language_model.model.layers.1.block_sparse_moe.experts.5.w1.weight",
    }
