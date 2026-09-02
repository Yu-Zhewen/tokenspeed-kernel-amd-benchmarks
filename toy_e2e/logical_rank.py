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
"""One physical GPU emulation of Kimi-K3 TP8/EP1 rank 0."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import torch


@dataclass(frozen=True)
class CollectiveEvent:
    operation: str
    group_size: int
    input_bytes: int
    output_bytes: int


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


class LogicalRankCommBackend:
    """Shape-correct local substitutes plus a collective-volume trace."""

    def __init__(self) -> None:
        self.events: list[CollectiveEvent] = []

    def _record(self, operation: str, group, source, output) -> None:
        self.events.append(
            CollectiveEvent(
                operation=operation,
                group_size=len(group),
                input_bytes=_tensor_bytes(source),
                output_bytes=_tensor_bytes(output),
            )
        )

    def all_reduce(self, tensor, group, op=None):
        del op
        output = tuple(tensor) if isinstance(tensor, tuple) else tensor
        self._record("all_reduce", group, tensor, output)
        return output

    def grouped_all_reduce(self, tensors, group, op=None):
        del op
        output = tuple(tensors)
        self._record("grouped_all_reduce", group, tensors, output)
        return output

    def prepare_all_reduce_lane(self, group, hidden_dim):
        del group, hidden_dim
        return False

    def can_acquire_all_reduce_outputs(self, shapes, like, group, op=None):
        del shapes, like, group, op
        return False

    def acquire_all_reduce_outputs(self, shapes, like, group, op=None):
        del op
        output = tuple(like.new_empty(shape) for shape in shapes)
        self._record("acquire_all_reduce_outputs", group, like, output)
        return output

    def all_gather(self, tensor, group, dim=0):
        output = torch.cat([tensor] * len(group), dim=dim)
        self._record("all_gather", group, tensor, output)
        return output

    def all_gather_into_tensor(self, output, input, group):
        gathered = torch.cat([input] * len(group), dim=0)
        output.copy_(gathered.reshape_as(output))
        self._record("all_gather_into_tensor", group, input, output)

    def reduce_scatter(self, tensor, group, op=None):
        del op
        output = tensor.chunk(len(group), dim=0)[0].contiguous()
        self._record("reduce_scatter", group, tensor, output)
        return output

    def reduce_scatter_tensor(self, output, input, group, op=None):
        del op
        output.copy_(input.chunk(len(group), dim=0)[0].reshape_as(output))
        self._record("reduce_scatter_tensor", group, input, output)

    def all_to_all_single(self, output, input, group, **kwargs):
        del kwargs
        output.copy_(input)
        self._record("all_to_all_single", group, input, output)

    def token_all_gather(self, tensor, group, scattered_num_tokens):
        del scattered_num_tokens
        output = torch.cat([tensor] * len(group), dim=0)
        self._record("token_all_gather", group, tensor, output)
        return output

    def token_reduce_scatter(self, tensor, group, scattered_num_tokens):
        del scattered_num_tokens
        output = tensor.chunk(len(group), dim=0)[0].contiguous()
        self._record("token_reduce_scatter", group, tensor, output)
        return output

    def snapshot(self, *, reset: bool = False) -> list[dict[str, Any]]:
        result = [asdict(event) for event in self.events]
        if reset:
            self.events.clear()
        return result


@contextmanager
def logical_rank_runtime():
    """Install the local collective backend for a TP8/EP1 rank-0 process."""
    from tokenspeed.runtime.distributed.comm_backend import registry
    from tokenspeed.runtime.models import kimi_k3

    backend = LogicalRankCommBackend()

    def local_reduce_attn(self, attn_partial, prefix_sum, combine=None):
        # K3AttnComm has backend-owned fusion paths that assume a live RCCL
        # communicator.  The logical rank needs only the rank-local partial.
        del combine
        reduced = backend.all_reduce(attn_partial, self.mapping.attn.tp_group)
        output = reduced if prefix_sum is None else prefix_sum + reduced
        return output, None

    original_backend = registry._global_backend
    registry._global_backend = backend
    try:
        with mock.patch.object(
            kimi_k3.KimiLinearDecoderLayer,
            "_reduce_attn_accumulate",
            new=local_reduce_attn,
        ):
            yield backend
    finally:
        registry._global_backend = original_backend


def build_server_args(
    checkpoint: Path,
    *,
    load_format: str | type = "safetensors",
    max_model_len: int = 8192,
    max_num_seqs: int = 16,
    chunked_prefill_size: int = 8192,
    enforce_eager: bool = True,
):
    from tokenspeed.runtime.utils.server_args import ServerArgs

    server_args = ServerArgs(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        load_format="safetensors",
        language_model_only=True,
        attn_tp_size=8,
        dense_tp_size=8,
        moe_tp_size=8,
        ep_size=1,
        attention_backend="mla",
        kv_cache_dtype="fp8_e4m3",
        device="cuda",
        enforce_eager=enforce_eager,
        disable_prefill_graph=True,
        disable_autotune=True,
        moe_backend="auto",
        enable_allreduce_fusion=False,
        comm_fusion_max_num_tokens=0,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        chunked_prefill_size=chunked_prefill_size,
        disable_kvstore=True,
        kvstore_ratio=0.0,
        enable_prefix_caching=False,
    )
    server_args.resolve_basic_defaults()
    server_args.resolve_parallelism()
    server_args.mapping.rank = 0
    server_args.enable_allreduce_fusion = False
    # Custom rank-state loaders are Python classes and must be installed only
    # after ServerArgs has completed string-valued format resolution.
    server_args.load_format = load_format
    return server_args


def load_logical_rank(
    checkpoint: Path,
    *,
    load_format: str | type = "safetensors",
    max_model_len: int = 8192,
    max_num_seqs: int = 16,
    chunked_prefill_size: int = 8192,
):
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.execution.model_runner import ModelRunner

    server_args = build_server_args(
        checkpoint,
        load_format=load_format,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        chunked_prefill_size=chunked_prefill_size,
    )
    model_config = ModelConfig(
        str(checkpoint),
        trust_remote_code=True,
        context_length=server_args.max_model_len,
        model_override_args="{}",
        dtype=server_args.dtype,
        quantization=server_args.quantization,
        server_args=server_args,
    )
    runner = ModelRunner(
        model_config=model_config,
        server_args=server_args,
        gpu_id=0,
        global_rank=0,
    )
    return server_args, model_config, runner


def model_summary(server_args, runner) -> dict[str, Any]:
    language_model = runner.model.language_model
    layers = language_model.model.layers
    moe_layers = [
        layer.block_sparse_moe for layer in layers if hasattr(layer, "block_sparse_moe")
    ]
    return {
        "model_type": type(runner.model).__name__,
        "num_layers": len(layers),
        "attn_tp_size": server_args.mapping.attn.tp_size,
        "attn_tp_rank": server_args.mapping.attn.tp_rank,
        "moe_tp_size": server_args.mapping.moe.tp_size,
        "moe_tp_rank": server_args.mapping.moe.tp_rank,
        "moe_ep_size": server_args.mapping.moe.ep_size,
        "moe_ep_rank": server_args.mapping.moe.ep_rank,
        "local_experts": sorted(
            {int(layer.experts.num_local_experts) for layer in moe_layers}
        ),
        "moe_solutions": sorted(
            {str(layer.experts.plan["solution"]) for layer in moe_layers}
        ),
        "allocated_gib": torch.cuda.memory_allocated() / (1 << 30),
        "reserved_gib": torch.cuda.memory_reserved() / (1 << 30),
    }
