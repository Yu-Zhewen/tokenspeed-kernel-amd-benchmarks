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
"""Run one logical Kimi-K3 TP8/EP8 rank without distributed peers."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def logicalize_topk(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    expert_start: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map global top-k IDs to one EP rank and zero remote contributions."""
    expert_end = expert_start + num_local_experts
    is_local = (topk_ids >= expert_start) & (topk_ids < expert_end)
    local_ids = torch.where(is_local, topk_ids - expert_start, 0)
    local_weights = torch.where(is_local, topk_weights, 0)
    return local_ids, local_weights, is_local


def cache_parent_page(child_page: int, cache_blocks_per_parent: int) -> int:
    """Map a positive group-local child page to its shared LCM parent."""
    if child_page <= 0 or cache_blocks_per_parent <= 0:
        raise ValueError("page and packing values must be positive")
    return (child_page - 1) // cache_blocks_per_parent + 1


class LogicalRankCommBackend:
    """Shape-preserving single-rank emulation of multi-rank collectives."""

    def all_reduce(self, tensor, group, op=None):
        del group, op
        if isinstance(tensor, tuple):
            return tuple(tensor)
        return tensor

    def prepare_all_reduce_lane(self, group, hidden_dim):
        del group, hidden_dim
        return False

    def acquire_all_reduce_outputs(self, shapes, like, group, op=None):
        del group, op
        return tuple(like.new_empty(shape) for shape in shapes)

    def all_gather(self, tensor, group, dim=0):
        return torch.cat([tensor] * len(group), dim=dim)

    def all_gather_into_tensor(self, output, input, group):
        gathered = self.all_gather(input, group, dim=0)
        output.copy_(gathered.reshape_as(output))

    def reduce_scatter(self, tensor, group):
        return tensor.chunk(len(group), dim=0)[0].contiguous()

    def all_to_all_single(self, output, input, group):
        del group
        output.copy_(input)

    def token_all_gather(self, tensor, group, scattered_num_tokens):
        del scattered_num_tokens
        return torch.cat([tensor] * len(group), dim=0)

    def token_reduce_scatter(self, tensor, group, scattered_num_tokens):
        del scattered_num_tokens
        return tensor.chunk(len(group), dim=0)[0].contiguous()


def _logical_execution_plan(kimi_k3):
    return kimi_k3.Kimi3MoEExecutionPlan(
        use_native=False,
        use_trtllm=False,
        overlap_shared_experts=False,
        joint_moe_reduce=False,
        use_marlin=True,
    )


@contextmanager
def logical_rank_runtime(stats: dict[str, Any]):
    """Install test-only communication and local-MoE adapters."""
    from tokenspeed.runtime.distributed.comm_backend import registry
    from tokenspeed.runtime.layers.attention.backends import mla as mla_backend
    from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
    from tokenspeed.runtime.models import deepseek_v3, kimi_k3

    original_backend = registry._global_backend
    original_moe_layer = kimi_k3.MoELayer
    original_routed_experts = kimi_k3.KimiLinearMoE._routed_experts
    original_reduce_attn = kimi_k3.KimiLinearDecoderLayer._reduce_attn_accumulate
    original_mla_projection = kimi_k3.KimiLinearMLAAttention._project_q_latent_gated
    original_absorb_attn = deepseek_v3.DeepseekV3AttentionMLA.forward_absorb_attn_v_proj
    original_mla_decode = mla_backend.mla_decode_with_kvcache

    def local_moe_layer(*args, **kwargs):
        global_experts = int(kwargs["num_experts"])
        ep_size = int(kwargs["ep_size"])
        if global_experts % ep_size:
            raise ValueError("experts must divide evenly across logical EP ranks")
        kwargs["num_experts"] = global_experts // ep_size
        kwargs["ep_rank"] = 0
        kwargs["ep_size"] = 1
        # Triton's SiTU implementation is the dynamic MXFP4-activation path;
        # its BF16-activation weight-only path supports SwiGLU only.
        kwargs["internal_activation_dtype_override"] = "mxfp4"
        layer = original_moe_layer(*args, **kwargs)
        layer.logical_global_num_experts = global_experts
        layer.logical_expert_start = 0
        return layer

    def local_routed_experts(
        self,
        routed_in,
        topk_output,
        num_global_tokens,
        max_num_tokens_per_gpu,
        skip_reduce=False,
    ):
        del skip_reduce
        local_ids, local_weights, is_local = logicalize_topk(
            topk_output.topk_ids,
            topk_output.topk_weights,
            expert_start=self.experts.logical_expert_start,
            num_local_experts=self.experts.num_experts,
        )
        stats["moe_topk_slots"] = stats.get("moe_topk_slots", 0) + is_local.numel()
        stats["moe_local_slots"] = stats.get("moe_local_slots", 0) + int(
            is_local.sum().item()
        )
        local_topk = StandardTopKOutput(
            local_weights,
            local_ids,
            topk_output.router_logits,
        )
        return self.experts(
            hidden_states=routed_in,
            topk_output=local_topk,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )

    def local_reduce_attn(self, attn_partial, prefix_sum, combine=None):
        del self, combine
        output = attn_partial if prefix_sum is None else prefix_sum + attn_partial
        return output, None

    def record_mla_projection(self, *args, **kwargs):
        outputs = original_mla_projection(self, *args, **kwargs)
        ctx = args[1] if len(args) > 1 else kwargs.get("ctx")
        phase = "decode" if ctx is not None and ctx.num_extends == 0 else "prefill"
        names = ("query", "latent_cache", "gate", "absorbed_query")
        stats.setdefault("mla_projection", {}).setdefault(phase, [])
        for name, tensor in zip(names, outputs):
            if tensor is None:
                continue
            finite = torch.isfinite(tensor)
            stats["mla_projection"][phase].append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "finite": bool(finite.all().item()),
                    "nonfinite": int((~finite).sum().item()),
                    "abs_max": float(
                        torch.nan_to_num(tensor.float()).abs().max().item()
                    ),
                }
            )
        return outputs

    def record_absorb_attn(self, q, k, *args, **kwargs):
        if q.shape[0] == 1:
            stats["mla_decode_kernel_input"] = {
                "q_dtype": str(q.dtype),
                "q_finite": bool(torch.isfinite(q.float()).all().item()),
                "q_abs_max": float(torch.nan_to_num(q.float()).abs().max().item()),
                "k_dtype": None if k is None else str(k.dtype),
                "k_finite": (
                    None if k is None else bool(torch.isfinite(k.float()).all().item())
                ),
                "k_abs_max": (
                    None
                    if k is None
                    else float(torch.nan_to_num(k.float()).abs().max().item())
                ),
            }
        return original_absorb_attn(self, q, k, *args, **kwargs)

    def record_mla_decode(*args, **kwargs):
        output = original_mla_decode(*args, **kwargs)
        q = kwargs.get("q", args[0] if args else None)
        cache = kwargs.get("kv_cache", args[1] if len(args) > 1 else None)
        page_table = kwargs.get("page_table", args[2] if len(args) > 2 else None)
        cache_seqlens = kwargs.get("cache_seqlens", args[3] if len(args) > 3 else None)
        kv_lora_rank = int(kwargs.get("kv_lora_rank", args[6] if len(args) > 6 else 0))
        softmax_scale = float(
            kwargs.get("softmax_scale", args[8] if len(args) > 8 else 1.0)
        )
        seq_len = int(cache_seqlens[0].item())
        page_size = int(cache.shape[1])
        pages = page_table[0, : (seq_len + page_size - 1) // page_size]
        keys = cache.index_select(0, pages.to(torch.int64)).reshape(
            -1, cache.shape[-2], cache.shape[-1]
        )[:seq_len, 0]
        query = q[0, 0].float()
        scores = torch.einsum("hd,td->ht", query, keys.float())
        scores.mul_(softmax_scale)
        probabilities = torch.softmax(scores, dim=-1)
        reference = torch.einsum(
            "ht,td->hd",
            probabilities,
            keys[:, :kv_lora_rank].float(),
        )
        stats["mla_decode_reference"] = {
            "kernel_finite": bool(torch.isfinite(output.float()).all().item()),
            "reference_finite": bool(torch.isfinite(reference).all().item()),
            "score_finite": bool(torch.isfinite(scores).all().item()),
            "score_abs_max": float(torch.nan_to_num(scores).abs().max().item()),
            "cache_finite": bool(torch.isfinite(keys.float()).all().item()),
            "cache_abs_max": float(keys.float().abs().max().item()),
            "cache_nonfinite_by_token": (~torch.isfinite(keys.float()))
            .sum(dim=-1)
            .tolist(),
        }
        return output

    registry._global_backend = LogicalRankCommBackend()
    try:
        with (
            mock.patch.object(kimi_k3, "MoELayer", side_effect=local_moe_layer),
            mock.patch.object(
                kimi_k3.Kimi3MoEExecutionPlan,
                "build",
                side_effect=lambda *args, **kwargs: _logical_execution_plan(kimi_k3),
            ),
            mock.patch.object(
                kimi_k3.KimiLinearMoE,
                "_routed_experts",
                new=local_routed_experts,
            ),
            mock.patch.object(
                kimi_k3.KimiLinearDecoderLayer,
                "_reduce_attn_accumulate",
                new=local_reduce_attn,
            ),
            mock.patch.object(
                kimi_k3.KimiLinearMLAAttention,
                "_project_q_latent_gated",
                new=record_mla_projection,
            ),
            mock.patch.object(
                deepseek_v3.DeepseekV3AttentionMLA,
                "forward_absorb_attn_v_proj",
                new=record_absorb_attn,
            ),
            mock.patch.object(
                mla_backend,
                "mla_decode_with_kvcache",
                new=record_mla_decode,
            ),
        ):
            yield
    finally:
        registry._global_backend = original_backend
        kimi_k3.MoELayer = original_moe_layer
        kimi_k3.KimiLinearMoE._routed_experts = original_routed_experts
        kimi_k3.KimiLinearDecoderLayer._reduce_attn_accumulate = original_reduce_attn
        kimi_k3.KimiLinearMLAAttention._project_q_latent_gated = original_mla_projection
        deepseek_v3.DeepseekV3AttentionMLA.forward_absorb_attn_v_proj = (
            original_absorb_attn
        )
        mla_backend.mla_decode_with_kvcache = original_mla_decode


def _server_args(checkpoint: Path):
    from tokenspeed.runtime.utils.server_args import ServerArgs

    server_args = ServerArgs(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        load_format="safetensors",
        language_model_only=True,
        attn_tp_size=8,
        ep_size=8,
        attention_backend="mla",
        kv_cache_dtype="fp8_e4m3",
        device="cuda",
        enforce_eager=True,
        disable_prefill_graph=True,
        disable_autotune=True,
        moe_backend="triton",
        enable_allreduce_fusion=False,
        comm_fusion_max_num_tokens=0,
        max_model_len=2048,
        max_num_seqs=1,
        chunked_prefill_size=128,
    )
    server_args.resolve_basic_defaults()
    server_args.resolve_parallelism()
    server_args.mapping.rank = 0
    return server_args


def load_logical_rank(checkpoint: Path):
    """Construct and load the real four-layer logical rank."""
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.execution.model_runner import ModelRunner

    server_args = _server_args(checkpoint)
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


def _model_summary(server_args, model_config, runner, stats):
    language_model = runner.model.language_model
    layers = language_model.model.layers
    moe_layers = [
        layer.block_sparse_moe for layer in layers if hasattr(layer, "block_sparse_moe")
    ]
    return {
        "phase": "load",
        "model_type": type(runner.model).__name__,
        "num_layers": len(layers),
        "layer_types": [type(layer.self_attn).__name__ for layer in layers],
        "attn_tp_size": server_args.mapping.attn.tp_size,
        "attn_tp_rank": server_args.mapping.attn.tp_rank,
        "moe_ep_size": server_args.mapping.moe.ep_size,
        "moe_ep_rank": server_args.mapping.moe.ep_rank,
        "local_kda_heads": [
            getattr(layer.self_attn, "local_num_heads", None)
            for layer in layers
            if type(layer.self_attn).__name__ == "KimiLinearKDA"
        ],
        "local_mla_heads": [
            getattr(layer.self_attn, "num_local_heads", None)
            for layer in layers
            if type(layer.self_attn).__name__ == "KimiLinearMLAAttention"
        ],
        "local_experts": [layer.experts.num_experts for layer in moe_layers],
        "moe_solutions": [layer.experts.plan["solution"] for layer in moe_layers],
        "dtype": str(model_config.dtype),
        "allocated_gib": torch.cuda.memory_allocated() / (1 << 30),
        "reserved_gib": torch.cuda.memory_reserved() / (1 << 30),
        **stats,
    }


def _validate_cache_group_tables(pool, forward_op) -> dict[str, list[list[int]]]:
    """Reject scheduler tables whose live groups alias one LCM parent."""
    arrays = forward_op.block_tables_arrays()
    packing = {
        str(group.group_id): int(group.cache_blocks_per_lcm_block)
        for group in pool.plan.groups
    }
    if set(arrays) != set(packing):
        raise ValueError(
            "scheduler cache groups do not match the runtime plan: "
            f"{sorted(arrays)} != {sorted(packing)}"
        )

    parent_owners: dict[int, str] = {}
    serialized: dict[str, list[list[int]]] = {}
    for group_id, array in arrays.items():
        serialized[group_id] = array.tolist()
        for child_page in set(int(page) for page in array.flat if page > 0):
            parent_page = cache_parent_page(child_page, packing[group_id])
            owner = parent_owners.setdefault(parent_page, group_id)
            if owner != group_id:
                raise ValueError(
                    f"cache groups {owner!r} and {group_id!r} alias "
                    f"LCM parent {parent_page}"
                )
    return serialized


def _scheduler_forward_ops(pool, prefill_tokens: int):
    """Allocate one request's prefill/decode cache tables with the real scheduler."""
    from tokenspeed.runtime.engine.scheduler_utils import (
        make_spec,
        pool_to_paged_cache_groups,
        scheduler_cache_geometry_from_pool,
    )
    from tokenspeed_scheduler import Scheduler, SchedulerConfig

    geometry = scheduler_cache_geometry_from_pool(
        pool,
        fallback_token_capacity=pool.size,
        fallback_page_size=pool.page_size,
    )
    # Use the scheduler binding's stable core fields directly. This keeps the
    # harness compatible with validation images whose compiled scheduler may
    # predate optional runtime-helper fields such as prefix_replay_tokens.
    config = SchedulerConfig()
    config.num_device_pages = geometry.num_device_pages
    config.max_scheduled_tokens = max(prefill_tokens, geometry.page_size)
    config.max_batch_size = 1
    config.block_size = geometry.page_size
    config.num_host_pages = 0
    config.disable_l2_cache = True
    config.enable_l3_storage = False
    config.disable_prefix_cache = True
    config.paged_cache_groups = pool_to_paged_cache_groups(pool)
    scheduler = Scheduler(config)
    scheduler.submit_requests(
        [make_spec("logical-rank-0", [1] * prefill_tokens, max_new_tokens=1)]
    )
    prefill_op = scheduler.next_execution_plan().forward[0]
    decode_op = scheduler.next_execution_plan().forward[0]
    if prefill_op.num_extends() != 1 or decode_op.num_extends() != 0:
        raise RuntimeError("scheduler did not produce prefill followed by decode")
    return prefill_op, decode_op


def _cache_metadata(pool, forward_op):
    from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
        CacheBatchMetadata,
    )

    _validate_cache_group_tables(pool, forward_op)
    metadata = CacheBatchMetadata.from_forward_op(
        forward_op,
        device="cuda",
        contract=pool.runtime_contract,
        num_requests=1,
    )
    return metadata


def _init_forward(
    backend,
    pool,
    forward_op,
    *,
    seq_len: int,
    num_tokens: int,
    is_decode: bool,
):
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    mode = ForwardMode.DECODE if is_decode else ForwardMode.EXTEND
    num_extends = 0 if is_decode else 1
    req_pool_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    metadata = _cache_metadata(pool, forward_op)
    tables = dict(metadata.tables(active_forward_op=forward_op))
    page_table = tables.get("full_attention")
    if page_table is None:
        page_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    extend_seq_lens = torch.tensor(
        [] if is_decode else [num_tokens],
        dtype=torch.int32,
        device="cuda",
    )
    extend_prefix_lens = torch.tensor(
        [] if is_decode else [seq_len - num_tokens],
        dtype=torch.int32,
        device="cuda",
    )
    backend.init_forward_metadata(
        bs=1,
        num_extends=num_extends,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        page_table=page_table,
        forward_mode=mode,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=extend_seq_lens.cpu(),
        extend_prefix_lens=extend_prefix_lens,
        extend_prefix_lens_cpu=extend_prefix_lens.cpu(),
        cache_metadata=metadata,
        forward_batch=forward_op,
        num_tokens=num_tokens,
    )
    return ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=pool,
        bs=1,
        num_extends=num_extends,
        input_num_tokens=num_tokens,
        forward_mode=mode,
        global_num_tokens=[num_tokens] * 8,
        global_bs=[1] * 8,
        all_decode_or_idle=is_decode,
        all_extend=not is_decode,
    )


def run_prefill_decode(
    server_args,
    model_config,
    runner,
    stats,
    prefill_tokens,
    mla_decode_mode,
    mla_kernel_solution,
):
    """Execute the four-layer backbone once for prefill and once for decode."""
    import tokenspeed_kernel.profiling as kernel_profiling
    from tokenspeed.runtime.layers.attention import registry as attention_registry

    with mock.patch.object(
        attention_registry,
        "profile_available_cache_memory_bytes",
        return_value=512 << 20,
    ):
        backend, pool, _, _, capacity, storage = (
            attention_registry.create_attn_components(
                server_args,
                model_config,
                gpu_id=0,
                rank=0,
                gpu_memory=432,
            )
        )
    if mla_decode_mode == "composed":
        backend.full_attn_backend.supports_mla_projected_value_decode = False
    if mla_kernel_solution != "auto":
        backend.full_attn_backend.kernel_solution = mla_kernel_solution
    stats["mla_decode_mode"] = mla_decode_mode
    stats["mla_kernel_solution"] = mla_kernel_solution
    pool.clear_kv_buffers()
    model = runner.model.language_model.model
    prefill_op, decode_op = _scheduler_forward_ops(pool, prefill_tokens)
    stats["scheduler_cache_tables"] = {
        "prefill": _validate_cache_group_tables(pool, prefill_op),
        "decode": _validate_cache_group_tables(pool, decode_op),
    }
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    active_phase = {"name": "setup"}
    hooks = []

    def first_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            return next(
                (
                    tensor
                    for item in value
                    if (tensor := first_tensor(item)) is not None
                ),
                None,
            )
        return None

    def record_output(name):
        def hook(module, args, output):
            del module, args
            tensor = first_tensor(output)
            if tensor is None:
                return
            finite = torch.isfinite(tensor)
            diagnostics.setdefault(active_phase["name"], []).append(
                {
                    "module": name,
                    "shape": list(tensor.shape),
                    "finite": bool(finite.all().item()),
                    "nonfinite": int((~finite).sum().item()),
                }
            )

        return hook

    for index, layer in enumerate(model.layers):
        hooks.append(
            layer.self_attn.register_forward_hook(
                record_output(f"layer{index}.attention")
            )
        )
        ffn = (
            layer.block_sparse_moe if hasattr(layer, "block_sparse_moe") else layer.mlp
        )
        hooks.append(ffn.register_forward_hook(record_output(f"layer{index}.ffn")))
        hooks.append(layer.register_forward_hook(record_output(f"layer{index}.output")))

    def execute(*, phase, num_tokens, seq_len, is_decode, forward_op):
        active_phase["name"] = phase
        ctx = _init_forward(
            backend,
            pool,
            forward_op,
            seq_len=seq_len,
            num_tokens=num_tokens,
            is_decode=is_decode,
        )
        if is_decode:
            metadata = backend.full_attn_backend.forward_decode_metadata
            stats["mla_decode_metadata"] = {
                "page_table": metadata.page_table[0, :4].tolist(),
                "seq_lens": metadata.seq_lens.tolist(),
                "out_cache_loc": metadata.group_out_cache_loc.tolist(),
                "kernel_page_size": int(backend.full_attn_backend.page_size),
            }
        input_ids = torch.ones(num_tokens, dtype=torch.int64, device="cuda")
        positions = torch.arange(
            seq_len - num_tokens,
            seq_len,
            dtype=torch.int64,
            device="cuda",
        )
        out_cache_loc = backend.select_out_cache_loc(
            model.layers[-1].self_attn.attn_mqa,
            torch.zeros(num_tokens, dtype=torch.int64, device="cuda"),
            ctx.forward_mode,
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        hidden, _ = model(
            input_ids=input_ids,
            positions=positions,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
        )
        end.record()
        torch.cuda.synchronize()
        return hidden, float(start.elapsed_time(end))

    capture_path = Path("/tmp/kimi_k3_logical_rank_shapes.json")
    kernel_profiling.start_shape_capture()
    try:
        prefill_hidden, prefill_ms = execute(
            phase="prefill",
            num_tokens=prefill_tokens,
            seq_len=prefill_tokens,
            is_decode=False,
            forward_op=prefill_op,
        )
    except BaseException:
        kernel_profiling.stop_shape_capture(capture_path)
        raise
    mla_layer = model.layers[-1].self_attn
    prefill_cache_locs = (
        backend.full_attn_backend.forward_prefill_metadata.group_out_cache_loc
    )
    live_cache = (
        pool.get_key_buffer(len(model.layers) - 1)
        .index_select(0, prefill_cache_locs.to(torch.int64))
        .float()
    )
    stats["mla_prefill_cache"] = {
        "shape": list(live_cache.shape),
        "finite": bool(torch.isfinite(live_cache).all().item()),
        "nonfinite": int((~torch.isfinite(live_cache)).sum().item()),
        "abs_max": float(torch.nan_to_num(live_cache).abs().max().item()),
        "nonzero": int(torch.count_nonzero(live_cache).item()),
    }
    stats["mla_absorbed_weights"] = {
        "w_kc_finite": bool(torch.isfinite(mla_layer.w_kc).all().item()),
        "w_vc_finite": bool(torch.isfinite(mla_layer.w_vc).all().item()),
        "w_kc_abs_max": float(mla_layer.w_kc.abs().max().item()),
        "w_vc_abs_max": float(mla_layer.w_vc.abs().max().item()),
    }
    try:
        decode_hidden, decode_ms = execute(
            phase="decode",
            num_tokens=1,
            seq_len=prefill_tokens + 1,
            is_decode=True,
            forward_op=decode_op,
        )
    finally:
        kernel_profiling.stop_shape_capture(capture_path)
    stats["kernel_shapes"] = json.loads(capture_path.read_text(encoding="utf-8"))
    for hook in hooks:
        hook.remove()
    stats.update(
        {
            "phase": "prefill-decode",
            "prefill_tokens": prefill_tokens,
            "prefill_ms_including_jit": prefill_ms,
            "decode_ms_including_jit": decode_ms,
            "prefill_output_shape": list(prefill_hidden.shape),
            "decode_output_shape": list(decode_hidden.shape),
            "prefill_output_finite": bool(torch.isfinite(prefill_hidden).all().item()),
            "decode_output_finite": bool(torch.isfinite(decode_hidden).all().item()),
            "diagnostics": diagnostics,
            "cache_token_capacity": capacity,
            "cache_storage": storage,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("load", "prefill-decode"), default="load")
    parser.add_argument("--prefill-tokens", type=int, default=8)
    parser.add_argument(
        "--mla-decode-mode",
        choices=("composed", "projected"),
        default="composed",
    )
    parser.add_argument(
        "--mla-kernel-solution",
        choices=("auto", "triton"),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)

    stats: dict[str, Any] = {}
    torch.cuda.set_device(0)
    try:
        with logical_rank_runtime(stats):
            server_args, model_config, runner = load_logical_rank(args.checkpoint)
            if args.phase == "prefill-decode":
                run_prefill_decode(
                    server_args,
                    model_config,
                    runner,
                    stats,
                    args.prefill_tokens,
                    args.mla_decode_mode,
                    args.mla_kernel_solution,
                )
            result = _model_summary(server_args, model_config, runner, stats)
            result["status"] = (
                "failed_nonfinite"
                if args.phase == "prefill-decode"
                and (
                    not result["prefill_output_finite"]
                    or not result["decode_output_finite"]
                )
                else "passed"
            )
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            if args.output is not None:
                args.output.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            del runner
    finally:
        torch.cuda.empty_cache()
    return 2 if result["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
