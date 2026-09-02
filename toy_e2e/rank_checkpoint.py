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
"""Portable raw rank-state checkpoint support for the Kimi-K3 estimator.

TokenSpeed's built-in ``sharded_state`` format stores weights *after* kernel
specific preprocessing.  Kimi-K3's MXFP4 expert layout differs between gfx950
and gfx1250, so such a checkpoint is not portable between those architectures.

This module stores the initialized model state after TP slicing but before
``process_weights_after_loading``.  The loader restores that raw state first,
then lets the target machine perform its own architecture-specific
preprocessing.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

FORMAT_NAME = "tokenspeed_raw_rank_state_v1"
MANIFEST_NAME = "rank-local-manifest.json"
DEFAULT_PATTERN = "model-rank-{rank}-part-{part:05d}.safetensors"

_METADATA_FILES = (
    "LICENSE",
    "README.md",
    "config.json",
    "configuration_kimi_k3.py",
    "encoding_k3.py",
    "generation_config.json",
    "kimi_k3_processor.py",
    "kimi_k3_vision_processing.py",
    "media_utils.py",
    "modeling_kimi_k3.py",
    "modeling_kimi_linear.py",
    "preprocessor_config.json",
    "tiktoken.model",
    "tokenization_kimi.py",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class ExportOptions:
    output_dir: Path
    source_dir: Path
    tp_size: int = 8
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    max_part_bytes: int = 2 << 30
    overwrite: bool = False
    tokenspeed_revision: str | None = None
    source_revision: str | None = None
    process_after_export: bool = False

    def validate(self) -> None:
        if not self.source_dir.is_dir():
            raise FileNotFoundError(self.source_dir)
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("invalid TP size/rank")
        if self.ep_size <= 0 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("invalid EP size/rank")
        if self.max_part_bytes <= 0:
            raise ValueError("max_part_bytes must be positive")


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_output(options: ExportOptions) -> None:
    options.validate()
    output_dir = options.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        if not options.overwrite:
            raise RuntimeError(f"output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in _METADATA_FILES:
        source = options.source_dir / filename
        if source.is_file():
            shutil.copy2(source, output_dir / filename)


def _portable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    from tokenspeed.runtime.model_loader.loader import ShardedStateLoader

    # Match the built-in sharded-state alias filtering so tied/subtensor state
    # does not make safetensors reject a part for shared storage.
    return dict(ShardedStateLoader._filter_subtensors(model.state_dict()))


def export_raw_rank_state(
    model: torch.nn.Module,
    options: ExportOptions,
) -> dict[str, Any]:
    """Write a TP-sliced, pre-kernel-processing model state in bounded parts."""
    from safetensors.torch import save_file

    _prepare_output(options)
    state = _portable_state_dict(model)
    descriptions: dict[str, dict[str, Any]] = {}
    parts: list[dict[str, Any]] = []
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0
    part_number = 0

    def flush() -> None:
        nonlocal part_number, pending, pending_bytes
        if not pending:
            return
        filename = DEFAULT_PATTERN.format(rank=options.tp_rank, part=part_number)
        destination = options.output_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        save_file(
            pending,
            str(temporary),
            metadata={
                "format": "pt",
                "checkpoint_format": FORMAT_NAME,
                "rank": str(options.tp_rank),
            },
        )
        temporary.replace(destination)
        parts.append(
            {
                "filename": filename,
                "bytes": pending_bytes,
                "tensor_count": len(pending),
            }
        )
        print(
            f"Wrote {filename}: {len(pending)} tensors, "
            f"{pending_bytes / (1 << 30):.2f} GiB",
            flush=True,
        )
        part_number += 1
        pending = {}
        pending_bytes = 0

    for name in sorted(state):
        source = state[name].detach()
        size = tensor_nbytes(source)
        if pending and pending_bytes + size > options.max_part_bytes:
            flush()
        host_tensor = source.to(device="cpu", copy=True).contiguous()
        pending[name] = host_tensor
        pending_bytes += size
        descriptions[name] = {
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "bytes": size,
            "part": part_number,
        }
    flush()

    total_bytes = sum(item["bytes"] for item in descriptions.values())
    config_path = options.output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    manifest = {
        "complete": True,
        "format": FORMAT_NAME,
        "architecture": config.get("architectures", []),
        "num_hidden_layers": int(text_config["num_hidden_layers"]),
        "num_experts": int(text_config.get("num_experts") or 0),
        "tp_size": options.tp_size,
        "tp_rank": options.tp_rank,
        "ep_size": options.ep_size,
        "ep_rank": options.ep_rank,
        "raw_state_before_kernel_preprocessing": True,
        "architecture_portable": ["gfx950", "gfx1250"],
        "tokenspeed_revision": options.tokenspeed_revision,
        "source_revision": options.source_revision,
        "tensor_count": len(descriptions),
        "tensor_bytes": total_bytes,
        "parts": parts,
        "tensors": descriptions,
    }
    _write_json(options.output_dir / MANIFEST_NAME, manifest)
    return manifest


def process_weights_after_loading(
    model: torch.nn.Module,
    target_device: torch.device,
) -> None:
    """Mirror DefaultModelLoader's architecture-specific post-load phase."""
    from tokenspeed.runtime.model_loader.loader import device_loading_context

    # Kimi's normal checkpoint loader calls this from model.load_weights() to
    # build unregistered derived tensors (fused KDA convolution weights,
    # absorbed MLA weights, and AttnRes products). A state-dict restore bypasses
    # model.load_weights(), so rebuild those tensors explicitly.
    post_load_weights = getattr(model, "post_load_weights", None)
    if callable(post_load_weights):
        post_load_weights()

    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

        process_method = getattr(module, "process_weights_after_loading", None)
        if process_method is not None:
            with device_loading_context(module, target_device):
                process_method(module)

    post_quant_warmup = getattr(model, "post_quant_warmup", None)
    if callable(post_quant_warmup):
        post_quant_warmup()


def make_exporting_loader(options: ExportOptions) -> type:
    """Return a DefaultModelLoader variant that exports before preprocessing."""
    from tokenspeed.runtime.configs.load_config import LoadFormat
    from tokenspeed.runtime.model_loader.loader import (
        DefaultModelLoader,
        _initialize_model,
    )
    from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype

    class ExportingDefaultModelLoader(DefaultModelLoader):
        def __init__(self, load_config) -> None:
            # A custom loader class is carried in LoadConfig.load_format.  The
            # inherited file iterator still needs the concrete source format.
            load_config.load_format = LoadFormat.SAFETENSORS
            super().__init__(load_config)

        def load_model(self, *, model_config, device_config):
            target_device = torch.device(device_config.device)
            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = _initialize_model(model_config, self.load_config)
                model.load_weights(self._get_all_weights(model_config, model))
                export_raw_rank_state(model, options)
                if options.process_after_export:
                    process_weights_after_loading(model, target_device)
            return model.eval()

    return ExportingDefaultModelLoader


class RawRankStateLoader:
    """Load :data:`FORMAT_NAME` and preprocess for the current GPU architecture."""

    def __init__(self, load_config) -> None:
        self.load_config = load_config

    def download_model(self, model_config) -> None:
        del model_config

    def load_model(self, *, model_config, device_config):
        from safetensors import safe_open
        from tokenspeed.runtime.model_loader.loader import _initialize_model
        from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype

        model_path = Path(model_config.model_path)
        manifest_path = model_path / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT_NAME or not manifest.get("complete"):
            raise RuntimeError(f"invalid or incomplete raw rank state: {manifest_path}")
        expected_topology = {
            "tp_size": 8,
            "tp_rank": 0,
            "ep_size": 1,
            "ep_rank": 0,
        }
        mismatches = {
            name: (manifest.get(name), expected)
            for name, expected in expected_topology.items()
            if int(manifest.get(name, -1)) != expected
        }
        if mismatches:
            raise ValueError(
                f"raw rank state topology does not match TP8/EP1 rank 0: {mismatches}"
            )
        if manifest.get("raw_state_before_kernel_preprocessing") is not True:
            raise ValueError("rank state is not marked as raw pre-kernel state")

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(model_config, self.load_config)

            state = _portable_state_dict(model)
            loaded: set[str] = set()
            for part in manifest["parts"]:
                part_path = model_path / part["filename"]
                with safe_open(part_path, framework="pt", device="cpu") as source:
                    for name in source.keys():
                        if name not in state:
                            raise KeyError(f"checkpoint tensor has no model target: {name}")
                        if name in loaded:
                            raise KeyError(f"checkpoint tensor is duplicated: {name}")
                        destination = state[name]
                        tensor = source.get_tensor(name)
                        if tensor.shape != destination.shape:
                            raise ValueError(
                                f"shape mismatch for {name}: "
                                f"{tuple(tensor.shape)} != {tuple(destination.shape)}"
                            )
                        if tensor.dtype != destination.dtype:
                            raise TypeError(
                                f"dtype mismatch for {name}: "
                                f"{tensor.dtype} != {destination.dtype}"
                            )
                        destination.copy_(tensor)
                        loaded.add(name)

            if len(loaded) != int(manifest["tensor_count"]):
                raise ValueError(
                    "manifest tensor count does not match checkpoint contents: "
                    f"{manifest['tensor_count']} != {len(loaded)}"
                )
            missing = sorted(set(state) - loaded)
            if missing:
                preview = ", ".join(missing[:8])
                raise KeyError(
                    f"raw rank state is missing {len(missing)} tensors: {preview}"
                )
            # Post-load preprocessors replace large raw parameters one module at
            # a time. Do not retain the state-dict tensor aliases here, or every
            # raw MXFP4 tensor survives alongside every processed allocation and
            # a full Kimi-K3 rank exceeds 288 GiB during conversion.
            del destination, state, tensor
            torch.cuda.empty_cache()
            process_weights_after_loading(model, target_device)
        return model.eval()
