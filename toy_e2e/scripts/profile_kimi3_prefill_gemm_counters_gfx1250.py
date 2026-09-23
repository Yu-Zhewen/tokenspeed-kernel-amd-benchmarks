#!/usr/bin/env python3
"""Launch one exact Kimi-K3 prefill GEMM for rocprofv3 counter collection."""

from __future__ import annotations

import argparse

import torch
from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
    gluon_mm_a16w16_largem_gfx1250,
)

SHAPES = {
    "shared_down": (7168, 1536),
    "routed_up": (7168, 3584),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "gluon"), required=True)
    parser.add_argument("--shape", choices=tuple(SHAPES), required=True)
    parser.add_argument("--m", choices=(4096, 8192), type=int, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    props = torch.cuda.get_device_properties(0)
    if not props.gcnArchName.startswith("gfx1250"):
        raise RuntimeError(f"expected gfx1250, got {props.gcnArchName}")
    n, k = SHAPES[args.shape]
    activation = torch.empty(args.m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(args.m, n, device="cuda", dtype=torch.bfloat16)

    if args.backend == "torch":

        def launch() -> torch.Tensor:
            return torch.mm(activation, weight.T, out=output)

    else:

        def launch() -> torch.Tensor:
            return gluon_mm_a16w16_largem_gfx1250(
                activation,
                weight,
                out=output,
            )

    for _ in range(args.iterations):
        launch()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
