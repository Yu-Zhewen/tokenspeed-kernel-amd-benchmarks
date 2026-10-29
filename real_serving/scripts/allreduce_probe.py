#!/usr/bin/env python3
"""Time bf16 all-reduces over a model's TP group outside the server.

Run once per node, in the serving image with the serving container's network
setup, all nodes within a few seconds of each other. Rank 0 prints one line
per mode and size, the format generate_hotspot_report.py reads from
`<run>/allreduce_probe.txt`:

  eager tokens 4 bytes 57344 us 18.6

`eager` issues back-to-back collectives; `graph` replays them from a CUDA
graph, as the server's decode does.
"""

import argparse
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(x, eager_iters, graph_iters):
    for _ in range(20):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(eager_iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    eager = (time.perf_counter() - start) / eager_iters
    if not graph_iters:
        return eager, None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            dist.all_reduce(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_iters):
            dist.all_reduce(x)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    return eager, (time.perf_counter() - start) / (20 * graph_iters)


def worker(local, args):
    rank = args.node_rank * args.gpus + local
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{args.master}:{args.port}", rank=rank,
        world_size=args.nnodes * args.gpus, device_id=torch.device("cuda", local),
    )
    for tokens in args.tokens:
        x = torch.ones(tokens, args.hidden, dtype=torch.bfloat16, device="cuda")
        small = tokens <= 256
        eager, graph = run(x, 200 if small else 20, 50 if small else 0)
        if rank == 0:
            for mode, seconds in (("eager", eager), ("graph", graph)):
                if seconds is not None:
                    print(f"{mode} tokens {tokens} bytes {x.numel() * 2} us "
                          f"{seconds * 1e6:.1f}", flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--master", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29511)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 64, 256, 8192])
    args = parser.parse_args()
    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == "__main__":
    main()
