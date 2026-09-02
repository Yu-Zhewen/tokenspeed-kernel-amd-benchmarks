# Kimi-K3 TP8/EP1 result index

This index intentionally contains only the three approved targets:

| Target | Status | Result |
|---|---|---|
| gfx950 toy 1-GPU logical TP8 rank 0 | complete | [`gfx950_toy_1gpu_0b1061eb/`](gfx950_toy_1gpu_0b1061eb/) |
| gfx950 real 8-GPU TP8/EP1 | complete | [`gfx950_real_8gpu_0b1061eb/`](gfx950_real_8gpu_0b1061eb/) |
| gfx1250 toy 1-GPU logical TP8 rank 0 | pending physical run | [`gfx1250_toy_1gpu_pending/`](gfx1250_toy_1gpu_pending/) |

“Toy 1-GPU” is one physical GPU executing rank 0 of the TP8 model with local
collective substitutes. “Real 8-GPU” executes all eight ranks with physical
collectives.

Every directory follows [`../RESULT_TEMPLATE.md`](../RESULT_TEMPLATE.md):
the same environment, correctness, unprofiled performance, category hotspot,
exact-kernel, command, artifact, and limitation sections. Both collected
gfx950 results now use the same eager GPU-kernel hotspot methodology.

Use [`../RUNBOOK.md`](../RUNBOOK.md) to collect any target and
[`../TEST_PLAN.md`](../TEST_PLAN.md) for the required metric contract. A new
TokenSpeed revision gets a new revision-suffixed directory; never overwrite an
older revision with a different software stack.
