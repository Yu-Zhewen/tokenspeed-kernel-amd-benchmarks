"""Shared deterministic synthetic-token workload helpers."""

from __future__ import annotations

import random

DEFAULT_PROMPT_SEED = 7
DEFAULT_SYNTHETIC_VOCAB_SIZE = 160_000


def synthetic_prompt(
    *,
    length: int,
    seed: int,
    request_index: int,
    vocabulary_size: int,
) -> list[int]:
    """Return a reproducible varied prompt for one synthetic request."""
    if length <= 0:
        raise ValueError("prompt length must be positive")
    if request_index < 0:
        raise ValueError("request index cannot be negative")
    if vocabulary_size <= 0:
        raise ValueError("vocabulary size must be positive")

    generator = random.Random(seed + request_index)
    return [generator.randrange(vocabulary_size) for _ in range(length)]
