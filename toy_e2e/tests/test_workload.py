import pytest

from toy_e2e.workload import synthetic_prompt


def test_synthetic_prompt_is_deterministic_varied_and_in_vocabulary():
    first = synthetic_prompt(
        length=32,
        seed=7,
        request_index=3,
        vocabulary_size=101,
    )
    repeated = synthetic_prompt(
        length=32,
        seed=7,
        request_index=3,
        vocabulary_size=101,
    )
    next_request = synthetic_prompt(
        length=32,
        seed=7,
        request_index=4,
        vocabulary_size=101,
    )

    assert first == repeated
    assert first != next_request
    assert len(set(first)) > 1
    assert all(0 <= token < 101 for token in first)


@pytest.mark.parametrize(
    ("length", "request_index", "vocabulary_size"),
    [(0, 0, 10), (1, -1, 10), (1, 0, 0)],
)
def test_synthetic_prompt_rejects_invalid_dimensions(
    length,
    request_index,
    vocabulary_size,
):
    with pytest.raises(ValueError):
        synthetic_prompt(
            length=length,
            seed=7,
            request_index=request_index,
            vocabulary_size=vocabulary_size,
        )
