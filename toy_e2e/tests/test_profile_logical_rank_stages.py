from toy_e2e.scripts.profile_logical_rank_stages import StageTrace


class _FakeProfiler:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def export_chrome_trace(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{}")


def test_stage_trace_captures_only_requested_forward_count(monkeypatch, tmp_path):
    profiler = _FakeProfiler()
    shape_calls = []
    monkeypatch.setattr(
        "toy_e2e.scripts.profile_logical_rank_stages.torch.profiler.profile",
        lambda **_kwargs: profiler,
    )
    monkeypatch.setattr(
        "toy_e2e.scripts.profile_logical_rank_stages.torch.cuda.synchronize",
        lambda: None,
    )
    monkeypatch.setattr(
        "toy_e2e.scripts.profile_logical_rank_stages.start_shape_capture",
        lambda: shape_calls.append("start"),
    )
    monkeypatch.setattr(
        "toy_e2e.scripts.profile_logical_rank_stages.stop_shape_capture",
        lambda path: shape_calls.append(("stop", path)),
    )
    output = tmp_path / "trace.json"
    trace = StageTrace(phase="decode", output=output, max_steps=2)

    trace.before_forward("prefill")
    assert profiler.started is False

    trace.before_forward("decode")
    trace.after_forward("decode")
    assert profiler.started is True
    assert profiler.stopped is False

    trace.before_forward("decode")
    trace.after_forward("decode")
    assert profiler.stopped is True
    assert trace.steps == 2
    assert output.read_text(encoding="utf-8") == "{}"
    assert shape_calls == ["start", ("stop", tmp_path / "trace.shapes.json")]


def test_stage_trace_can_capture_shapes_without_gpu_trace(monkeypatch, tmp_path):
    shape_calls = []
    monkeypatch.setattr(
        "toy_e2e.scripts.profile_logical_rank_stages.torch.profiler.profile",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("GPU profiler must stay disabled")
        ),
    )
    monkeypatch.setattr(
        "toy_e2e.scripts.profile_logical_rank_stages.start_shape_capture",
        lambda: shape_calls.append("start"),
    )
    monkeypatch.setattr(
        "toy_e2e.scripts.profile_logical_rank_stages.stop_shape_capture",
        lambda path: shape_calls.append(("stop", path)),
    )
    output = tmp_path / "trace.json"
    trace = StageTrace(
        phase="prefill",
        output=output,
        max_steps=1,
        capture_trace=False,
    )

    trace.before_forward("prefill")
    trace.after_forward("prefill")

    assert trace.steps == 1
    assert not output.exists()
    assert shape_calls == ["start", ("stop", tmp_path / "trace.shapes.json")]
