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
    monkeypatch.setattr(
        "toy_e2e.scripts.profile_logical_rank_stages.torch.profiler.profile",
        lambda **_kwargs: profiler,
    )
    monkeypatch.setattr(
        "toy_e2e.scripts.profile_logical_rank_stages.torch.cuda.synchronize",
        lambda: None,
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
