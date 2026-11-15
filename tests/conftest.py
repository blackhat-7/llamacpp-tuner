import pytest

from llamacpp_tuner import servers


@pytest.fixture(autouse=True)
def no_real_gpu_hold(monkeypatch):
    """`lct serve` keeps AMD GPUs awake by default; tests must not touch real ones."""
    monkeypatch.setattr(servers, "keep_awake", lambda: 0)
