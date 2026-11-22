import pytest

from llamacpp_tuner import servers


@pytest.fixture(autouse=True)
def no_real_gpu_hold(monkeypatch):
    """`lct serve` keeps AMD GPUs awake by default; tests must not touch real ones."""
    monkeypatch.setattr(servers, "keep_awake", lambda: 0)


@pytest.fixture(autouse=True)
def private_lct_home(monkeypatch, tmp_path):
    """Keep every test away from the checkout's profiles.toml and tmp/."""
    monkeypatch.setenv("LCT_HOME", str(tmp_path))
