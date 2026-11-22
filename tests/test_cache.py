"""Tests for user-writable cache paths."""

from llamacpp_tuner import cache


def test_lct_home_overrides_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("LCT_HOME", str(tmp_path))

    assert cache.get_cache_dir() == tmp_path


def test_checkout_uses_local_tmp(monkeypatch, tmp_path):
    monkeypatch.delenv("LCT_HOME", raising=False)
    monkeypatch.setattr(cache, "_PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").touch()

    assert cache.get_cache_dir() == tmp_path / "tmp"


def test_installed_package_uses_xdg_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("LCT_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(cache, "_PROJECT_ROOT", tmp_path / "package")

    assert cache.get_cache_dir() == tmp_path / "lct"


def test_checkout_keeps_profiles_in_the_repository(monkeypatch, tmp_path):
    monkeypatch.delenv("LCT_HOME", raising=False)
    monkeypatch.setattr(cache, "_PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").touch()

    assert cache.get_profiles_path() == tmp_path / "profiles.toml"


def test_lct_home_moves_only_the_cache_dir_of_a_checkout(monkeypatch, tmp_path):
    # A worktree that shares the main checkout's downloads keeps its own profiles.
    monkeypatch.setenv("LCT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cache, "_PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").touch()

    assert cache.get_profiles_path() == tmp_path / "profiles.toml"
    assert cache.get_cache_dir() == tmp_path / "home"


def test_installed_package_keeps_profiles_in_the_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("LCT_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(cache, "_PROJECT_ROOT", tmp_path / "package")

    assert cache.get_profiles_path() == tmp_path / "lct" / "aliases.toml"
