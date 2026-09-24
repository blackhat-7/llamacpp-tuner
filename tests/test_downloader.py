"""Tests for GGUF discovery and download."""

from pathlib import Path

import pytest

from llamacpp_tuner import downloader


def test_repository_storage_is_namespaced(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, "get_models_dir", lambda: tmp_path)

    assert downloader._repository_dir("owner/model") == tmp_path / "owner%2Fmodel"
    assert downloader._repository_dir("model") == tmp_path / "model"


def test_accepts_any_exact_quant():
    files = ["model-Q4_K_M.gguf", "model-Q6_K.gguf"]

    assert downloader._select_model(files, quant="Q6_K") == ["model-Q6_K.gguf"]
    assert downloader._select_model(files, quant="Q6") == []


def test_rejects_ambiguous_quant():
    files = ["model-Q6_K.gguf", "model-vision-Q6_K.gguf"]

    with pytest.raises(ValueError, match="--file"):
        downloader._select_model(files, quant="Q6_K")

    assert downloader._select_model(files, filename="model-Q6_K.gguf") == [
        "model-Q6_K.gguf"
    ]


def test_returns_nested_shards_in_order():
    files = [
        "quants/model-Q6_K-00002-of-00002.gguf",
        "quants/model-Q6_K-00001-of-00002.gguf",
    ]

    assert downloader._select_model(files, quant="Q6_K") == [
        "quants/model-Q6_K-00001-of-00002.gguf",
        "quants/model-Q6_K-00002-of-00002.gguf",
    ]


def test_rejects_incomplete_shards():
    files = [
        "quants/model-Q6_K-00001-of-00003.gguf",
        "quants/model-Q6_K-00003-of-00003.gguf",
    ]

    with pytest.raises(ValueError, match="Incomplete"):
        downloader._select_model(files, quant="Q6_K")


def test_downloads_every_shard(monkeypatch, tmp_path):
    files = ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"]
    monkeypatch.setattr(downloader, "find_gguf_files", lambda *args, **kwargs: files)
    downloaded: list[str] = []

    def fake_download(repo: str, filename: str, force: bool) -> Path:
        downloaded.append(filename)
        return tmp_path / filename

    monkeypatch.setattr(downloader, "_download", fake_download)

    result = downloader.download_model("owner/model", quant="Q6_K")

    assert result == tmp_path / files[0]
    assert downloaded == files


def test_requires_explicit_projector_when_multiple_exist():
    files = ["model.mmproj-Q8_0.gguf", "model.mmproj-f16.gguf"]

    with pytest.raises(ValueError, match="--mmproj-file"):
        downloader._select_projector(files)
    assert (
        downloader._select_projector(files, filename="model.mmproj-f16.gguf")
        == "model.mmproj-f16.gguf"
    )


def test_requires_explicit_projector_across_families():
    files = ["model-a-mmproj-f16.gguf", "model-b-mmproj-f16.gguf"]

    with pytest.raises(ValueError, match="--mmproj-file"):
        downloader._select_projector(files)


def test_downloaded_repo_resolution_is_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, "get_models_dir", lambda: tmp_path)
    repo_dir = tmp_path / "owner%2Fmodel"
    repo_dir.mkdir(parents=True)
    model = repo_dir / "model-Q6_K.gguf"
    model.touch()
    monkeypatch.setattr(
        downloader,
        "_repo_files",
        lambda repo: (_ for _ in ()).throw(AssertionError("network accessed")),
    )

    assert downloader.get_model_path("owner/model", quant="Q6_K") == model


def test_same_filename_in_different_repositories_stays_isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, "get_models_dir", lambda: tmp_path)
    first = tmp_path / "one%2Fmodel" / "model-Q6_K.gguf"
    second = tmp_path / "two%2Fmodel" / "model-Q6_K.gguf"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()

    assert downloader.get_model_path("one/model", quant="Q6_K") == first
    assert downloader.get_model_path("two/model", quant="Q6_K") == second


def test_bare_filename_must_be_unambiguous(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, "get_models_dir", lambda: tmp_path)
    for owner in ("one", "two"):
        path = tmp_path / owner / "model.gguf"
        path.parent.mkdir()
        path.touch()

    with pytest.raises(ValueError, match="Multiple local files"):
        downloader.resolve_model("model.gguf")


def test_missing_explicit_path_does_not_fall_back_by_basename(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, "get_models_dir", lambda: tmp_path)
    cached = tmp_path / "owner" / "model.gguf"
    cached.parent.mkdir()
    cached.touch()

    with pytest.raises(FileNotFoundError):
        downloader.resolve_model("/missing/model.gguf")


def test_repository_id_ending_in_gguf_uses_selector(monkeypatch, tmp_path):
    expected = tmp_path / "model.gguf"
    monkeypatch.setattr(downloader, "get_model_path", lambda *args, **kwargs: expected)

    assert downloader.resolve_model("owner/repo.gguf", quant="Q6_K") == expected


def test_lists_nested_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, "get_models_dir", lambda: tmp_path)
    model = tmp_path / "owner" / "repo" / "model.gguf"
    model.parent.mkdir(parents=True)
    model.touch()

    assert downloader.list_downloaded_models() == [model]


def test_list_repo_models_skips_projectors_and_other_files(monkeypatch):
    from huggingface_hub import RepoFile

    def entry(path: str, size: int) -> RepoFile:
        return RepoFile(path=path, size=size, oid="0", lfs=None, last_commit=None)

    class FakeApi:
        def list_repo_tree(self, repo_id, recursive):
            return [
                entry("model-Q4_K_M.gguf", 10),
                entry("mmproj-F16.gguf", 2),
                entry("README.md", 1),
            ]

    monkeypatch.setattr(downloader, "HfApi", FakeApi)

    assert downloader.list_repo_models("owner/repo") == [("model-Q4_K_M.gguf", 10)]
