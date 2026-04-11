"""Tests for downloader module."""

from llamacpp_tuner.downloader import (
    _is_valid_model,
    _quant_to_pattern,
    find_gguf_file,
    get_model_path,
    list_invalid_models,
    list_valid_models,
    remove_model,
)


class TestQuantToPattern:
    def test_q4_k_m(self):
        assert _quant_to_pattern("Q4_K_M") == "q4_k_m"

    def test_q4_k_s(self):
        assert _quant_to_pattern("Q4_K_S") == "q4_k_s"

    def test_q8_0(self):
        assert _quant_to_pattern("Q8_0") == "q8_0"


class TestIsValidModel:
    def test_nonexistent_file(self, tmp_path):
        assert _is_valid_model(tmp_path / "fake.gguf") is False

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.gguf"
        empty.write_text("")
        assert _is_valid_model(empty) is False

    def test_too_small_file(self, tmp_path):
        small = tmp_path / "small.gguf"
        small.write_bytes(b"GGUF")  # 4 bytes
        assert _is_valid_model(small) is False

    def test_valid_file(self, tmp_path):
        valid = tmp_path / "valid.gguf"
        valid.write_bytes(b"x" * 2048)  # 2KB
        assert _is_valid_model(valid) is True


class TestGetModelPath:
    def test_no_models_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "llamacpp_tuner.downloader.get_models_dir", lambda: tmp_path
        )
        monkeypatch.setattr(
            "llamacpp_tuner.downloader.find_gguf_file", lambda repo, quant: None
        )
        assert get_model_path("owner/repo", "Q4_K_M") is None

    def test_returns_local_path_when_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "llamacpp_tuner.downloader.get_models_dir", lambda: tmp_path
        )
        monkeypatch.setattr(
            "llamacpp_tuner.downloader.find_gguf_file",
            lambda repo, quant: "model-q4_k_m.gguf",
        )
        model = tmp_path / "model-q4_k_m.gguf"
        model.write_bytes(b"x" * 2048)

        result = get_model_path("owner/repo", "Q4_K_M")
        assert result == model

    def test_returns_none_when_local_file_invalid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "llamacpp_tuner.downloader.get_models_dir", lambda: tmp_path
        )
        monkeypatch.setattr(
            "llamacpp_tuner.downloader.find_gguf_file",
            lambda repo, quant: "model-q4_k_m.gguf",
        )
        model = tmp_path / "model-q4_k_m.gguf"
        model.write_bytes(b"x" * 10)  # Too small, invalid

        result = get_model_path("owner/repo", "Q4_K_M")
        assert result is None

    def test_returns_none_when_repo_has_no_gguf(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "llamacpp_tuner.downloader.get_models_dir", lambda: tmp_path
        )
        monkeypatch.setattr(
            "llamacpp_tuner.downloader.find_gguf_file", lambda repo, quant: None
        )
        # Local file exists but should be ignored
        local = tmp_path / "other-model-q4_k_m.gguf"
        local.write_bytes(b"x" * 2048)

        result = get_model_path("owner/repo", "Q4_K_M")
        assert result is None


class TestFindGgufFile:
    def test_returns_none_for_invalid_repo(self, monkeypatch):
        from unittest.mock import Mock

        from huggingface_hub.errors import RepositoryNotFoundError

        def mock_list_repo_files(repo_id):
            mock_response = Mock()
            mock_response.status_code = 404
            raise RepositoryNotFoundError("Repo not found", response=mock_response)

        monkeypatch.setattr(
            "llamacpp_tuner.downloader.list_repo_files", mock_list_repo_files
        )
        result = find_gguf_file("bad/repo", "Q4_K_M")
        assert result is None

    def test_finds_exact_quant_match(self, monkeypatch):
        files = [
            "model-q4_k_m.gguf",
            "model-q5_k_m.gguf",
            "model-q8_0.gguf",
        ]

        def mock_list_repo_files(repo_id):
            return files

        monkeypatch.setattr(
            "llamacpp_tuner.downloader.list_repo_files", mock_list_repo_files
        )
        assert find_gguf_file("owner/repo", "Q4_K_M") == "model-q4_k_m.gguf"

    def test_falls_back_to_any_q4(self, monkeypatch):
        files = ["model-q4_0.gguf"]  # not q4_k_m

        def mock_list_repo_files(repo_id):
            return files

        monkeypatch.setattr(
            "llamacpp_tuner.downloader.list_repo_files", mock_list_repo_files
        )
        assert find_gguf_file("owner/repo", "Q4_K_M") == "model-q4_0.gguf"


class TestListValidInvalidModels:
    def test_lists_valid_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "llamacpp_tuner.downloader.get_models_dir", lambda: tmp_path
        )
        valid1 = tmp_path / "model1-q4_k_m.gguf"
        valid1.write_bytes(b"x" * 2048)
        valid2 = tmp_path / "model2-q5_k_m.gguf"
        valid2.write_bytes(b"x" * 4096)
        invalid = tmp_path / "incomplete.gguf"
        invalid.write_bytes(b"x" * 10)

        result = list_valid_models()
        assert len(result) == 2
        assert valid1 in result
        assert valid2 in result

    def test_lists_invalid_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "llamacpp_tuner.downloader.get_models_dir", lambda: tmp_path
        )
        valid = tmp_path / "valid.gguf"
        valid.write_bytes(b"x" * 2048)
        invalid1 = tmp_path / "incomplete1.gguf"
        invalid1.write_bytes(b"x" * 10)
        invalid2 = tmp_path / "incomplete2.gguf"
        invalid2.write_text("")

        result = list_invalid_models()
        assert len(result) == 2
        assert invalid1 in result
        assert invalid2 in result

    def test_empty_dir_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "llamacpp_tuner.downloader.get_models_dir", lambda: tmp_path
        )
        assert list_valid_models() == []
        assert list_invalid_models() == []


class TestRemoveModel:
    def test_removes_gguf_file(self, tmp_path):
        model = tmp_path / "test.gguf"
        model.write_bytes(b"x" * 2048)

        assert model.exists()
        assert remove_model(model) is True
        assert not model.exists()

    def test_returns_false_for_nonexistent(self, tmp_path):
        missing = tmp_path / "missing.gguf"
        assert remove_model(missing) is False

    def test_returns_false_for_non_gguf(self, tmp_path):
        other = tmp_path / "test.txt"
        other.write_text("hello")
        assert remove_model(other) is False
        assert other.exists()
