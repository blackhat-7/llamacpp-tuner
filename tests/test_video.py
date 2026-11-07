"""Tests for video clips: workflows, the ComfyUI client, media helpers and commands."""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from llamacpp_tuner import servers, video
from llamacpp_tuner.cli import main


def links(graph: dict) -> list[list]:
    return [
        value
        for node in graph.values()
        for value in node["inputs"].values()
        if isinstance(value, list)
    ]


def nodes(graph: dict, class_type: str) -> list[dict]:
    return [n["inputs"] for n in graph.values() if n["class_type"] == class_type]


def assert_wired(graph: dict) -> None:
    """Every link points at an existing node, as ComfyUI requires."""
    for node_id, output in links(graph):
        assert node_id in graph and isinstance(output, int)


def test_frames_snap_up_to_h3s_grid():
    assert video.frame_count(5) == 124
    assert video.frame_count(10) == 243
    assert video.frame_count(15) == 362
    assert video.frame_count(0.1) == 5


def test_canvas_matches_h3s_native_sizes():
    assert video.canvas(16 / 9, "standard") == (1344, 768)
    assert video.canvas(9 / 16, "high") == (768, 1344)
    assert video.canvas(1.0, "standard") == (768, 768)
    assert video.canvas(16 / 9, "draft") == (832, 480)


def test_text_clip_uses_fl2va_and_its_turbo_lora():
    graph = video.h3_graph("a cat", (1344, 768), 124, 8, 7, "make")

    assert_wired(graph)
    assert (
        nodes(graph, "UNETLoader")[0]["unet_name"]
        == video.Path(video.FILES["fl2va"][1]).name
    )
    assert "fl2v_turbo_8step" in nodes(graph, "LoraLoaderModelOnly")[0]["lora_name"]
    conditioning = nodes(graph, "MiniMaxH3ImageToVideo")[0]
    assert (conditioning["width"], conditioning["height"], conditioning["length"]) == (
        1344,
        768,
        124,
    )
    assert nodes(graph, "BasicScheduler")[0]["steps"] == 8
    assert nodes(graph, "RandomNoise")[0]["noise_seed"] == 7
    assert nodes(graph, "SaveVideo")[0]["filename_prefix"] == "video/make"
    assert not nodes(graph, "MiniMaxH3ReferenceToVideo")


def test_first_frame_goes_to_fl2va_directly():
    graph = video.h3_graph("x", (832, 480), 124, 4, 1, "make", first_frame="a.png")

    assert_wired(graph)
    assert "first_frame" in nodes(graph, "MiniMaxH3ImageToVideo")[0]
    assert "fl2v_turbo_4step" in nodes(graph, "LoraLoaderModelOnly")[0]["lora_name"]


def test_references_use_ref2va_and_a_guide_continues_a_clip():
    graph = video.h3_graph(
        "x",
        (1344, 768),
        146,
        8,
        1,
        "extend-part",
        photos=("p1.png", "p2.png"),
        audios=("voice.wav",),
        first_frame="f.png",
        guide="tail.mp4",
    )

    assert_wired(graph)
    assert "ref2va" in nodes(graph, "UNETLoader")[0]["unet_name"]
    assert "ref2v_turbo_8step" in nodes(graph, "LoraLoaderModelOnly")[0]["lora_name"]
    refs = nodes(graph, "MiniMaxH3ReferenceToVideo")[0]
    assert {
        "ref_images.ref_image_0",
        "ref_images.ref_image_1",
        "ref_audios.ref_audio_0",
    } <= set(refs)
    guides = nodes(graph, "MiniMaxH3AddGuide")
    # The first frame is anchored at 0, then the continuation clip with its sound.
    assert [g["frame_idx"] for g in guides] == [0, 0]
    assert "audio" in guides[1] and "audio_vae" in guides[1]


def test_edit_passes_the_source_and_its_sound_as_video_1():
    graph = video.h3_graph("x", (1344, 768), 124, 8, 1, "edit", videos=("src.mp4",))

    assert_wired(graph)
    refs = nodes(graph, "MiniMaxH3ReferenceToVideo")[0]
    assert {"ref_videos.ref_video_0", "ref_video_audios.ref_video_audio_0"} <= set(refs)
    assert nodes(graph, "LoadVideo")[0]["file"] == "src.mp4"


def test_upscale_keeps_sound_and_skips_colour_correction():
    graph = video.upscale_graph("clip.mp4", 1080, 3, "upscale")

    assert_wired(graph)
    resize = nodes(graph, "ResizeImageMaskNode")[0]
    assert resize["resize_type.shorter_size"] == 1080
    assert nodes(graph, "SeedVR2PostProcessing")[0]["color_correction_method"] == "none"
    assert "audio" in nodes(graph, "CreateVideo")[0]


def test_progress_reads_the_last_tqdm_step():
    tail = "  0%|          | 0/8 [00:00<?, ?it/s]\r 25%|██▌       | 2/8 [01:57<05:51, 58.55s/it]"
    assert video.progress_line(tail) == "step 2/8 (01:57)"
    assert video.progress_line("loading model") == ""


def test_wait_returns_the_saved_video_and_reports_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("LCT_HOME", str(tmp_path))
    saved = {
        "p1": {
            "status": {"status_str": "success"},
            "outputs": {
                "2": {"images": [{"filename": "lct-1-tail.mp4", "type": "input"}]},
                "9": {
                    "images": [
                        {
                            "filename": "make_00001_.mp4",
                            "subfolder": "video",
                            "type": "output",
                        }
                    ]
                },
            },
        }
    }
    failed = {
        "p1": {
            "status": {
                "status_str": "error",
                "messages": [
                    [
                        "execution_error",
                        {
                            "node_type": "VAEDecode",
                            "exception_message": "HIP out of memory",
                        },
                    ]
                ],
            }
        }
    }
    with patch("llamacpp_tuner.video._request", return_value=saved):
        assert (
            video.wait("http://h", "p1")
            == tmp_path / "comfyui/data/output/video/make_00001_.mp4"
        )
    with (
        patch("llamacpp_tuner.video._request", return_value=failed),
        pytest.raises(RuntimeError, match="VAEDecode failed: HIP out of memory"),
    ):
        video.wait("http://h", "p1")


def test_serve_comfyui_maps_host_and_port(tmp_path):
    with patch("llamacpp_tuner.cli.run_comfyui") as run_comfyui:
        result = CliRunner().invoke(
            main,
            ["serve", str(tmp_path), "--backend", "comfyui", "--host", "0.0.0.0"]
            + ["--port", "8188", "--extra-args", "--use-flash-attention"],
        )
        bad = CliRunner().invoke(
            main, ["serve", str(tmp_path), "--backend", "comfyui", "--ctx", "9"]
        )

    assert result.exit_code == 0, result.output
    assert run_comfyui.call_args.args == (
        tmp_path,
        ["--listen", "0.0.0.0", "--port", "8188", "--use-flash-attention"],
    )
    assert bad.exit_code != 0 and "base directory" in bad.output


def test_setup_comfyui_adds_the_video_profile_and_downloads_models(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LCT_HOME", str(tmp_path))
    with (
        patch("llamacpp_tuner.cli.install_comfyui", return_value=Path("/venv/python")),
        patch("llamacpp_tuner.cli.missing_files", return_value=["fl2va"]),
        patch("llamacpp_tuner.cli.download_files") as download,
    ):
        first = CliRunner().invoke(main, ["setup", "--backend", "comfyui"])
        second = CliRunner().invoke(main, ["setup", "--backend", "comfyui"])

    assert first.exit_code == 0, first.output
    assert "Added profile 'video'" in first.output
    assert "Added profile" not in second.output
    assert (tmp_path / "aliases.toml").read_text().count("--backend comfyui") == 1
    download.assert_called_with(["fl2va"])


def test_comfyuis_ready_line_gives_its_address():
    assert (
        servers.address("To see the GUI go to: http://127.0.0.1:8188")
        == "http://127.0.0.1:8188"
    )


def make_with(args: list[str], tmp_path: Path) -> tuple:
    """Run 'lct video make' against a fake server; returns the result and every submitted graph."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    graphs = []
    with (
        patch("llamacpp_tuner.cli.video_server", return_value="http://h"),
        patch("llamacpp_tuner.cli.missing_for", return_value=[]),
        patch(
            "llamacpp_tuner.cli.submit",
            side_effect=lambda url, graph: graphs.append(graph) or "p",
        ),
        patch("llamacpp_tuner.cli.wait", return_value=clip),
        patch("llamacpp_tuner.cli.probe", return_value=(1344, 768, 5.2)),
        patch("llamacpp_tuner.cli.stage", side_effect=lambda path: path.name),
    ):
        result = CliRunner().invoke(main, ["video", "make", *args])
    return result, graphs


def test_make_turns_quality_and_length_into_one_workflow(tmp_path):
    result, graphs = make_with(
        ["a cat", "--seconds", "10", "--quality", "draft", "--aspect", "9:16"], tmp_path
    )

    assert result.exit_code == 0, result.output
    conditioning = nodes(graphs[0], "MiniMaxH3ImageToVideo")[0]
    assert (conditioning["width"], conditioning["height"], conditioning["length"]) == (
        480,
        832,
        243,
    )
    assert "Saved:" in result.output


def test_high_quality_also_upscales_and_out_gets_a_copy(tmp_path):
    out = tmp_path / "final.mp4"
    photo = tmp_path / "alice.png"
    photo.write_bytes(b"png")
    result, graphs = make_with(
        ["x", "--quality", "high", "--photo", str(photo), "--out", str(out)], tmp_path
    )

    assert result.exit_code == 0, result.output
    assert "<Picture 1> alice.png" in result.output
    assert nodes(graphs[0], "MiniMaxH3ReferenceToVideo") and nodes(
        graphs[1], "SeedVR2Conditioning"
    )
    assert out.read_bytes() == b"video"


def test_make_needs_exactly_one_prompt(tmp_path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("from a file")
    both = CliRunner().invoke(
        main, ["video", "make", "text", "--prompt-file", str(prompt)]
    )
    none = CliRunner().invoke(main, ["video", "make"])

    assert both.exit_code != 0 and none.exit_code != 0


def test_video_refuses_a_busy_gpu(monkeypatch):
    with (
        patch("llamacpp_tuner.cli.load_aliases", return_value={"video": "x"}),
        patch("llamacpp_tuner.cli.missing_for", return_value=[]),
        patch(
            "llamacpp_tuner.cli.servers.running", return_value={"strata-flash-next": 1}
        ),
        patch(
            "llamacpp_tuner.cli.servers.system",
            return_value={"vram_total": 24 * 2**30, "vram_used": 22 * 2**30},
        ),
        patch("llamacpp_tuner.cli.servers.start") as start,
    ):
        result = CliRunner().invoke(main, ["video", "make", "a cat"])

    assert result.exit_code != 0
    assert "strata-flash-next" in result.output and "lct down" in result.output
    start.assert_not_called()


ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def clip(path: Path, seconds: float, size: str = "320x180", sound: bool = True) -> Path:
    sources = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24:duration={seconds}"]
    if sound:
        sources += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *sources, "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@ffmpeg
def test_extension_joins_without_repeating_the_anchored_frames(monkeypatch, tmp_path):
    monkeypatch.setenv("LCT_HOME", str(tmp_path))
    source = clip(tmp_path / "a.mp4", 2)
    tail = video.data_dir() / "input" / video.stage_tail(source)
    part = clip(tmp_path / "b.mp4", 2, size="640x360", sound=False)

    joined = video.join(
        [source, part], tmp_path / "joined.mp4", skip=video.GUIDE_FRAMES
    )

    assert video._frames(tail) == video.GUIDE_FRAMES
    assert video._frames(joined) == 48 + 48 - video.GUIDE_FRAMES
    assert video.probe(joined)[:2] == (320, 180) and video.has_audio(joined)


@ffmpeg
def test_staged_video_is_24_fps_and_at_most_15_seconds(monkeypatch, tmp_path):
    monkeypatch.setenv("LCT_HOME", str(tmp_path))
    source = clip(tmp_path / "long.mp4", 16)

    staged = video.data_dir() / "input" / video.stage_video(source)

    assert video._frames(staged) == 15 * 24
    assert video.stage_video(source) == staged.name  # copied once


def test_a_job_needs_only_the_files_it_loads(monkeypatch, tmp_path):
    monkeypatch.setenv("LCT_HOME", str(tmp_path))
    graph = video.h3_graph("x", (832, 480), 124, 4, 1, "t")
    assert video.missing_for(graph) == [
        "audio_vae",
        "fl2va",
        "fl2va_turbo_4",
        "text_encoder",
        "video_vae",
    ]
    for key in ("audio_vae", "fl2va", "fl2va_turbo_4", "text_encoder", "video_vae"):
        video.model_path(key).parent.mkdir(parents=True, exist_ok=True)
        video.model_path(key).write_bytes(b"x")
    assert video.missing_for(graph) == []


def test_a_crashed_server_is_named_not_a_timeout(tmp_path):
    with (
        patch("llamacpp_tuner.cli.video_server", return_value="http://h"),
        patch("llamacpp_tuner.cli.missing_for", return_value=[]),
        patch("llamacpp_tuner.cli.submit", return_value="p"),
        patch("llamacpp_tuner.cli.wait", side_effect=TimeoutError("timed out")),
        patch("llamacpp_tuner.cli.servers.running", return_value={}),
    ):
        result = CliRunner().invoke(main, ["video", "make", "a cat"])

    assert result.exit_code != 0 and "ComfyUI crashed during the job" in result.output


def test_wait_stops_at_a_crash_in_the_server_log(monkeypatch, tmp_path):
    log = tmp_path / "video.log"
    log.write_text("[INFO] got prompt\n")
    busy = {"p1": None}

    def request(url, payload=None):
        log.write_text(log.read_text() + "Fatal Python error: Segmentation fault\n")
        return busy

    monkeypatch.setattr("llamacpp_tuner.video._request", request)
    monkeypatch.setattr("llamacpp_tuner.video.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="crashed .Fatal Python error"):
        video.wait("http://h", "p1", log)


def test_comfyui_starts_with_the_rocm_fix_node(tmp_path):
    python = tmp_path / "python"
    python.write_text("")
    with (
        patch("llamacpp_tuner.video._python", return_value=python),
        patch("llamacpp_tuner.video.run_process") as run_process,
        patch("llamacpp_tuner.video.resource.setrlimit"),
    ):
        video.run_comfyui(tmp_path, ["--port", "8188"])

    node = tmp_path / "custom_nodes" / "lct_rocm_fix.py"
    compile(node.read_text(), str(node), "exec")
    assert "Qwen35VisionPatchEmbed" in node.read_text()
    assert run_process.call_args.args[0][-4:] == [
        "--base-directory",
        str(tmp_path),
        "--port",
        "8188",
    ]
