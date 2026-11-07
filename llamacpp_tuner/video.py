"""Video clips with sound from MiniMax H3, an open audio-video model, run by ComfyUI.

lct installs ComfyUI in its own venv and runs it like llama-server
(`lct serve --backend comfyui`). Clips are made by sending ComfyUI a fixed
workflow over its HTTP API, so nobody has to edit a node graph. The workflows
follow ComfyUI's own MiniMax H3 and SeedVR2 templates.
"""

import hashlib
import json
import math
import re
import resource
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download

from llamacpp_tuner.cache import get_cache_dir
from llamacpp_tuner.llama import run_process

COMFYUI_REPO_URL = "https://github.com/Comfy-Org/ComfyUI.git"
# A release rather than master, which changes daily.
COMFYUI_VERSION = "v0.44.2"
TORCH = ("torch", "torchvision", "torchaudio")

H3 = "Comfy-Org/MiniMax-H3"
# Every file a clip needs: name -> (repository, path in it, ComfyUI model folder
# for a file at the repository's root). Comfy-Org's repositories already use
# ComfyUI's folder layout. int8 models are ComfyUI's choice for 24 GB cards.
FILES = {
    "text_encoder": (
        H3,
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "",
    ),
    "video_vae": (H3, "vae/minimax_h3_video_vae_fp16.safetensors", ""),
    "audio_vae": (H3, "vae/minimax_h3_audio_vae_fp32.safetensors", ""),
    "fl2va": (
        H3,
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "",
    ),
    "fl2va_turbo_4": (
        H3,
        "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "",
    ),
    "fl2va_turbo_8": (
        H3,
        "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "",
    ),
    "ref2va": (
        H3,
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "",
    ),
    "ref2va_turbo_4": (
        H3,
        "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "",
    ),
    "ref2va_turbo_8": (
        "lightx2v/Minimax-h3-Turbo",
        "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors",
        "loras",
    ),
    "upscaler": (
        "Comfy-Org/SeedVR2",
        "diffusion_models/seedvr2_7b_int8_convrot.safetensors",
        "",
    ),
    "upscaler_vae": ("Comfy-Org/SeedVR2", "vae/seedvr2_ema_vae_fp16.safetensors", ""),
    # MiniMax's prompt guides, for 'lct video guide'; they land in models/docs.
    "guide": ("MiniMaxAI/MiniMax-H3", "docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md", ""),
    "guide_ref": (
        "MiniMaxAI/MiniMax-H3",
        "docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md",
        "",
    ),
}

# Sampling steps per quality, each with the turbo LoRA distilled for that many;
# "high" is "standard" upscaled to 1080p.
STEPS = {"draft": 4, "standard": 8, "high": 8}
QUALITIES = tuple(STEPS)
ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4}
FPS = 24
MIN_SECONDS, MAX_SECONDS = 4, 15  # H3's trained range
# Frames of a video being extended that are anchored at the new clip's start,
# so motion carries over; H3 anchors clips of 17k + 5 frames.
GUIDE_FRAMES = 22
UPSCALE_SHORT_SIDE = 1080


def comfyui_dir() -> Path:
    return get_cache_dir() / "comfyui"


def data_dir() -> Path:
    """ComfyUI's --base-directory: models, input and output, kept apart from the
    code so a reinstall keeps them."""
    return comfyui_dir() / "data"


def output_dir() -> Path:
    return data_dir() / "output" / "video"


def _python() -> Path:
    return comfyui_dir() / "venv" / "bin" / "python"


def install_comfyui(torch_index: str | None = None, force: bool = False) -> Path:
    """Clone ComfyUI and install it into its own venv; returns that venv's python.

    torch_index is a PyTorch wheel index, such as a ROCm one; PyPI's torch is CUDA.
    """
    if _python().is_file() and not force:
        return _python()
    uv = shutil.which("uv")
    if not uv:
        raise FileNotFoundError("uv not found; it is needed to install ComfyUI.")
    source, venv = comfyui_dir() / "src", comfyui_dir() / "venv"
    if force:
        shutil.rmtree(source, ignore_errors=True)
    if not (source / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", COMFYUI_VERSION]
            + [COMFYUI_REPO_URL, str(source)],
            check=True,
        )
    # uv's own Python: a distribution's Python, such as Nix's, may not find the
    # system libstdc++ that torch links against.
    subprocess.run(
        [uv, "venv", "--clear", "--managed-python", "-p", "3.13", str(venv)], check=True
    )
    index = ["--index-url", torch_index] if torch_index else []
    subprocess.run([uv, "pip", "install", "-p", str(venv), *TORCH, *index], check=True)
    # Pin the torch just installed, so ComfyUI's requirements cannot swap it for
    # PyPI's CUDA build.
    frozen = subprocess.run(
        [uv, "pip", "freeze", "-p", str(venv)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    constraints = comfyui_dir() / "torch-constraints.txt"
    constraints.write_text(
        "".join(
            f"{line}\n" for line in frozen.splitlines() if line.split("==")[0] in TORCH
        )
    )
    subprocess.run(
        [uv, "pip", "install", "-p", str(venv), "-r", str(source / "requirements.txt")]
        + ["-c", str(constraints)],
        check=True,
    )
    return _python()


# A ComfyUI custom node lct installs. ComfyUI turns MIOpen off on AMD, and
# PyTorch's own ROCm kernel for Qwen3-VL's patch-embedding Conv3d (kernel equal
# to stride) segfaults the process, so every H3 job with a photo, a frame or a
# source video crashed. This runs only that convolution through MIOpen.
ROCM_FIX = """\"\"\"Installed by lct: Qwen3-VL's patch embedding through MIOpen on AMD GPUs.\"\"\"
import torch
from comfy.text_encoders import qwen35

if torch.version.hip:
    _forward = qwen35.Qwen35VisionPatchEmbed.forward

    def _forward_with_miopen(self, x):
        with torch.backends.cudnn.flags(enabled=True):
            return _forward(self, x)

    qwen35.Qwen35VisionPatchEmbed.forward = _forward_with_miopen

NODE_CLASS_MAPPINGS = {}
"""


def run_comfyui(
    base_dir: Path, args: list[str], env: dict[str, str] | None = None
) -> None:
    """Run ComfyUI's server in the foreground; base_dir holds models, input and output."""
    if not _python().is_file():
        raise FileNotFoundError(
            "ComfyUI is not installed. Run 'lct setup --backend comfyui'."
        )
    if not base_dir.is_dir():
        raise FileNotFoundError(f"No ComfyUI base directory at {base_dir}.")
    nodes = base_dir / "custom_nodes"  # ComfyUI lists it but does not create it.
    nodes.mkdir(exist_ok=True)
    (nodes / "lct_rocm_fix.py").write_text(ROCM_FIX)
    # No core dumps: a crash would spend minutes dumping ~40 GB of memory while
    # the server still looks alive.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    run_process(
        [str(_python()), "main.py", "--base-directory", str(base_dir), *args],
        env=env,
        cwd=comfyui_dir() / "src",
    )


# Model files


def model_path(name: str) -> Path:
    _, path, folder = FILES[name]
    return data_dir() / "models" / folder / path


def _model_name(name: str) -> str:
    """The file name ComfyUI's loaders take."""
    return Path(FILES[name][1]).name


def missing_files() -> list[str]:
    return [name for name in FILES if not model_path(name).is_file()]


def missing_for(graph: dict) -> list[str]:
    """Model files a workflow loads that are not downloaded yet."""
    keys = {_model_name(key): key for key in FILES}
    loads = {
        value
        for node in graph.values()
        for name, value in node["inputs"].items()
        if name in ("unet_name", "lora_name", "clip_name", "vae_name")
    }
    return sorted(
        keys[f] for f in loads if f in keys and not model_path(keys[f]).is_file()
    )


def download_files(names: list[str]) -> None:
    for name in names:
        repo, path, folder = FILES[name]
        hf_hub_download(repo, path, local_dir=data_dir() / "models" / folder)


# Sizes and lengths


def canvas(ratio: float, quality: str) -> tuple[int, int]:
    """Width and height for a width/height ratio, in multiples of 32 as H3 needs.

    H3's native canvas has a 768-pixel short edge, capped at 1344x768's area;
    drafts use about 0.4 megapixels, as ComfyUI's template does.
    """
    if quality == "draft":
        area = 0.4e6
    else:
        area = min(768 * 768 * max(ratio, 1 / ratio), 1344 * 768)
    width = math.sqrt(area * ratio)
    return max(32, round(width / 32) * 32), max(32, round(width / ratio / 32) * 32)


def frame_count(seconds: float) -> int:
    """Frames at 24 fps, rounded up to H3's 17k + 5 grid (5 s -> 124)."""
    frames = max(5, round(seconds * FPS))
    return frames + (5 - frames % 17) % 17


# ComfyUI workflows, in its API format: node id -> class and inputs, where
# [id, n] is node id's n-th output.


def _add(graph: dict, class_type: str, inputs: dict) -> str:
    node = str(len(graph) + 1)
    graph[node] = {"class_type": class_type, "inputs": inputs}
    return node


def _save(
    graph: dict, images: list, audio: list | None, fps: object, prefix: str
) -> None:
    created = {"images": images, "fps": fps} | ({"audio": audio} if audio else {})
    video = _add(graph, "CreateVideo", created)
    _add(
        graph,
        "SaveVideo",
        {
            "video": [video, 0],
            "filename_prefix": f"video/{prefix}",
            "format": "mp4",
            "codec": "h264",
        },
    )


def h3_graph(
    prompt: str,
    size: tuple[int, int],
    frames: int,
    steps: int,
    seed: int,
    prefix: str,
    photos: tuple[str, ...] = (),
    videos: tuple[str, ...] = (),
    audios: tuple[str, ...] = (),
    first_frame: str | None = None,
    last_frame: str | None = None,
    guide: str | None = None,
) -> dict:
    """A MiniMax H3 clip. Input files are names in ComfyUI's input folder.

    With reference photos, videos or audio it runs Ref2VA, which the prompt
    refers to as <Picture i>, <Video k> and <Audio j>; otherwise FL2VA, which
    makes video from text alone or from a first and/or last frame. guide is a
    short clip anchored at the start, to continue a video.
    """
    graph: dict = {}
    kind = "ref2va" if photos or videos or audios else "fl2va"
    unet = _add(
        graph, "UNETLoader", {"unet_name": _model_name(kind), "weight_dtype": "default"}
    )
    lora = {
        "model": [unet, 0],
        "lora_name": _model_name(f"{kind}_turbo_{steps}"),
        "strength_model": 1.0,
    }
    model = _add(graph, "LoraLoaderModelOnly", lora)
    clip = _add(
        graph,
        "CLIPLoader",
        {
            "clip_name": _model_name("text_encoder"),
            "type": "minimax",
            "device": "default",
        },
    )
    vae = _add(graph, "VAELoader", {"vae_name": _model_name("video_vae")})
    audio_vae = _add(graph, "VAELoader", {"vae_name": _model_name("audio_vae")})
    width, height = size
    common = {
        "clip": [clip, 0],
        "vae": [vae, 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": frames,
    }
    frames_to_anchor = []  # (image node, frame index) for Ref2VA, which has no frame inputs
    if kind == "ref2va":
        inputs = common | {"audio_vae": [audio_vae, 0], "ref_image_size": "match"}
        for i, photo in enumerate(photos):
            inputs[f"ref_images.ref_image_{i}"] = [
                _add(graph, "LoadImage", {"image": photo}),
                0,
            ]
        for i, video in enumerate(videos):
            loaded = _add(graph, "LoadVideo", {"file": video})
            parts = _add(graph, "GetVideoComponents", {"video": [loaded, 0]})
            inputs[f"ref_videos.ref_video_{i}"] = [parts, 0]
            inputs[f"ref_video_audios.ref_video_audio_{i}"] = [parts, 1]
        for i, audio in enumerate(audios):
            inputs[f"ref_audios.ref_audio_{i}"] = [
                _add(graph, "LoadAudio", {"audio": audio}),
                0,
            ]
        conditioning = _add(graph, "MiniMaxH3ReferenceToVideo", inputs)
        for image, index in ((first_frame, 0), (last_frame, -1)):
            if image:
                frames_to_anchor.append(
                    (_add(graph, "LoadImage", {"image": image}), index)
                )
    else:
        inputs = dict(common)
        for key, image in (("first_frame", first_frame), ("last_frame", last_frame)):
            if image:
                inputs[key] = [_add(graph, "LoadImage", {"image": image}), 0]
        conditioning = _add(graph, "MiniMaxH3ImageToVideo", inputs)
    positive, latent = [conditioning, 0], [conditioning, 1]
    for image, index in frames_to_anchor:
        guided = {"positive": positive, "vae": [vae, 0], "latent": latent}
        guided |= {"image": [image, 0], "frame_idx": index}
        positive = [_add(graph, "MiniMaxH3AddGuide", guided), 0]
    if guide:
        loaded = _add(graph, "LoadVideo", {"file": guide})
        parts = _add(graph, "GetVideoComponents", {"video": [loaded, 0]})
        guided = {"positive": positive, "vae": [vae, 0], "audio_vae": [audio_vae, 0]}
        guided |= {
            "latent": latent,
            "image": [parts, 0],
            "audio": [parts, 1],
            "frame_idx": 0,
        }
        positive = [_add(graph, "MiniMaxH3AddGuide", guided), 0]
    noise = _add(graph, "RandomNoise", {"noise_seed": seed})
    sampler = _add(graph, "KSamplerSelect", {"sampler_name": "res_multistep"})
    sigmas = _add(
        graph,
        "BasicScheduler",
        {"model": [unet, 0], "scheduler": "simple", "steps": steps, "denoise": 1.0},
    )
    guider = _add(graph, "BasicGuider", {"model": [model, 0], "conditioning": positive})
    sampled = _add(
        graph,
        "SamplerCustomAdvanced",
        {
            "noise": [noise, 0],
            "guider": [guider, 0],
            "sampler": [sampler, 0],
            "sigmas": [sigmas, 0],
            "latent_image": latent,
        },
    )
    images = _add(graph, "VAEDecode", {"samples": [sampled, 0], "vae": [vae, 0]})
    audio = _add(
        graph, "VAEDecodeAudio", {"samples": [sampled, 0], "vae": [audio_vae, 0]}
    )
    _save(graph, [images, 0], [audio, 0], FPS, prefix)
    return graph


def upscale_graph(video: str, short_side: int, seed: int, prefix: str) -> dict:
    """SeedVR2 7B, a one-step restoration model, enlarging a clip so its short
    side is short_side pixels; the soundtrack is kept."""
    graph: dict = {}
    loaded = _add(graph, "LoadVideo", {"file": video})
    parts = _add(graph, "GetVideoComponents", {"video": [loaded, 0]})
    resized = _add(
        graph,
        "ResizeImageMaskNode",
        {
            "input": [parts, 0],
            "resize_type": "scale shorter dimension",
            "resize_type.shorter_size": short_side,
            "scale_method": "lanczos",
        },
    )
    padded = _add(graph, "SeedVR2Preprocess", {"resized_images": [resized, 0]})
    vae = _add(graph, "VAELoader", {"vae_name": _model_name("upscaler_vae")})
    tiles = {
        "tile_size": 512,
        "overlap": 128,
        "temporal_size": 64,
        "temporal_overlap": 8,
    }
    encoded = _add(
        graph, "VAEEncodeTiled", {"pixels": [padded, 0], "vae": [vae, 0]} | tiles
    )
    # Chunks sized to free VRAM, so long clips fit.
    chunks = _add(
        graph,
        "SeedVR2TemporalChunk",
        {"latent": [encoded, 0], "temporal_overlap": 0, "chunking_mode": "auto"},
    )
    unet = _add(
        graph,
        "UNETLoader",
        {"unet_name": _model_name("upscaler"), "weight_dtype": "default"},
    )
    conditioning = _add(
        graph,
        "SeedVR2Conditioning",
        {"model": [unet, 0], "vae_conditioning": [chunks, 0]},
    )
    sampled = _add(
        graph,
        "KSampler",
        {
            "model": [unet, 0],
            "positive": [conditioning, 0],
            "negative": [conditioning, 1],
            "latent_image": [chunks, 0],
            "seed": seed,
            "steps": 1,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    )
    merged = _add(
        graph,
        "SeedVR2TemporalMerge",
        {"latents": [sampled, 0], "temporal_overlap": [chunks, 1]},
    )
    decoded = _add(
        graph, "VAEDecodeTiled", {"samples": [merged, 0], "vae": [vae, 0]} | tiles
    )
    # "lab" colour correction garbles output on ROCm; ComfyUI's template uses none.
    fixed = _add(
        graph,
        "SeedVR2PostProcessing",
        {
            "images": [decoded, 0],
            "original_resized_images": [resized, 0],
            "color_correction_method": "none",
        },
    )
    _save(graph, [fixed, 0], [parts, 1], [parts, 2], prefix)
    return graph


# Talking to ComfyUI


def _request(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"ComfyUI refused the workflow: {error.read().decode()}"
        ) from error


def submit(base_url: str, graph: dict) -> str:
    """Queue a workflow; returns its prompt id."""
    return _request(f"{base_url}/prompt", {"prompt": graph})["prompt_id"]


def cancel(base_url: str, prompt_id: str) -> None:
    """Drop a queued workflow, or stop it if it is running."""
    _request(f"{base_url}/queue", {"delete": [prompt_id]})
    _request(f"{base_url}/interrupt", {"prompt_id": prompt_id})


# Lines a dying ComfyUI writes; the process can linger after them.
CRASHES = ("Fatal Python error", "Memory access fault")


def progress_line(log_tail: str) -> str:
    """The sampler's last 'step N/M' from the server's tqdm output, if any."""
    found = re.findall(r"(\d+)/(\d+) \[(\d+:\d+)<", log_tail)
    if not found:
        return ""
    step, total, elapsed = found[-1]
    return f"step {step}/{total} ({elapsed})"


def wait(
    base_url: str,
    prompt_id: str,
    log: Path | None = None,
    on_progress: Callable[[str], None] = lambda _: None,
) -> Path:
    """Wait for a queued workflow and return the video it saved.

    log is the server's log; its tqdm lines become progress messages.
    """
    offset = log.stat().st_size if log and log.is_file() else 0
    last = ""
    while True:
        entry = _request(f"{base_url}/history/{prompt_id}").get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                details = [
                    m[1]
                    for m in status.get("messages", [])
                    if m[0] == "execution_error"
                ]
                problem = details[0] if details else {}
                raise RuntimeError(
                    f"{problem.get('node_type', 'ComfyUI')} failed: "
                    f"{problem.get('exception_message', 'see the server log').strip()}"
                )
            # Loaded inputs are listed too, as previews of type "input".
            for output in entry.get("outputs", {}).values():
                for files in output.values():
                    for item in files if isinstance(files, list) else []:
                        if isinstance(item, dict) and item.get("type") == "output":
                            folder = data_dir() / "output" / item.get("subfolder", "")
                            return folder / item["filename"]
            raise RuntimeError("ComfyUI finished without saving a video.")
        if log and log.is_file():
            with log.open("rb") as file:
                file.seek(offset)
                tail = file.read().decode(errors="replace")
            if crash := next((c for c in CRASHES if c in tail), ""):
                raise RuntimeError(f"ComfyUI crashed ({crash}); see {log}")
            if (message := progress_line(tail)) and message != last:
                on_progress(message)
                last = message
        time.sleep(2)


# Media files: ffmpeg prepares inputs and joins clips.

ENCODE = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium"]
ENCODE += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]


def _run(tool: str, *args: str) -> str:
    if not shutil.which(tool):
        raise FileNotFoundError(f"{tool} not found; install ffmpeg to work on videos.")
    try:
        return subprocess.run(
            [tool, "-v", "error", *args], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{tool} failed: {error.stderr.strip()}") from error


def _ffmpeg(*args: str) -> None:
    _run("ffmpeg", "-y", *args)


def probe(path: Path) -> tuple[int, int, float]:
    """Width, height and duration in seconds of a video."""
    info = json.loads(
        _run(
            "ffprobe",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        )
    )
    stream = info["streams"][0]
    return (
        int(stream["width"]),
        int(stream["height"]),
        float(info["format"]["duration"]),
    )


def has_audio(path: Path) -> bool:
    return bool(
        _run(
            "ffprobe",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ).strip()
    )


def _frames(path: Path) -> int:
    return int(
        _run(
            "ffprobe",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ).strip()
    )


def _input_file(path: Path, suffix: str) -> tuple[str, Path]:
    """A name in ComfyUI's input folder taken from the file's content, so the same
    file is copied once, and where it goes."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    name = f"lct-{digest}{suffix}"
    target = data_dir() / "input" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    return name, target


def stage(path: Path) -> str:
    """Copy an image or audio file into ComfyUI's input folder; returns its name there."""
    name, target = _input_file(path, path.suffix.lower())
    if not target.is_file():
        shutil.copyfile(path, target)
    return name


def stage_video(path: Path) -> str:
    """A video as H3 reads references, at 24 fps and at most 15 s, in ComfyUI's
    input folder; returns its name there."""
    name, target = _input_file(path, ".mp4")
    if not target.is_file():
        streams = ["-map", "0:v:0", "-map", "0:a:0?"]
        _ffmpeg(
            "-i",
            str(path),
            "-t",
            str(MAX_SECONDS),
            "-vf",
            f"fps={FPS}",
            *streams,
            *ENCODE,
            str(target),
        )
    return name


def stage_tail(path: Path) -> str:
    """The last GUIDE_FRAMES frames of a video at 24 fps, with their sound, in
    ComfyUI's input folder, to anchor a continuation; returns its name there."""
    name, target = _input_file(path, "-tail.mp4")
    if target.is_file():
        return name
    end = target.with_name(f"{target.stem}-end.mp4")
    streams = ["-map", "0:v:0", "-map", "0:a:0?"]
    _ffmpeg(
        "-sseof",
        "-3",
        "-i",
        str(path),
        "-vf",
        f"fps={FPS}",
        *streams,
        *ENCODE,
        str(end),
    )
    start = _frames(end) - GUIDE_FRAMES
    if start < 0:
        end.unlink()
        raise ValueError(f"{path} is too short to extend; it needs at least 1 second.")
    trim = ["-vf", f"trim=start_frame={start},setpts=PTS-STARTPTS"]
    if has_audio(end):
        trim += ["-af", f"atrim=start={start / FPS},asetpts=PTS-STARTPTS"]
    _ffmpeg("-i", str(end), *trim, *streams, *ENCODE, str(target))
    end.unlink()
    return name


def join(clips: list[Path], out: Path, skip: int = 0) -> Path:
    """Join clips end to end at 24 fps and the first clip's size; skip drops that
    many frames from the start of every clip after the first (an extension's
    re-anchored frames). Clips without sound get silence."""
    width, height, _ = probe(clips[0])
    inputs, chains, pairs = [], [], ""
    for i, clip in enumerate(clips):
        inputs += ["-i", str(clip)]
        start = skip if i else 0
        chains.append(
            f"[{i}:v]fps={FPS},trim=start_frame={start},setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
        )
        if has_audio(clip):
            chains.append(
                f"[{i}:a]atrim=start={start / FPS},asetpts=PTS-STARTPTS,"
                f"aresample=48000,aformat=channel_layouts=stereo[a{i}]"
            )
        else:
            seconds = probe(clip)[2] - start / FPS
            chains.append(
                f"anullsrc=r=48000:cl=stereo,atrim=duration={seconds:.3f}[a{i}]"
            )
        pairs += f"[v{i}][a{i}]"
    graph = ";".join([*chains, f"{pairs}concat=n={len(clips)}:v=1:a=1[v][a]"])
    out.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        *inputs,
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "[a]",
        *ENCODE,
        str(out),
    )
    return out
