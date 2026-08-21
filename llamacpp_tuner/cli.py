"""CLI commands for lct."""

import sys

import click

from llamacpp_tuner import __version__
from llamacpp_tuner.benchmark import BenchmarkResult, benchmark_server
from llamacpp_tuner.cache import load_cache, save_cache
from llamacpp_tuner.calculator import calculate_optimal_args, format_args
from llamacpp_tuner.constants import (
    DEFAULT_BENCHMARK_PORT,
    DEFAULT_CTX_SIZE,
    DEFAULT_HOST,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PORT,
    DEFAULT_QUANT,
)
from llamacpp_tuner.downloader import (
    download_mmproj,
    download_model,
    get_mmproj_path,
    list_downloaded_models,
    list_invalid_models,
    list_valid_models,
    remove_model,
)
from llamacpp_tuner.exceptions import (
    HardwareNotDetectedError,
    LlamaNotInstalledError,
    ModelNotFoundError,
)
from llamacpp_tuner.hardware import HardwareProfile, format_hardware
from llamacpp_tuner.llama import (
    get_llama_binary,
    install_llama,
    is_llama_installed,
    run_server,
)
from llamacpp_tuner.llama import (
    has_gpu_support as llama_has_gpu_support,
)
from llamacpp_tuner.types import QUANT_CHOICES
from llamacpp_tuner.validators import (
    validate_hardware_cache,
    validate_llama_installed,
    validate_model_path,
)


def _handle_error(error: Exception) -> None:
    """Handle errors in CLI commands."""
    print(str(error), file=sys.stderr)
    sys.exit(1)


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Auto-optimize llama.cpp parameters for your hardware."""
    pass


@main.command()
def setup() -> None:
    """Setup lct: detect hardware and install llama.cpp."""
    from llamacpp_tuner.hardware import detect_hardware

    print("Detecting hardware...")
    hardware = detect_hardware()
    print(format_hardware(hardware))
    print("")

    save_cache(
        "hardware",
        {
            "gpus": [
                {"name": g.name, "vram_mb": g.vram_mb, "compute": g.compute_capability}
                for g in hardware.gpus
            ],
            "cpu_cores": hardware.cpu_cores,
            "cpu_threads": hardware.cpu_threads,
            "ram_mb": hardware.ram_mb,
            "backend": hardware.backend,
        },
    )
    print("Hardware profile saved to cache.")
    print("")

    if is_llama_installed():
        binary = get_llama_binary()
        has_gpu = llama_has_gpu_support()

        print(f"llama.cpp already installed: {binary}")
        print(f"GPU support: {'Yes' if has_gpu else 'No'}")

        if hardware.gpus and not has_gpu:
            print("")
            print("GPU detected but current binary doesn't have GPU support!")
            print("Rebuilding with GPU support...")
            try:
                from llamacpp_tuner.llama import build_from_source

                binary = build_from_source(force=True)
                print(f"\nRebuilt with GPU support: {binary}")
            except Exception as e:
                print(f"Failed to rebuild: {e}", file=sys.stderr)
                print("Continuing with existing binary...")
    else:
        print("Installing llama.cpp...")
        try:
            install_llama()
        except Exception as e:
            print(f"Failed to install llama.cpp: {e}", file=sys.stderr)
            sys.exit(1)


@main.command()
@click.argument("repo_id")
@click.option(
    "--quant",
    "-q",
    type=click.Choice(QUANT_CHOICES),
    default=DEFAULT_QUANT,
    show_default=True,
)
@click.option("--force", "-f", is_flag=True, help="Re-download if already exists")
def pull(repo_id: str, quant: str, force: bool) -> None:
    """Download a model (and mmproj if available) from HuggingFace Hub."""
    print(f"Looking for {quant} in {repo_id}...")
    path = download_model(repo_id, quant, force=force)  # type: ignore[arg-type]

    if not path:
        print(f"Could not find {quant} in {repo_id}", file=sys.stderr)
        sys.exit(1)

    if force or not path.exists():
        print(f"Model downloaded: {path}")
    else:
        print(f"Model already exists: {path}")

    mmproj_path = get_mmproj_path(repo_id)
    if mmproj_path and not force:
        print(f"mmproj already exists: {mmproj_path}")
    elif mmproj_path := download_mmproj(repo_id, force=force):
        print(f"mmproj downloaded: {mmproj_path}")


@main.command()
@click.argument("repo_id")
@click.option(
    "--quant",
    "-q",
    type=click.Choice(QUANT_CHOICES),
    default=DEFAULT_QUANT,
    show_default=True,
)
@click.option(
    "--ctx", "-c", default=DEFAULT_CTX_SIZE, show_default=True, help="Context size"
)
def args(repo_id: str, quant: str, ctx: int) -> None:
    """Show optimal llama.cpp arguments for a model."""
    try:
        hardware = validate_hardware_cache()
        model_path = validate_model_path(repo_id, quant)  # type: ignore[arg-type]
    except (HardwareNotDetectedError, ModelNotFoundError) as e:
        _handle_error(e)

    optimal, warnings = calculate_optimal_args(
        hardware=hardware,
        model_name=repo_id,
        quant=quant,  # type: ignore[arg-type]
        ctx_size=ctx,
        llama_has_gpu=llama_has_gpu_support(),
    )

    print(f"Model: {repo_id}")
    print(f"Quantization: {quant}")
    print("")
    print("Optimal arguments:")
    print(format_args(optimal))
    print("")

    mmproj_path = get_mmproj_path(repo_id)
    if mmproj_path:
        print(f"Multimodal: {mmproj_path}")
        print("")

    print("Full command:")
    cmd_args = optimal.to_list(str(model_path))
    cmd_args.extend(["--host", DEFAULT_HOST, "--port", str(DEFAULT_PORT)])
    if mmproj_path:
        cmd_args.extend(["--mmproj", str(mmproj_path)])
    print(f"llama-server {' '.join(cmd_args)}")

    if warnings:
        print("")
        for w in warnings:
            click.echo(click.style(f"Warning: {w}", fg="yellow"))


@main.command()
@click.argument("repo_id")
@click.option(
    "--quant",
    "-q",
    type=click.Choice(QUANT_CHOICES),
    default=DEFAULT_QUANT,
    show_default=True,
)
@click.option(
    "--ctx", "-c", default=DEFAULT_CTX_SIZE, show_default=True, help="Context size"
)
@click.option(
    "--port", "-p", default=DEFAULT_PORT, show_default=True, help="Server port"
)
@click.option(
    "--host", "-h", default=DEFAULT_HOST, show_default=True, help="Server host"
)
@click.option(
    "--extra-args",
    "-e",
    type=str,
    default="",
    help="Extra args to pass to llama-server (e.g., '--no-prefill-assistant')",
)
@click.option(
    "--no-mmproj",
    is_flag=True,
    default=False,
    help="Disable multimodal support (don't use mmproj)",
)
@click.option(
    "--no-mmproj-offload",
    is_flag=True,
    default=False,
    help="Disable GPU offloading for mmproj",
)
@click.option(
    "--default",
    "use_default",
    is_flag=True,
    default=False,
    help="Use llama.cpp default args instead of optimized args",
)
def serve(
    repo_id: str,
    quant: str,
    ctx: int,
    port: int,
    host: str,
    extra_args: str,
    no_mmproj: bool,
    no_mmproj_offload: bool,
    use_default: bool,
) -> None:
    """Run llama.cpp server with optimal arguments."""
    try:
        validate_llama_installed()
        hardware = validate_hardware_cache()
        model_path = validate_model_path(repo_id, quant)  # type: ignore[arg-type]
    except (LlamaNotInstalledError, HardwareNotDetectedError, ModelNotFoundError) as e:
        _handle_error(e)

    if use_default:
        print("Using default llama.cpp arguments")
        cmd_args = ["-m", str(model_path), "-c", str(ctx)]
    else:
        optimal, warnings = calculate_optimal_args(
            hardware=hardware,
            model_name=repo_id,
            quant=quant,  # type: ignore[arg-type]
            ctx_size=ctx,
            llama_has_gpu=llama_has_gpu_support(),
        )

        if warnings:
            for w in warnings:
                click.echo(click.style(f"Warning: {w}", fg="yellow"))
            print("")

        cmd_args = optimal.to_list(str(model_path))

    cmd_args.extend(["--host", host, "--port", str(port)])

    mmproj_path = None
    if not no_mmproj:
        mmproj_path = get_mmproj_path(repo_id)
        if not mmproj_path:
            mmproj_path = download_mmproj(repo_id)
        if mmproj_path:
            cmd_args.extend(["--mmproj", str(mmproj_path)])
            print(f"Multimodal: {mmproj_path}")
    else:
        print("Multimodal: disabled")

    if mmproj_path and no_mmproj_offload:
        cmd_args.append("--no-mmproj-offload")

    if extra_args:
        cmd_args.extend(extra_args.split())

    print(f"Starting server on {host}:{port}...")
    run_server(cmd_args)


@main.command()
@click.argument("repo_id")
@click.option(
    "--quant",
    "-q",
    type=click.Choice(QUANT_CHOICES),
    default=DEFAULT_QUANT,
    show_default=True,
)
@click.option(
    "--ctx", "-c", default=DEFAULT_CTX_SIZE, show_default=True, help="Context size"
)
@click.option(
    "--port",
    "-p",
    default=DEFAULT_BENCHMARK_PORT,
    show_default=True,
    help="Benchmark port",
)
@click.option(
    "--args",
    "-a",
    type=str,
    default="",
    help="Server args as string (e.g., '-ngl -1 -t 8'), or 'optimal' for auto-optimized",
)
@click.option(
    "--max-tokens",
    default=DEFAULT_MAX_TOKENS,
    show_default=True,
    help="Max tokens to generate",
)
def benchmark(
    repo_id: str,
    quant: str,
    ctx: int,
    port: int,
    args: str,
    max_tokens: int,
) -> None:
    """Benchmark TPS with given args. Use --args optimal for optimized args."""
    try:
        validate_llama_installed()
        hardware = validate_hardware_cache()
        model_path = validate_model_path(repo_id, quant)  # type: ignore[arg-type]
    except (LlamaNotInstalledError, HardwareNotDetectedError, ModelNotFoundError) as e:
        _handle_error(e)

    server_args: list[str] = []
    if args == "optimal":
        optimal, warnings = calculate_optimal_args(
            hardware=hardware,
            model_name=repo_id,
            quant=quant,  # type: ignore[arg-type]
            ctx_size=ctx,
            llama_has_gpu=llama_has_gpu_support(),
        )
        server_args = optimal.to_list(str(model_path))
        server_args = [a for a in server_args if a not in ("-m", str(model_path))]
        print("Using optimized arguments")
        for w in warnings:
            click.echo(click.style(f"Warning: {w}", fg="yellow"))
        print("")
    elif args:
        server_args = args.split()
        print(f"Using custom args: {args}")
    else:
        print("Using default llama.cpp arguments")

    print(f"Benchmarking {repo_id} ({quant}) with context {ctx}...")
    print("")

    result = benchmark_server(
        model_path=str(model_path),
        server_args=server_args,
        port=port,
        max_tokens=max_tokens,
    )

    if result:
        print(result.summary())
        print("")
        click.echo(
            click.style(f"Generation TPS: {result.gen_tps:.2f}", fg="green", bold=True)
        )
    else:
        print("Benchmark failed. Check server logs.", file=sys.stderr)
        sys.exit(1)


@main.command()
@click.argument("repo_id")
@click.option(
    "--quant",
    "-q",
    type=click.Choice(QUANT_CHOICES),
    default=DEFAULT_QUANT,
    show_default=True,
)
@click.option(
    "--ctx", "-c", default=DEFAULT_CTX_SIZE, show_default=True, help="Context size"
)
@click.option(
    "--config",
    "-C",
    type=str,
    multiple=True,
    default=["default", "optimal"],
    help="Configurations to compare: 'default', 'optimal', or custom args string",
)
@click.option(
    "--max-tokens",
    default=DEFAULT_MAX_TOKENS,
    show_default=True,
    help="Max tokens to generate",
)
def compare(
    repo_id: str, quant: str, ctx: int, config: tuple[str, ...], max_tokens: int
) -> None:
    """Compare TPS across multiple configurations."""
    try:
        validate_llama_installed()
        hardware = validate_hardware_cache()
        model_path = validate_model_path(repo_id, quant)  # type: ignore[arg-type]
    except (LlamaNotInstalledError, HardwareNotDetectedError, ModelNotFoundError) as e:
        _handle_error(e)

    print(f"Comparing configurations for {repo_id} ({quant}) with context {ctx}...")
    print("")

    results: list[tuple[str, BenchmarkResult | None]] = []
    port = DEFAULT_BENCHMARK_PORT

    for cfg_name in config:
        server_args: list[str] = []
        display_name = cfg_name

        if cfg_name == "optimal":
            optimal, _ = calculate_optimal_args(
                hardware=hardware,
                model_name=repo_id,
                quant=quant,  # type: ignore[arg-type]
                ctx_size=ctx,
                llama_has_gpu=llama_has_gpu_support(),
            )
            server_args = optimal.to_list(str(model_path))
            server_args = [a for a in server_args if a not in ("-m", str(model_path))]
        elif cfg_name == "default":
            server_args = ["-c", str(ctx)]
        else:
            display_name = f"custom: {cfg_name[:40]}"
            server_args = cfg_name.split()
            if "-c" not in server_args and "--ctx-size" not in server_args:
                server_args = ["-c", str(ctx)] + server_args

        print(f"Testing: {display_name}")
        if cfg_name == "optimal":
            print(f"  Args: {' '.join(server_args[:20])}...")
        result = benchmark_server(
            model_path=str(model_path),
            server_args=server_args,
            port=port,
            max_tokens=max_tokens,
        )
        results.append((cfg_name, result))
        port += 1

        if result:
            print(
                f"  Gen TPS: {result.gen_tps:.2f}, Prompt TPS: {result.prompt_tps:.2f}"
            )
        else:
            print("  Failed")
        print("")

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Configuration':<30} {'Gen TPS':>10} {'Prompt TPS':>12}")
    print("-" * 60)

    for name, result in results:
        if result:
            print(f"{name:<30} {result.gen_tps:>10.2f} {result.prompt_tps:>12.2f}")
        else:
            print(f"{name:<30} {'Failed':>10} {'Failed':>12}")


@main.command()
def models() -> None:
    """List downloaded models with status."""
    valid = list_valid_models()
    invalid = list_invalid_models()

    if not valid and not invalid:
        print("No models downloaded yet.")
        return

    if valid:
        print("Valid models:")
        for m in valid:
            size_mb = m.stat().st_size / (1024 * 1024)
            print(f"  {m.name} ({size_mb:.1f} MB)")

    if invalid:
        print("")
        print("Incomplete/corrupted models:")
        for m in invalid:
            size_kb = m.stat().st_size / 1024
            print(f"  {m.name} ({size_kb:.1f} KB) - incomplete")
        print("")
        print("Run 'lct clean' to remove incomplete models.")


@main.command()
@click.option("--force", "-f", is_flag=True, help="Remove without confirmation")
def clean(force: bool) -> None:
    """Remove incomplete or corrupted model downloads."""
    invalid = list_invalid_models()

    if not invalid:
        print("No incomplete or corrupted models found.")
        return

    print("Incomplete/corrupted models:")
    total_kb = 0
    for m in invalid:
        size_kb = m.stat().st_size / 1024
        total_kb += size_kb
        print(f"  {m.name} ({size_kb:.1f} KB)")

    print(f"\nTotal: {len(invalid)} files, {total_kb:.1f} KB")

    if not force and not click.confirm("\nRemove these files?", default=False):
        print("Cancelled.")
        return

    removed = 0
    for m in invalid:
        if remove_model(m):
            removed += 1

    print(f"Removed {removed} incomplete model(s).")


@main.command()
def status() -> None:
    """Show current status: hardware and installation."""
    cached = load_cache("hardware")
    if cached:
        hardware = HardwareProfile.from_dict(cached)
        print("Cached hardware profile:")
        print(format_hardware(hardware))
    else:
        print("No hardware profile cached. Run 'lct setup'.")

    print("")

    if is_llama_installed():
        binary = get_llama_binary()
        print(f"llama.cpp: {binary}")
    else:
        print("llama.cpp: not installed")

    print("")

    downloaded = list_downloaded_models()
    if downloaded:
        print(f"Downloaded models: {len(downloaded)}")
        for m in downloaded:
            size_mb = m.stat().st_size / (1024 * 1024)
            is_mmproj = "mmproj" in m.name.lower()
            mmproj_tag = " [mmproj]" if is_mmproj else ""
            print(f"  - {m.name} ({size_mb:.1f} MB){mmproj_tag}")
    else:
        print("No models downloaded.")


if __name__ == "__main__":
    main()
