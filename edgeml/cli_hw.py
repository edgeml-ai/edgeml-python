from __future__ import annotations

import json as json_mod
import logging
import re
import sys

import click

logger = logging.getLogger(__name__)


@click.group()
def hw() -> None:
    """Hardware detection and optimization."""


@hw.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def detect(as_json: bool) -> None:
    """Detect GPU, CPU, and RAM capabilities."""
    from .hardware import detect_hardware

    profile = detect_hardware(force=True)

    if as_json:
        import dataclasses

        def _serialize(obj: object) -> object:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return dataclasses.asdict(obj)
            if hasattr(obj, "value"):
                return obj.value
            return str(obj)

        click.echo(
            json_mod.dumps(dataclasses.asdict(profile), indent=2, default=_serialize)
        )
        return

    # Pretty print
    click.secho("\n  Hardware Profile\n", bold=True)

    # GPU
    if profile.gpu and profile.gpu.gpus:
        click.secho("  GPU", bold=True)
        for gpu in profile.gpu.gpus:
            click.echo(f"    [{gpu.index}] {gpu.name}")
            click.echo(
                f"        VRAM: {gpu.memory.total_gb:.1f} GB "
                f"(free: {gpu.memory.free_gb:.1f} GB)"
            )
            if gpu.compute_capability:
                click.echo(f"        Compute: {gpu.compute_capability}")
            if gpu.architecture:
                click.echo(f"        Arch: {gpu.architecture}")
        click.echo(f"    Total VRAM: {profile.gpu.total_vram_gb:.1f} GB")
        click.echo(f"    Backend: {profile.gpu.backend}")
        if profile.gpu.driver_version:
            click.echo(f"    Driver: {profile.gpu.driver_version}")
        if profile.gpu.cuda_version:
            click.echo(f"    CUDA: {profile.gpu.cuda_version}")
        if profile.gpu.rocm_version:
            click.echo(f"    ROCm: {profile.gpu.rocm_version}")
        if profile.gpu.detection_method:
            click.echo(f"    Detection: {profile.gpu.detection_method}")
    else:
        click.secho("  GPU: None detected", fg="yellow")

    click.echo()

    # CPU
    click.secho("  CPU", bold=True)
    click.echo(f"    {profile.cpu.brand}")
    click.echo(f"    Cores: {profile.cpu.cores} ({profile.cpu.threads} threads)")
    click.echo(f"    Speed: {profile.cpu.base_speed_ghz:.2f} GHz")
    click.echo(f"    Arch: {profile.cpu.architecture}")
    features = []
    if profile.cpu.has_avx512:
        features.append("AVX-512")
    if profile.cpu.has_avx2:
        features.append("AVX2")
    if profile.cpu.has_neon:
        features.append("NEON")
    if features:
        click.echo(f"    Features: {', '.join(features)}")
    click.echo(f"    Est. GFLOPS: {profile.cpu.estimated_gflops:.1f}")

    click.echo()

    # RAM
    click.secho("  Memory", bold=True)
    click.echo(f"    Total: {profile.total_ram_gb:.1f} GB")
    click.echo(f"    Available: {profile.available_ram_gb:.1f} GB")

    click.echo()
    click.echo(f"  Best backend: {profile.best_backend}")

    # Diagnostics
    if profile.diagnostics:
        click.echo()
        click.secho("  Diagnostics", bold=True, fg="yellow")
        for d in profile.diagnostics:
            click.echo(f"    \u2022 {d}")

    click.echo()


@hw.command()
@click.argument("model", required=False)
@click.option(
    "--context-length",
    "-c",
    default=4096,
    type=int,
    help="Context length in tokens.",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option(
    "--priority",
    "-p",
    type=click.Choice(["speed", "quality", "balanced"]),
    default="balanced",
    help="Optimization priority.",
)
def optimize(
    model: str | None, context_length: int, as_json: bool, priority: str
) -> None:
    """Optimize inference configuration for your hardware.

    With MODEL: show optimal quantization and settings for a specific model.
    Without MODEL: show general recommendations for your hardware.
    """
    from .hardware import detect_hardware
    from .model_optimizer import ModelOptimizer

    profile = detect_hardware()
    optimizer = ModelOptimizer(profile)

    if model:
        # Parse model size from model tag (e.g., "llama3.1:8b" -> 8.0)
        model_size_b = _parse_model_size(model)
        if model_size_b is None:
            click.echo(
                f"Cannot determine model size from '{model}'. "
                "Specify a model with a known size (e.g., llama3.1:8b, mistral:7b).",
                err=True,
            )
            sys.exit(1)

        config = optimizer.pick_quant_and_offload(model_size_b, context_length)
        speed = optimizer.predict_speed(model_size_b, config)
        env = optimizer.env_vars(model_size_b)
        cmd = optimizer.serve_command(model, config)

        if as_json:
            import dataclasses

            click.echo(
                json_mod.dumps(
                    {
                        "model": model,
                        "model_size_b": model_size_b,
                        "context_length": context_length,
                        "config": dataclasses.asdict(config),
                        "speed": dataclasses.asdict(speed),
                        "env_vars": env,
                        "command": cmd,
                    },
                    indent=2,
                    default=str,
                )
            )
            return

        click.secho(f"\n  Optimization for {model}\n", bold=True)
        click.echo(f"    Model size: {model_size_b}B params")
        click.echo(f"    Context: {context_length} tokens")
        click.echo(f"    Quantization: {config.quantization}")
        click.echo(f"    Strategy: {config.strategy.value}")
        click.echo(
            f"    GPU layers: {'all' if config.gpu_layers == -1 else config.gpu_layers}"
        )
        click.echo(f"    VRAM usage: {config.vram_gb:.1f} GB")
        click.echo(f"    RAM usage: {config.ram_gb:.1f} GB")
        click.echo(
            f"    Est. speed: {speed.tokens_per_second:.1f} tok/s "
            f"({speed.confidence} confidence)"
        )
        if config.warning:
            click.secho(f"    Warning: {config.warning}", fg="yellow")
        click.echo()
        click.secho("  Run command:", bold=True)
        click.echo(f"    {cmd}")
        click.echo()
        if env:
            click.secho("  Environment variables:", bold=True)
            for k, v in env.items():
                click.echo(f"    export {k}={v}")
        click.echo()
    else:
        # General recommendations
        recs = optimizer.recommend(priority=priority)
        env = optimizer.env_vars()

        if as_json:
            import dataclasses

            click.echo(
                json_mod.dumps(
                    {
                        "priority": priority,
                        "recommendations": [dataclasses.asdict(r) for r in recs],
                        "env_vars": env,
                    },
                    indent=2,
                    default=str,
                )
            )
            return

        click.secho(f"\n  Recommendations ({priority} priority)\n", bold=True)
        for rec in recs:
            if rec.speed.tokens_per_second >= 15:
                speed_color = "green"
            elif rec.speed.tokens_per_second >= 5:
                speed_color = "yellow"
            else:
                speed_color = "red"
            click.echo(f"    {rec.model_size} @ {rec.quantization}")
            click.echo(f"      {rec.reason}")
            click.secho(
                f"      Speed: {rec.speed.tokens_per_second:.1f} tok/s "
                f"({rec.speed.confidence})",
                fg=speed_color,
            )
            click.echo(f"      $ {rec.ollama_command}")
            click.echo()

        if env:
            click.secho("  Environment variables:", bold=True)
            for k, v in env.items():
                click.echo(f"    export {k}={v}")
        click.echo()


def _parse_model_size(model_tag: str) -> float | None:
    """Extract model size in billions from a model tag like 'llama3.1:8b' or 'mistral:7b'."""
    tag = model_tag.lower().replace("-", "").replace("_", "")

    # Try to find NB pattern (e.g., "8b", "70b", "0.5b")
    match = re.search(r"(\d+\.?\d*)b", tag)
    if match:
        return float(match.group(1))

    # Common model name -> size mappings
    SIZE_MAP: dict[str, float] = {
        "phi-mini": 3.8,
        "phi-medium": 14.0,
        "gemma-1b": 1.0,
        "gemma-2b": 2.0,
        "gemma-7b": 7.0,
        "mistral": 7.0,
        "mixtral": 46.7,
        "llama2": 7.0,
        "llama3": 8.0,
        "qwen2": 7.0,
        "deepseek-coder": 6.7,
        "smollm": 0.36,
    }

    for name, size in SIZE_MAP.items():
        if name.replace("-", "") in tag:
            return size

    return None


# Command to register in main CLI
def interactive_cmd_factory(cli_group: click.Group) -> click.Command:
    """Create the interactive command that needs a reference to the CLI group."""

    @click.command("interactive")
    def interactive() -> None:
        """Open interactive command panel."""
        from .interactive import launch_interactive

        launch_interactive(cli_group)

    return interactive
