"""
EdgeML command-line interface.

Usage::

    edgeml serve gemma-1b --port 8080
    edgeml deploy gemma-1b --phone
    edgeml dashboard
    edgeml push model.pt --name sentiment-v1 --version 1.0.0
    edgeml pull sentiment-v1 --version 1.0.0 --format coreml
    edgeml check model.pt
    edgeml convert model.pt --target ios,android
    edgeml status sentiment-v1
    edgeml benchmark gemma-1b --share
    edgeml login
"""

from __future__ import annotations

import os
import sys
import webbrowser
from typing import Optional

import click


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Read API key from env, keychain, or raise."""
    key = os.environ.get("EDGEML_API_KEY", "")
    if not key:
        config_path = os.path.expanduser("~/.edgeml/credentials")
        if os.path.exists(config_path):
            with open(config_path) as f:
                for line in f:
                    if line.startswith("api_key="):
                        key = line.split("=", 1)[1].strip()
                        break
    return key


def _require_api_key() -> str:
    key = _get_api_key()
    if not key:
        click.echo("No API key found. Run `edgeml login` first.", err=True)
        sys.exit(1)
    return key


def _get_client():  # type: ignore[no-untyped-def]
    from .client import Client

    return Client(api_key=_require_api_key())


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="1.0.0", prog_name="edgeml")
def main() -> None:
    """EdgeML — serve, deploy, and observe ML models on edge devices."""


# ---------------------------------------------------------------------------
# edgeml serve
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model")
@click.option("--port", "-p", default=8080, help="Port to listen on.")
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--benchmark", is_flag=True, help="Run latency benchmark on startup.")
@click.option(
    "--share", is_flag=True, help="Share anonymous benchmark data with EdgeML."
)
def serve(model: str, port: int, host: str, benchmark: bool, share: bool) -> None:
    """Start a local OpenAI-compatible inference server.

    Serves MODEL via the best available backend (mlx-lm on Apple Silicon,
    llama.cpp on other platforms). No account required.

    Example:

        edgeml serve gemma-1b --port 8080

        curl localhost:8080/v1/chat/completions \\
            -d '{"model":"gemma-1b","messages":[{"role":"user","content":"Hi"}]}'
    """
    api_key = _get_api_key() if share else None
    api_base = os.environ.get("EDGEML_API_BASE", "https://api.edgeml.io/api/v1")

    click.echo(f"Starting EdgeML serve on {host}:{port}")
    click.echo(f"Model: {model}")
    click.echo(f"OpenAI-compatible API: http://localhost:{port}/v1/chat/completions")
    click.echo(f"Health check: http://localhost:{port}/health")

    if benchmark:
        click.echo("Benchmark mode: will run latency test after model loads.")

    if share and not api_key:
        click.echo(
            "Warning: --share requires an API key to upload benchmark data. "
            "Run `edgeml login` or set EDGEML_API_KEY.",
            err=True,
        )

    from .serve import run_server

    run_server(model, port=port, host=host, api_key=api_key, api_base=api_base)


# ---------------------------------------------------------------------------
# edgeml check
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--devices",
    "-d",
    default=None,
    help="Comma-separated device profiles (e.g. iphone_15_pro,pixel_8).",
)
def check(model_path: str, devices: Optional[str]) -> None:
    """Check device compatibility for a local model file.

    Analyzes the model and reports which edge devices can run it,
    estimated latency, memory usage, and required optimizations.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from ml_service.conversion.device_compat import DeviceCompatibilityChecker
    except ImportError:
        click.echo(
            "Device compatibility checker not available. "
            "Run from the fed-learning repo root or install ml_service.",
            err=True,
        )
        sys.exit(1)

    checker = DeviceCompatibilityChecker()
    target_devices = devices.split(",") if devices else None

    click.echo(f"Checking compatibility: {model_path}")
    report = checker.check_compatibility(model_path, target_devices)

    click.echo(f"\nModel category: {report.get('category', 'unknown')}")
    click.echo(f"Summary: {report.get('summary', '')}")

    for rec in report.get("recommendations", []):
        click.echo(f"  - {rec}")


# ---------------------------------------------------------------------------
# edgeml convert
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--target",
    "-t",
    default="onnx",
    help="Comma-separated target formats: onnx, coreml, tflite.",
)
@click.option("--output", "-o", default="./converted", help="Output directory.")
def convert(model_path: str, target: str, output: str) -> None:
    """Convert a model to edge formats locally.

    Converts MODEL_PATH (PyTorch .pt) to target formats. Runs entirely
    on your machine — no account needed.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from ml_service.conversion.converter import ModelConverter
    except ImportError:
        click.echo(
            "Model converter not available. "
            "Run from the fed-learning repo root or install ml_service.",
            err=True,
        )
        sys.exit(1)

    formats = [f.strip() for f in target.split(",")]
    os.makedirs(output, exist_ok=True)

    click.echo(f"Converting {model_path} → {', '.join(formats)}")
    click.echo(f"Output: {output}")

    converter = ModelConverter()

    import torch

    model = torch.jit.load(model_path)
    sample = torch.randn(1, 3, 224, 224)  # Default; may need adjustment

    results = converter.convert_all(
        model=model,
        sample_input=sample,
        output_dir=output,
        model_name=os.path.splitext(os.path.basename(model_path))[0],
    )

    for fmt, path in results.items():
        click.echo(f"  {fmt}: {path}")

    click.echo("Done.")


# ---------------------------------------------------------------------------
# edgeml login
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--api-key", prompt="API key", hide_input=True, help="Your EdgeML API key."
)
def login(api_key: str) -> None:
    """Authenticate with EdgeML and store your API key.

    The key is stored in ~/.edgeml/credentials. You can also set the
    EDGEML_API_KEY environment variable instead.
    """
    config_dir = os.path.expanduser("~/.edgeml")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "credentials")

    with open(config_path, "w") as f:
        f.write(f"api_key={api_key}\n")

    os.chmod(config_path, 0o600)
    click.echo(f"API key saved to {config_path}")


# ---------------------------------------------------------------------------
# edgeml push
# ---------------------------------------------------------------------------


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--name", "-n", required=True, help="Model name.")
@click.option("--version", "-v", required=True, help="Semantic version (e.g. 1.0.0).")
@click.option("--description", "-d", default=None, help="Version description.")
@click.option(
    "--formats",
    "-f",
    default=None,
    help="Comma-separated target formats for server-side conversion.",
)
def push(
    file_path: str,
    name: str,
    version: str,
    description: Optional[str],
    formats: Optional[str],
) -> None:
    """Upload a model and trigger server-side conversion.

    Uploads FILE_PATH, registers it as NAME at VERSION, and optionally
    triggers conversion to mobile formats on the server.

    Example:

        edgeml push model.pt --name sentiment-v1 --version 1.0.0 --formats coreml,tflite
    """
    client = _get_client()
    click.echo(f"Pushing {file_path} as {name} v{version}...")

    result = client.push(
        file_path,
        name=name,
        version=version,
        description=description,
        formats=formats,
    )

    click.echo(f"Uploaded: {name} v{version}")
    for fmt, info in result.get("formats", {}).items():
        click.echo(f"  {fmt}: {info}")


# ---------------------------------------------------------------------------
# edgeml pull
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.option(
    "--version", "-v", default=None, help="Version to download. Defaults to latest."
)
@click.option(
    "--format",
    "-f",
    "fmt",
    default=None,
    help="Model format (onnx, coreml, tflite).",
)
@click.option("--output", "-o", default=".", help="Output directory.")
def pull(name: str, version: Optional[str], fmt: Optional[str], output: str) -> None:
    """Download a model from the registry.

    Downloads NAME at VERSION in the specified FORMAT to OUTPUT directory.

    Example:

        edgeml pull sentiment-v1 --version 1.0.0 --format coreml
    """
    client = _get_client()
    ver_str = version or "latest"
    click.echo(f"Pulling {name} v{ver_str}...")

    result = client.pull(name, version=version, format=fmt, destination=output)
    click.echo(f"Downloaded: {result['model_path']}")


# ---------------------------------------------------------------------------
# edgeml deploy
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.option(
    "--version", "-v", default=None, help="Version to deploy. Defaults to latest."
)
@click.option("--phone", is_flag=True, help="Deploy to your connected phone.")
@click.option("--rollout", "-r", default=100, help="Rollout percentage (1-100).")
@click.option(
    "--strategy",
    "-s",
    default="canary",
    type=click.Choice(["canary", "immediate"]),
    help="Rollout strategy.",
)
@click.option(
    "--target",
    "-t",
    default=None,
    help="Comma-separated target formats: ios, android.",
)
def deploy(
    name: str,
    version: Optional[str],
    phone: bool,
    rollout: int,
    strategy: str,
    target: Optional[str],
) -> None:
    """Deploy a model to edge devices.

    Deploys NAME at VERSION to devices. Use --phone for quick
    phone deployment, or --rollout for fleet percentage rollouts.

    Examples:

        edgeml deploy gemma-1b --phone
        edgeml deploy sentiment-v1 --rollout 10 --strategy canary
    """
    if phone:
        click.echo(f"Deploying {name} to phone...")
        click.echo("Scan the QR code in your EdgeML dashboard to connect your device.")
        click.echo("Opening dashboard...")
        dashboard_url = os.environ.get("EDGEML_DASHBOARD_URL", "https://app.edgeml.io")
        webbrowser.open(f"{dashboard_url}/deploy/phone?model={name}")
        return

    client = _get_client()
    click.echo(f"Deploying {name} at {rollout}% rollout ({strategy})...")

    result = client.deploy(
        name,
        version=version,
        rollout=rollout,
        strategy=strategy,
    )

    click.echo(f"Rollout created: {result.get('id', 'ok')}")
    click.echo(f"Status: {result.get('status', 'started')}")


# ---------------------------------------------------------------------------
# edgeml status
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
def status(name: str) -> None:
    """Show model status, active rollouts, and inference metrics.

    Example:

        edgeml status sentiment-v1
    """
    client = _get_client()
    info = client.status(name)

    model = info.get("model", {})
    click.echo(f"Model: {model.get('name', name)}")
    click.echo(f"ID: {model.get('id', 'unknown')}")
    click.echo(f"Framework: {model.get('framework', 'unknown')}")

    rollouts = info.get("active_rollouts", [])
    if rollouts:
        click.echo(f"\nActive rollouts: {len(rollouts)}")
        for r in rollouts:
            click.echo(
                f"  v{r.get('version', '?')} — "
                f"{r.get('rollout_percentage', 0)}% — "
                f"{r.get('status', 'unknown')}"
            )
    else:
        click.echo("\nNo active rollouts.")


# ---------------------------------------------------------------------------
# edgeml dashboard
# ---------------------------------------------------------------------------


@main.command()
def dashboard() -> None:
    """Open the EdgeML dashboard in your browser.

    Shows inference metrics across all devices — latency,
    throughput, errors, model versions side-by-side.
    """
    dashboard_url = os.environ.get("EDGEML_DASHBOARD_URL", "https://app.edgeml.io")
    click.echo(f"Opening dashboard: {dashboard_url}")
    webbrowser.open(dashboard_url)


# ---------------------------------------------------------------------------
# edgeml benchmark
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model")
@click.option(
    "--share", is_flag=True, help="Upload anonymous benchmark results to EdgeML."
)
@click.option("--iterations", "-n", default=10, help="Number of inference iterations.")
def benchmark(model: str, share: bool, iterations: int) -> None:
    """Run inference benchmarks on a model.

    Measures time-to-first-chunk, throughput, and memory usage
    across multiple iterations.

    Example:

        edgeml benchmark gemma-1b --share --iterations 20
    """
    import platform as _platform
    import time

    click.echo(f"Benchmarking {model} ({iterations} iterations)...")
    click.echo(f"Platform: {_platform.system()} {_platform.machine()}")

    from .serve import _detect_backend

    backend = _detect_backend(model)
    click.echo(f"Backend: {backend.name}")

    from .serve import GenerationRequest

    req = GenerationRequest(
        model=model,
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=50,
    )

    latencies: list[float] = []
    tps_list: list[float] = []

    for i in range(iterations):
        start = time.monotonic()
        text, metrics = backend.generate(req)
        elapsed = (time.monotonic() - start) * 1000
        latencies.append(elapsed)
        tps_list.append(metrics.tokens_per_second)
        click.echo(
            f"  [{i+1}/{iterations}] {elapsed:.1f}ms, {metrics.tokens_per_second:.1f} tok/s"
        )

    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    avg_tps = sum(tps_list) / len(tps_list)

    click.echo("\nResults:")
    click.echo(f"  Avg latency: {avg_latency:.1f}ms")
    click.echo(f"  P50 latency: {p50:.1f}ms")
    click.echo(f"  P95 latency: {p95:.1f}ms")
    click.echo(f"  Avg throughput: {avg_tps:.1f} tok/s")
    click.echo(f"  Backend: {backend.name}")

    if share:
        api_key = _get_api_key()
        if not api_key:
            click.echo(
                "\nSkipping share: no API key. Run `edgeml login` first.",
                err=True,
            )
            return

        click.echo("\nSharing anonymous benchmark data...")
        try:
            import httpx

            payload = {
                "model": model,
                "backend": backend.name,
                "platform": _platform.system(),
                "arch": _platform.machine(),
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "avg_tokens_per_second": avg_tps,
                "iterations": iterations,
            }
            api_base = os.environ.get("EDGEML_API_BASE", "https://api.edgeml.io/api/v1")
            resp = httpx.post(
                f"{api_base}/benchmarks",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            if resp.status_code < 400:
                click.echo("Benchmark data shared successfully.")
            else:
                click.echo(f"Failed to share: {resp.status_code}", err=True)
        except Exception as exc:
            click.echo(f"Failed to share: {exc}", err=True)


if __name__ == "__main__":
    main()
