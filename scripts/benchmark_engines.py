#!/usr/bin/env python3
"""Cross-engine inference benchmark — Ollama vs MLX vs llama.cpp vs Octomil.

Directly invokes each engine with HuggingFace/Ollama model references,
bypassing the catalog system. Tests multiple models, prompt types, and
measures tok/s, TTFT, and latency.

"Octomil" mode = smart router auto-selecting the fastest engine per request.

Usage:
    uv run python scripts/benchmark_engines.py
    uv run python scripts/benchmark_engines.py --models gemma-2b llama-3b
    uv run python scripts/benchmark_engines.py --iterations 20
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ── Model definitions ──────────────────────────────────────────────────
# Maps short names to engine-specific references.

MODELS: dict[str, dict[str, str]] = {
    "gemma-2b": {
        "ollama": "gemma2:2b-instruct-q4_K_M",
        "mlx": "mlx-community/gemma-2-2b-it-4bit",
        "gguf": "bartowski/gemma-2-2b-it-GGUF:gemma-2-2b-it-Q4_K_M.gguf",
        "params": "2B",
    },
    "llama-3b": {
        "ollama": "llama3.2:3b-instruct-q4_K_M",
        "mlx": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "gguf": "bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "params": "3B",
    },
    "qwen-3b": {
        "ollama": "qwen2.5:3b-instruct-q4_K_M",
        "mlx": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "gguf": "bartowski/Qwen2.5-3B-Instruct-GGUF:Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "params": "3B",
    },
    "llama-8b": {
        "ollama": "llama3.1:8b-instruct-q4_K_M",
        "mlx": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "gguf": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "params": "8B",
    },
    "mistral-7b": {
        "ollama": "mistral:7b-instruct-q4_K_M",
        "mlx": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "gguf": "bartowski/Mistral-7B-Instruct-v0.3-GGUF:Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "params": "7B",
    },
}

# ── Prompt types ───────────────────────────────────────────────────────

PROMPTS: dict[str, list[dict[str, str]]] = {
    "general": [
        {"role": "user", "content": "Explain how a hash table works in 3 sentences."},
    ],
    "coding": [
        {
            "role": "user",
            "content": (
                "Write a Python function that finds the longest common subsequence "
                "of two strings using dynamic programming. Include type hints and "
                "a docstring."
            ),
        },
    ],
    "multimodal_reasoning": [
        {
            "role": "user",
            "content": (
                "I have an image of a bar chart showing quarterly revenue for 2024. "
                "Q1=$2.1M, Q2=$2.8M, Q3=$3.4M, Q4=$4.1M. "
                "Analyze the trend, calculate QoQ growth rates, and predict Q1 2025."
            ),
        },
    ],
}


@dataclass
class BenchResult:
    engine: str
    model: str
    prompt_type: str
    tokens_per_second: float = 0.0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens_generated: int = 0
    error: Optional[str] = None
    iterations: list[dict[str, float]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (pct / 100.0) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ── Engine runners ─────────────────────────────────────────────────────


def bench_ollama(
    model_tag: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    n_iters: int,
    warmup: int = 1,
) -> BenchResult:
    """Benchmark via Ollama HTTP API."""
    import httpx

    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    # Warmup
    for _ in range(warmup):
        try:
            httpx.post(
                f"{base_url}/api/chat",
                json={
                    "model": model_tag,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": 8},
                },
                timeout=120.0,
            )
        except Exception:
            pass

    iters: list[dict[str, float]] = []
    for _ in range(n_iters):
        start = time.monotonic()
        try:
            resp = httpx.post(
                f"{base_url}/api/chat",
                json={
                    "model": model_tag,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=120.0,
            )
            resp.raise_for_status()
        except Exception as exc:
            return BenchResult(
                engine="ollama",
                model=model_tag,
                prompt_type="",
                error=str(exc),
            )
        elapsed_ms = (time.monotonic() - start) * 1000
        data = resp.json()
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        prompt_eval_ns = data.get("prompt_eval_duration", 0)

        tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
        ttft = prompt_eval_ns / 1e6 if prompt_eval_ns else 0.0

        iters.append(
            {
                "tps": tps,
                "ttft_ms": ttft,
                "total_ms": elapsed_ms,
                "tokens": eval_count,
            }
        )

    avg_tps = sum(i["tps"] for i in iters) / len(iters)
    avg_ttft = sum(i["ttft_ms"] for i in iters) / len(iters)
    avg_total = sum(i["total_ms"] for i in iters) / len(iters)
    avg_tokens = int(sum(i["tokens"] for i in iters) / len(iters))

    return BenchResult(
        engine="ollama",
        model=model_tag,
        prompt_type="",
        tokens_per_second=avg_tps,
        ttft_ms=avg_ttft,
        total_ms=avg_total,
        tokens_generated=avg_tokens,
        iterations=iters,
    )


def bench_mlx(
    repo_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    n_iters: int,
    warmup: int = 2,
) -> BenchResult:
    """Benchmark via mlx-lm directly."""
    try:
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        return BenchResult(engine="mlx-lm", model=repo_id, prompt_type="", error="mlx-lm not installed")

    try:
        model, tokenizer = mlx_lm.load(repo_id)
    except Exception as exc:
        return BenchResult(engine="mlx-lm", model=repo_id, prompt_type="", error=f"load failed: {exc}")

    # Format prompt
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    sampler = make_sampler(temp=0.7)

    # Warmup — JIT-compile Metal shaders
    for _ in range(warmup):
        for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, max_tokens=8, sampler=sampler):
            if resp.finish_reason:
                break

    iters: list[dict[str, float]] = []
    for _ in range(n_iters):
        start = time.monotonic()
        tokens = 0
        first_token_time = None
        gen_tps = 0.0

        for resp in mlx_lm.stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            prefill_step_size=4096,
        ):
            if first_token_time is None:
                first_token_time = time.monotonic()
            tokens += 1
            gen_tps = resp.generation_tps
            if resp.finish_reason:
                break

        elapsed_ms = (time.monotonic() - start) * 1000
        ttft = ((first_token_time or start) - start) * 1000
        tps = gen_tps if gen_tps > 0 else (tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0)

        iters.append({"tps": tps, "ttft_ms": ttft, "total_ms": elapsed_ms, "tokens": tokens})

    # Cleanup — aggressively free Metal memory
    del model, tokenizer, sampler
    gc.collect()
    try:
        import mlx.core as mx

        mx.metal.clear_cache()
    except Exception:
        pass

    avg_tps = sum(i["tps"] for i in iters) / len(iters)
    avg_ttft = sum(i["ttft_ms"] for i in iters) / len(iters)
    avg_total = sum(i["total_ms"] for i in iters) / len(iters)
    avg_tokens = int(sum(i["tokens"] for i in iters) / len(iters))

    return BenchResult(
        engine="mlx-lm",
        model=repo_id,
        prompt_type="",
        tokens_per_second=avg_tps,
        ttft_ms=avg_ttft,
        total_ms=avg_total,
        tokens_generated=avg_tokens,
        iterations=iters,
    )


def bench_llamacpp(
    gguf_ref: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    n_iters: int,
    warmup: int = 1,
) -> BenchResult:
    """Benchmark via llama-cpp-python directly.

    gguf_ref is either a local .gguf path or 'repo:filename' for HuggingFace.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        return BenchResult(
            engine="llama.cpp",
            model=gguf_ref,
            prompt_type="",
            error="llama-cpp-python not installed",
        )

    # Resolve HuggingFace GGUF
    try:
        if ":" in gguf_ref and "/" in gguf_ref.split(":")[0]:
            repo, filename = gguf_ref.split(":", 1)
            llm = Llama.from_pretrained(
                repo_id=repo,
                filename=filename,
                n_ctx=2048,
                n_gpu_layers=-1,  # offload all layers to Metal
                verbose=False,
            )
        else:
            llm = Llama(model_path=gguf_ref, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    except Exception as exc:
        return BenchResult(
            engine="llama.cpp",
            model=gguf_ref,
            prompt_type="",
            error=f"load failed: {exc}",
        )

    # Warmup
    for _ in range(warmup):
        try:
            llm.create_chat_completion(
                messages=messages,  # type: ignore[arg-type]
                max_tokens=8,
                temperature=0.7,
            )
        except Exception:
            pass

    iters: list[dict[str, float]] = []
    for _ in range(n_iters):
        start = time.monotonic()
        try:
            result = llm.create_chat_completion(
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=0.7,
            )
        except Exception as exc:
            return BenchResult(
                engine="llama.cpp",
                model=gguf_ref,
                prompt_type="",
                error=str(exc),
            )
        elapsed_ms = (time.monotonic() - start) * 1000

        # Extract timings from llama.cpp
        usage = result.get("usage", {})  # type: ignore[union-attr]
        completion_tokens = usage.get("completion_tokens", 0)

        prompt_ms = 0.0
        tps = completion_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0

        # Try to get decode-only tok/s from llama.cpp internals
        try:
            t = llm._ctx.timings
            if hasattr(t, "t_eval_ms") and hasattr(t, "n_eval") and t.n_eval > 0:
                tps = t.n_eval / (t.t_eval_ms / 1000)
            if hasattr(t, "t_p_eval_ms"):
                prompt_ms = t.t_p_eval_ms
        except Exception:
            pass

        iters.append(
            {
                "tps": tps,
                "ttft_ms": prompt_ms,
                "total_ms": elapsed_ms,
                "tokens": completion_tokens,
            }
        )

    del llm
    gc.collect()

    avg_tps = sum(i["tps"] for i in iters) / len(iters)
    avg_ttft = sum(i["ttft_ms"] for i in iters) / len(iters)
    avg_total = sum(i["total_ms"] for i in iters) / len(iters)
    avg_tokens = int(sum(i["tokens"] for i in iters) / len(iters))

    return BenchResult(
        engine="llama.cpp",
        model=gguf_ref,
        prompt_type="",
        tokens_per_second=avg_tps,
        ttft_ms=avg_ttft,
        total_ms=avg_total,
        tokens_generated=avg_tokens,
        iterations=iters,
    )


def bench_octomil(
    mlx_repo: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    n_iters: int,
    warmup: int = 1,
) -> BenchResult:
    """Benchmark through Octomil's real auto-select pipeline.

    Uses _detect_backend() which auto-benchmarks all available engines,
    picks the fastest, creates a backend, then runs inference through it.
    This measures the *actual* Octomil user experience including the
    engine selection overhead.
    """
    try:
        from octomil.serve import GenerationRequest, _detect_backend
    except ImportError:
        return BenchResult(
            engine="octomil",
            model=mlx_repo,
            prompt_type="",
            error="octomil.serve not available",
        )

    # Use MLX repo (has '/') so it bypasses catalog and goes straight to engine
    try:
        print("    [octomil] Auto-detecting and benchmarking engines...", flush=True)
        auto_start = time.monotonic()
        backend = _detect_backend(mlx_repo, cache_enabled=False)
        auto_ms = (time.monotonic() - auto_start) * 1000
        engine_name = getattr(backend, "name", "unknown")
        print(f"    [octomil] Selected: {engine_name} ({auto_ms:.0f}ms auto-select)", flush=True)
    except Exception as exc:
        return BenchResult(
            engine="octomil",
            model=mlx_repo,
            prompt_type="",
            error=f"auto-select failed: {exc}",
        )

    req = GenerationRequest(
        model=mlx_repo,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )

    # Warmup
    for _ in range(warmup):
        try:
            backend.generate(req)
        except Exception:
            pass

    iters: list[dict[str, float]] = []
    for _ in range(n_iters):
        start = time.monotonic()
        try:
            _text, metrics = backend.generate(req)
        except Exception as exc:
            return BenchResult(
                engine="octomil",
                model=mlx_repo,
                prompt_type="",
                error=str(exc),
            )
        elapsed_ms = (time.monotonic() - start) * 1000

        tps = metrics.tokens_per_second
        ttft = metrics.ttfc_ms
        tokens = metrics.total_tokens

        iters.append({"tps": tps, "ttft_ms": ttft, "total_ms": elapsed_ms, "tokens": tokens})

    # Cleanup — aggressively free Metal memory
    del backend
    gc.collect()
    try:
        import mlx.core as mx

        mx.metal.clear_cache()
    except Exception:
        pass

    avg_tps = sum(i["tps"] for i in iters) / len(iters)
    avg_ttft = sum(i["ttft_ms"] for i in iters) / len(iters)
    avg_total = sum(i["total_ms"] for i in iters) / len(iters)
    avg_tokens = int(sum(i["tokens"] for i in iters) / len(iters))

    return BenchResult(
        engine=f"octomil ({engine_name})",
        model=mlx_repo,
        prompt_type="",
        tokens_per_second=avg_tps,
        ttft_ms=avg_ttft,
        total_ms=avg_total,
        tokens_generated=avg_tokens,
        iterations=iters,
    )


# ── Output formatting ──────────────────────────────────────────────────


def print_model_results(model_name: str, results: list[BenchResult], params: str) -> None:
    """Print comparison table for one model across engines."""
    print(f"\n{'='*80}")
    print(f"  {model_name} ({params})")
    print(f"{'='*80}")
    print(
        f"  {'Engine':<20s} {'Avg tok/s':>10s} {'TTFT':>10s} "
        f"{'Total':>10s} {'p50 tok/s':>10s} {'p99 tok/s':>10s} {'Status':>8s}"
    )
    print(f"  {'-'*70}")

    # Sort by tok/s descending
    valid = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    valid.sort(key=lambda r: r.tokens_per_second, reverse=True)

    best_tps = valid[0].tokens_per_second if valid else 0

    for i, r in enumerate(valid):
        tps_values = [it["tps"] for it in r.iterations]
        p50 = percentile(tps_values, 50)
        p99 = percentile(tps_values, 99)
        marker = " *" if i == 0 else ""
        speedup = ""
        if i > 0 and r.tokens_per_second > 0:
            ratio = best_tps / r.tokens_per_second
            speedup = f" ({ratio:.1f}x slower)"
        print(
            f"  {r.engine:<20s} {r.tokens_per_second:>10.1f} "
            f"{r.ttft_ms:>8.1f}ms {r.total_ms:>8.1f}ms "
            f"{p50:>10.1f} {p99:>10.1f} {'OK':>8s}{marker}{speedup}"
        )

    for r in failed:
        print(
            f"  {r.engine:<20s} {'---':>10s} {'---':>10s} "
            f"{'---':>10s} {'---':>10s} {'---':>10s} {'FAIL':>8s}  {r.error}"
        )


def print_summary(all_results: dict[str, dict[str, list[BenchResult]]]) -> None:
    """Print final summary table — which engine wins per model & prompt type."""
    print(f"\n{'='*80}")
    print("  SUMMARY — Fastest engine per model and prompt type")
    print(f"{'='*80}")
    print(f"  {'Model':<15s} {'Prompt Type':<20s} {'Winner':<15s} {'tok/s':>8s} {'2nd place':>12s}")
    print(f"  {'-'*70}")

    octomil_wins = 0
    total = 0

    for model_name, prompt_results in all_results.items():
        for prompt_type, results in prompt_results.items():
            valid = [r for r in results if r.ok]
            if not valid:
                continue
            valid.sort(key=lambda r: r.tokens_per_second, reverse=True)
            winner = valid[0]
            second = valid[1] if len(valid) > 1 else None
            second_str = f"{second.engine} ({second.tokens_per_second:.1f})" if second else "---"

            print(
                f"  {model_name:<15s} {prompt_type:<20s} "
                f"{winner.engine:<25s} {winner.tokens_per_second:>8.1f} {second_str:>12s}"
            )

            if "octomil" in winner.engine.lower():
                octomil_wins += 1
            total += 1

    print(f"\n  Octomil wins: {octomil_wins}/{total} tests")


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-engine inference benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to benchmark",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=list(PROMPTS.keys()),
        choices=list(PROMPTS.keys()),
        help="Prompt types to test",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["ollama", "mlx", "llamacpp", "octomil"],
        help="Engines to benchmark (octomil = real auto-select pipeline)",
    )
    parser.add_argument("--iterations", "-n", type=int, default=5, help="Iterations per test")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per generation")
    parser.add_argument("--output-json", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    import platform

    try:
        import psutil

        ram = psutil.virtual_memory().total / (1024**3)
        ram_str = f"{ram:.0f}GB"
    except ImportError:
        ram_str = "unknown"

    print(f"Benchmark: {platform.system()} {platform.machine()}")
    print(f"Chip: {platform.processor() or platform.machine()}")
    print(f"RAM: {ram_str}")
    print(f"Iterations: {args.iterations}, Max tokens: {args.max_tokens}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Engines: {', '.join(args.engines)}")
    print(f"Prompts: {', '.join(args.prompts)}")

    all_results: dict[str, dict[str, list[BenchResult]]] = {}
    all_json: list[dict[str, Any]] = []

    for model_idx, model_name in enumerate(args.models):
        model_spec = MODELS[model_name]
        all_results[model_name] = {}

        # Force cleanup between models to prevent Metal OOM on 7B+ models
        if model_idx > 0:
            gc.collect()
            try:
                import mlx.core as mx

                mx.metal.clear_cache()
            except Exception:
                pass
            print(f"\n  [cleanup] Memory cleared before {model_name}", flush=True)

        for prompt_type in args.prompts:
            messages = PROMPTS[prompt_type]
            results: list[BenchResult] = []
            all_results[model_name][prompt_type] = results

            print(f"\n--- {model_name} / {prompt_type} ---")

            for engine in args.engines:
                if engine == "ollama" and "ollama" in model_spec:
                    print(f"  Benchmarking Ollama ({model_spec['ollama']})...", flush=True)
                    r = bench_ollama(model_spec["ollama"], messages, args.max_tokens, args.iterations)
                elif engine == "mlx" and "mlx" in model_spec:
                    print(f"  Benchmarking mlx-lm ({model_spec['mlx']})...", flush=True)
                    r = bench_mlx(model_spec["mlx"], messages, args.max_tokens, args.iterations)
                elif engine == "llamacpp" and "gguf" in model_spec:
                    print(f"  Benchmarking llama.cpp ({model_spec['gguf']})...", flush=True)
                    r = bench_llamacpp(model_spec["gguf"], messages, args.max_tokens, args.iterations)
                elif engine == "octomil" and "mlx" in model_spec:
                    print(f"  Benchmarking Octomil (auto-select) ({model_spec['mlx']})...", flush=True)
                    r = bench_octomil(model_spec["mlx"], messages, args.max_tokens, args.iterations)
                else:
                    continue

                r.prompt_type = prompt_type
                r.model = model_name
                results.append(r)

                if r.ok:
                    print(f"    -> {r.tokens_per_second:.1f} tok/s, TTFT {r.ttft_ms:.1f}ms")
                else:
                    print(f"    -> FAILED: {r.error}")

            print_model_results(model_name, results, model_spec.get("params", ""))

            # Collect JSON output
            for r in results:
                all_json.append(
                    {
                        "model": model_name,
                        "engine": r.engine,
                        "prompt_type": prompt_type,
                        "avg_tps": round(r.tokens_per_second, 1),
                        "ttft_ms": round(r.ttft_ms, 1),
                        "total_ms": round(r.total_ms, 1),
                        "tokens": r.tokens_generated,
                        "error": r.error,
                        "p50_tps": round(percentile([i["tps"] for i in r.iterations], 50), 1) if r.iterations else 0,
                        "p99_tps": round(percentile([i["tps"] for i in r.iterations], 99), 1) if r.iterations else 0,
                    }
                )

    print_summary(all_results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "platform": platform.system(),
                    "arch": platform.machine(),
                    "chip": platform.processor() or platform.machine(),
                    "ram": ram_str,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "config": {
                        "iterations": args.iterations,
                        "max_tokens": args.max_tokens,
                    },
                    "results": all_json,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
