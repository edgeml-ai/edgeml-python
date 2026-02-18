"""
OpenAI-compatible local inference server.

Dispatches to the best available backend:
  1. mlx-lm (Apple Silicon)
  2. llama-cpp-python (cross-platform)
  3. ONNX Runtime (fallback)

Usage::

    from edgeml.serve import create_app, run_server
    run_server("gemma-1b", port=8080)
"""

from __future__ import annotations

import json
import logging
import platform
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


@dataclass
class GenerationRequest:
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False


@dataclass
class GenerationChunk:
    text: str
    finish_reason: Optional[str] = None


@dataclass
class InferenceMetrics:
    """Metrics for a single inference request."""

    ttfc_ms: float = 0.0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    total_duration_ms: float = 0.0


class InferenceBackend:
    """Base class for inference backends."""

    name: str = "base"

    def load_model(self, model_name: str) -> None:
        raise NotImplementedError

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        raise NotImplementedError

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        raise NotImplementedError
        yield  # pragma: no cover — makes this an async generator

    def list_models(self) -> list[str]:
        raise NotImplementedError


class MLXBackend(InferenceBackend):
    """Apple Silicon backend using mlx-lm."""

    name = "mlx-lm"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_name: str = ""

    def load_model(self, model_name: str) -> None:
        import mlx_lm  # type: ignore[import-untyped]

        self._model_name = model_name
        logger.info("Loading model %s with mlx-lm...", model_name)
        self._model, self._tokenizer = mlx_lm.load(model_name)
        logger.info("Model %s loaded.", model_name)

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
            else:
                parts.append(f"User: {content}\n")
        parts.append("Assistant: ")
        return "".join(parts)

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        import mlx_lm  # type: ignore[import-untyped]

        prompt = self._messages_to_prompt(request.messages)
        start = time.monotonic()
        tokens: list[str] = []
        first_token_time: Optional[float] = None

        for token in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
        ):
            if first_token_time is None:
                first_token_time = time.monotonic()
            tokens.append(token)

        elapsed = time.monotonic() - start
        ttfc = ((first_token_time or start) - start) * 1000
        text = "".join(tokens)
        tps = len(tokens) / elapsed if elapsed > 0 else 0
        metrics = InferenceMetrics(
            ttfc_ms=ttfc,
            total_tokens=len(tokens),
            tokens_per_second=tps,
            total_duration_ms=elapsed * 1000,
        )
        return text, metrics

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        import asyncio

        import mlx_lm  # type: ignore[import-untyped]

        prompt = self._messages_to_prompt(request.messages)
        tokens = list(
            mlx_lm.stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=request.max_tokens,
            )
        )
        for i, token in enumerate(tokens):
            is_last = i == len(tokens) - 1
            yield GenerationChunk(
                text=token,
                finish_reason="stop" if is_last else None,
            )
            await asyncio.sleep(0)

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []


class LlamaCppBackend(InferenceBackend):
    """Cross-platform backend using llama-cpp-python."""

    name = "llama.cpp"

    def __init__(self) -> None:
        self._llm: Any = None
        self._model_name: str = ""

    def load_model(self, model_name: str) -> None:
        from llama_cpp import Llama  # type: ignore[import-untyped]

        self._model_name = model_name
        logger.info("Loading model %s with llama.cpp...", model_name)
        self._llm = Llama(model_path=model_name, n_ctx=2048, verbose=False)
        logger.info("Model %s loaded.", model_name)

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        start = time.monotonic()
        result = self._llm.create_chat_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        elapsed = time.monotonic() - start
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        total_tokens = usage.get("completion_tokens", 0)
        metrics = InferenceMetrics(
            total_tokens=total_tokens,
            tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0,
            total_duration_ms=elapsed * 1000,
        )
        return text, metrics

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        import asyncio

        for chunk in self._llm.create_chat_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            finish = chunk["choices"][0].get("finish_reason")
            if content or finish:
                yield GenerationChunk(text=content, finish_reason=finish)
                await asyncio.sleep(0)

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []


class EchoBackend(InferenceBackend):
    """Fallback backend that echoes input — useful for testing the API layer."""

    name = "echo"

    def __init__(self) -> None:
        self._model_name: str = ""

    def load_model(self, model_name: str) -> None:
        self._model_name = model_name
        logger.info("Echo backend loaded for model %s (no real inference).", model_name)

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        last_msg = request.messages[-1]["content"] if request.messages else ""
        text = f"[echo:{self._model_name}] {last_msg}"
        metrics = InferenceMetrics(total_tokens=len(text.split()))
        return text, metrics

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        import asyncio

        last_msg = request.messages[-1]["content"] if request.messages else ""
        words = f"[echo:{self._model_name}] {last_msg}".split()
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            yield GenerationChunk(
                text=word + ("" if is_last else " "),
                finish_reason="stop" if is_last else None,
            )
            await asyncio.sleep(0.02)

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []


def _detect_backend(model_name: str) -> InferenceBackend:
    """Pick the best available backend for the current platform."""

    # Apple Silicon → mlx-lm
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_lm  # type: ignore[import-untyped] # noqa: F401

            mlx_backend: InferenceBackend = MLXBackend()
            mlx_backend.load_model(model_name)
            return mlx_backend
        except (ImportError, Exception) as exc:
            logger.debug("mlx-lm not available: %s", exc)

    # llama.cpp — cross-platform, GGUF models
    try:
        import llama_cpp  # type: ignore[import-untyped] # noqa: F401

        cpp_backend: InferenceBackend = LlamaCppBackend()
        cpp_backend.load_model(model_name)
        return cpp_backend
    except (ImportError, Exception) as exc:
        logger.debug("llama-cpp-python not available: %s", exc)

    # Fallback to echo
    logger.warning(
        "No inference backend found (install mlx-lm or llama-cpp-python). "
        "Using echo backend."
    )
    echo_backend: InferenceBackend = EchoBackend()
    echo_backend.load_model(model_name)
    return echo_backend


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


@dataclass
class ServerState:
    """Shared mutable state for the serve app."""

    backend: Optional[InferenceBackend] = None
    model_name: str = ""
    start_time: float = field(default_factory=time.time)
    request_count: int = 0
    api_key: Optional[str] = None
    api_base: str = "https://api.edgeml.io/api/v1"


def create_app(
    model_name: str,
    *,
    api_key: Optional[str] = None,
    api_base: str = "https://api.edgeml.io/api/v1",
) -> Any:
    """Create a FastAPI app with OpenAI-compatible endpoints."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="EdgeML Serve", version="1.0.0")
    state = ServerState(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _load_model() -> None:
        state.backend = _detect_backend(model_name)
        state.start_time = time.time()

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        models = state.backend.list_models() if state.backend else []
        return {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "created": int(state.start_time),
                    "owned_by": "edgeml",
                }
                for m in models
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        if state.backend is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        body = await request.json()
        gen_req = GenerationRequest(
            model=body.get("model", state.model_name),
            messages=body.get("messages", []),
            max_tokens=body.get("max_tokens", 512),
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 1.0),
            stream=body.get("stream", False),
        )

        state.request_count += 1
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if gen_req.stream:
            return StreamingResponse(
                _stream_response(state, gen_req, req_id),
                media_type="text/event-stream",
            )

        text, metrics = state.backend.generate(gen_req)
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": gen_req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": metrics.total_tokens,
                "total_tokens": metrics.total_tokens,
            },
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": state.model_name,
            "backend": state.backend.name if state.backend else "none",
            "requests_served": state.request_count,
            "uptime_seconds": int(time.time() - state.start_time),
        }

    return app


async def _stream_response(
    state: ServerState,
    request: GenerationRequest,
    req_id: str,
) -> AsyncIterator[str]:
    """Yield SSE chunks in OpenAI streaming format."""
    assert state.backend is not None

    async for chunk in state.backend.generate_stream(request):
        data = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.text} if chunk.text else {},
                    "finish_reason": chunk.finish_reason,
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"


def run_server(
    model_name: str,
    *,
    port: int = 8080,
    host: str = "0.0.0.0",
    api_key: Optional[str] = None,
    api_base: str = "https://api.edgeml.io/api/v1",
) -> None:
    """Start the inference server (blocking)."""
    import uvicorn

    app = create_app(model_name, api_key=api_key, api_base=api_base)
    uvicorn.run(app, host=host, port=port, log_level="info")
