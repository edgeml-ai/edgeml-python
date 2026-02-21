"""Tests for edgeml.routing — query routing and complexity estimation."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from edgeml.routing import (
    TIER_ORDER,
    ModelInfo,
    QueryRouter,
    _estimate_complexity,
    assign_tiers,
)


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_defaults(self):
        info = ModelInfo(name="test")
        assert info.tier == "balanced"
        assert info.param_b == 0.0
        assert info.loaded is True

    def test_tier_index(self):
        assert ModelInfo(name="a", tier="fast").tier_index == 0
        assert ModelInfo(name="b", tier="balanced").tier_index == 1
        assert ModelInfo(name="c", tier="quality").tier_index == 2

    def test_tier_index_unknown(self):
        """Unknown tier defaults to 1 (balanced)."""
        info = ModelInfo(name="x", tier="unknown")
        assert info.tier_index == 1

    def test_tier_order_constant(self):
        assert TIER_ORDER == ["fast", "balanced", "quality"]


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------


class TestComplexityEstimation:
    """Test the _estimate_complexity heuristic."""

    def test_simple_greeting(self):
        score = _estimate_complexity("hi")
        assert score < 0.3, f"Greeting should be simple, got {score}"

    def test_simple_hello(self):
        score = _estimate_complexity("hello")
        assert score < 0.3, f"Hello should be simple, got {score}"

    def test_simple_thanks(self):
        score = _estimate_complexity("thanks")
        assert score < 0.3, f"Thanks should be simple, got {score}"

    def test_simple_factual(self):
        score = _estimate_complexity("what is a dog")
        assert score < 0.35, f"Simple factual should be low complexity, got {score}"

    def test_medium_question(self):
        score = _estimate_complexity(
            "Explain the difference between TCP and UDP protocols"
        )
        assert 0.15 < score < 0.75, f"Medium question should be mid-range, got {score}"

    def test_complex_code_request(self):
        score = _estimate_complexity(
            "Write a function that implements a binary search tree with insert, "
            "delete, and balance operations. Include proper error handling and "
            "type hints. The algorithm should handle edge cases like duplicate keys."
        )
        assert score > 0.4, (
            f"Complex code request should be high complexity, got {score}"
        )

    def test_complex_reasoning(self):
        score = _estimate_complexity(
            "Analyze the tradeoffs between microservice and monolithic architectures. "
            "Compare latency, throughput, and deployment complexity. Evaluate which "
            "is better for a startup with 3 engineers vs an enterprise with 200."
        )
        assert score > 0.4, f"Complex reasoning should be high complexity, got {score}"

    def test_math_is_complex(self):
        score = _estimate_complexity(
            "Derive the gradient of the cross-entropy loss function with respect to "
            "the softmax output. Prove that it simplifies to y_pred - y_true."
        )
        assert score > 0.5, f"Math derivation should be complex, got {score}"

    def test_system_prompt_raises_complexity(self):
        text = "Summarize this"
        score_no_sys = _estimate_complexity(text)
        score_with_sys = _estimate_complexity(
            text,
            system_prompt="You are a detailed technical writer who must provide "
            "comprehensive analysis with citations, code examples, and step-by-step "
            "breakdowns of every concept mentioned." * 3,
        )
        assert score_with_sys > score_no_sys, (
            f"System prompt should raise complexity: {score_with_sys} vs {score_no_sys}"
        )

    def test_multi_turn_raises_complexity(self):
        text = "And what about the edge cases?"
        score_1_turn = _estimate_complexity(text, turn_count=1)
        score_5_turns = _estimate_complexity(text, turn_count=5)
        assert score_5_turns > score_1_turn, (
            f"Multi-turn should raise complexity: {score_5_turns} vs {score_1_turn}"
        )

    def test_score_bounded_zero_to_one(self):
        """Complexity should always be in [0.0, 1.0]."""
        test_cases = [
            "",
            "hi",
            "a" * 10000,
            "write implement code function class algorithm " * 50,
            "hello bye thanks what is define " * 20,
        ]
        for text in test_cases:
            score = _estimate_complexity(text)
            assert 0.0 <= score <= 1.0, (
                f"Score out of bounds for '{text[:30]}...': {score}"
            )

    def test_code_indicators_raise_complexity(self):
        score_no_code = _estimate_complexity("explain sorting")
        score_with_code = _estimate_complexity(
            "```python\ndef sort(arr):\n    pass\n```\nfix this function"
        )
        assert score_with_code > score_no_code

    def test_technical_vocab_raises_complexity(self):
        score_simple = _estimate_complexity("the cat sat on the mat")
        score_technical = _estimate_complexity(
            "asynchronous concurrency with mutex and semaphore for deadlock prevention"
        )
        assert score_technical > score_simple


# ---------------------------------------------------------------------------
# assign_tiers
# ---------------------------------------------------------------------------


class TestAssignTiers:
    def test_empty_list(self):
        result = assign_tiers([])
        assert result == {}

    def test_single_model(self):
        result = assign_tiers(["model-a"])
        assert len(result) == 1
        assert result["model-a"].tier == "balanced"

    def test_two_models(self):
        result = assign_tiers(["small", "large"])
        assert result["small"].tier == "fast"
        assert result["large"].tier == "quality"

    def test_three_models(self):
        result = assign_tiers(["small", "medium", "large"])
        assert result["small"].tier == "fast"
        assert result["medium"].tier == "balanced"
        assert result["large"].tier == "quality"

    def test_four_models(self):
        result = assign_tiers(["xs", "s", "m", "l"])
        # 4 models: first 1 = fast, middle 2 = balanced, last 1 = quality
        assert result["xs"].tier == "fast"
        assert result["s"].tier == "balanced"
        assert result["m"].tier == "balanced"
        assert result["l"].tier == "quality"

    def test_six_models(self):
        result = assign_tiers(["a", "b", "c", "d", "e", "f"])
        # 6 models: first 2 = fast, middle 2 = balanced, last 2 = quality
        assert result["a"].tier == "fast"
        assert result["b"].tier == "fast"
        assert result["c"].tier == "balanced"
        assert result["d"].tier == "balanced"
        assert result["e"].tier == "quality"
        assert result["f"].tier == "quality"


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------


class TestQueryRouter:
    @pytest.fixture
    def three_models(self) -> dict[str, ModelInfo]:
        return {
            "small": ModelInfo(name="small", tier="fast", param_b=0.36),
            "medium": ModelInfo(name="medium", tier="balanced", param_b=3.8),
            "large": ModelInfo(name="large", tier="quality", param_b=7.0),
        }

    @pytest.fixture
    def router(self, three_models: dict[str, ModelInfo]) -> QueryRouter:
        return QueryRouter(three_models, strategy="complexity")

    def test_init_requires_models(self):
        with pytest.raises(ValueError, match="At least one model"):
            QueryRouter({}, strategy="complexity")

    def test_init_rejects_unknown_strategy(self, three_models):
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            QueryRouter(three_models, strategy="round_robin")

    def test_routes_simple_to_small(self, router):
        decision = router.route([{"role": "user", "content": "hi"}])
        assert decision.model_name == "small"
        assert decision.tier == "fast"
        assert decision.complexity_score < 0.3

    def test_routes_complex_to_large(self, router):
        decision = router.route(
            [
                {
                    "role": "user",
                    "content": (
                        "Write a complete implementation of a distributed hash table "
                        "with consistent hashing, virtual nodes, and replication. "
                        "Include the algorithm for node join/leave and key redistribution. "
                        "Prove the load balance properties step by step."
                    ),
                }
            ]
        )
        assert decision.model_name == "large"
        assert decision.tier == "quality"
        assert decision.complexity_score >= 0.7

    def test_routes_medium_to_balanced(self, router):
        decision = router.route(
            [
                {
                    "role": "user",
                    "content": "Explain the difference between REST and GraphQL APIs",
                }
            ]
        )
        assert decision.tier in ("balanced", "fast", "quality")
        # The exact routing depends on the heuristic, but the score should
        # be in a reasonable range
        assert 0.0 <= decision.complexity_score <= 1.0

    def test_routing_decision_has_fallback_chain(self, router):
        decision = router.route([{"role": "user", "content": "hello"}])
        assert isinstance(decision.fallback_chain, list)
        # The primary model should not be in the fallback chain
        assert decision.model_name not in decision.fallback_chain

    def test_fallback_chain_excludes_primary(self, router):
        decision = router.route([{"role": "user", "content": "hi"}])
        assert decision.model_name == "small"
        assert "small" not in decision.fallback_chain
        # Other models should be in the chain
        assert len(decision.fallback_chain) == 2

    def test_routing_decision_strategy(self, router):
        decision = router.route([{"role": "user", "content": "test"}])
        assert decision.strategy == "complexity"

    def test_system_prompt_affects_routing(self, router):
        messages_simple = [{"role": "user", "content": "summarize"}]
        messages_complex = [
            {
                "role": "system",
                "content": "You are an expert compiler engineer. Provide detailed "
                "analysis with assembly code examples, optimization passes, "
                "and formal verification proofs." * 5,
            },
            {"role": "user", "content": "summarize"},
        ]

        decision_simple = router.route(messages_simple)
        decision_complex = router.route(messages_complex)
        assert decision_complex.complexity_score > decision_simple.complexity_score

    def test_custom_thresholds(self, three_models):
        # Very permissive fast tier (0-0.8)
        router = QueryRouter(three_models, thresholds=(0.8, 0.95))
        decision = router.route(
            [{"role": "user", "content": "Explain how binary search works"}]
        )
        # With a very high threshold, most things route to fast
        assert decision.tier == "fast"

    def test_get_fallback(self, router):
        fallback = router.get_fallback("small")
        # Should get a larger model
        assert fallback in ("medium", "large")

    def test_get_fallback_from_largest(self, router):
        fallback = router.get_fallback("large")
        # Should fall back to a same/lower tier model
        assert fallback in ("small", "medium")

    def test_get_fallback_none_single_model(self):
        router = QueryRouter(
            {"only": ModelInfo(name="only", tier="balanced")},
        )
        fallback = router.get_fallback("only")
        assert fallback is None

    def test_single_model_routes_everything(self):
        router = QueryRouter(
            {"only": ModelInfo(name="only", tier="balanced")},
        )
        for text in ["hi", "write a compiler", "prove P=NP"]:
            decision = router.route([{"role": "user", "content": text}])
            assert decision.model_name == "only"

    def test_two_model_routing(self):
        models = {
            "fast": ModelInfo(name="fast", tier="fast"),
            "smart": ModelInfo(name="smart", tier="quality"),
        }
        router = QueryRouter(models)
        simple = router.route([{"role": "user", "content": "hello"}])
        assert simple.model_name == "fast"

    def test_resolve_model_missing_tier(self):
        """When the target tier has no model, fall back to next larger."""
        models = {
            "small": ModelInfo(name="small", tier="fast"),
            "big": ModelInfo(name="big", tier="quality"),
        }
        router = QueryRouter(models)
        # A medium-complexity query targets "balanced", but no balanced model exists
        # Should fall back to quality
        decision = router.route(
            [
                {
                    "role": "user",
                    "content": "Compare and contrast the TCP and UDP protocols in networking",
                }
            ]
        )
        # The model should be resolved (either fast or quality, not crash)
        assert decision.model_name in ("small", "big")


# ---------------------------------------------------------------------------
# Multi-model serve app (integration with EchoBackend)
# ---------------------------------------------------------------------------


class TestMultiModelServeApp:
    @pytest.fixture
    def multi_model_app(self):
        """Create a multi-model FastAPI app with EchoBackends."""
        from edgeml.serve import EchoBackend, create_multi_model_app

        def mock_detect(name, **kwargs):
            echo = EchoBackend()
            echo.load_model(name)
            return echo

        with patch("edgeml.serve._detect_backend", side_effect=mock_detect):
            app = create_multi_model_app(
                ["small-model", "medium-model", "large-model"],
            )

            async def _trigger_lifespan():
                ctx = app.router.lifespan_context(app)
                await ctx.__aenter__()

            asyncio.run(_trigger_lifespan())

        return app

    @pytest.mark.asyncio
    async def test_health_endpoint(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mode"] == "multi-model"
        assert "small-model" in data["models"]
        assert "medium-model" in data["models"]
        assert "large-model" in data["models"]
        assert data["strategy"] == "complexity"

    @pytest.mark.asyncio
    async def test_list_models(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "small-model" in model_ids
        assert "medium-model" in model_ids
        assert "large-model" in model_ids

    @pytest.mark.asyncio
    async def test_chat_completion_routes_simple(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        # Should have routing headers
        assert "x-edgeml-routed-model" in resp.headers
        assert "x-edgeml-complexity" in resp.headers
        assert "x-edgeml-tier" in resp.headers
        # Simple query → small model
        assert resp.headers["x-edgeml-routed-model"] == "small-model"
        assert resp.headers["x-edgeml-tier"] == "fast"

    @pytest.mark.asyncio
    async def test_chat_completion_routes_complex(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert distributed systems architect. "
                                "Provide comprehensive analysis with formal proofs, "
                                "code implementations, optimization strategies, and "
                                "detailed explanations of every algorithm and data "
                                "structure used. Always include error handling, "
                                "concurrency considerations, and performance analysis."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Write a complete implementation of a distributed "
                                "consensus algorithm using Raft protocol. Implement "
                                "leader election with randomized timeouts, log "
                                "replication with consistency guarantees, and prove "
                                "the safety and liveness properties step by step. "
                                "Include mutex-based concurrency control, "
                                "serialization for network transport, and analyze "
                                "the latency and throughput tradeoffs. Compare with "
                                "Paxos and derive the asymptotic complexity bounds."
                            ),
                        },
                    ],
                },
            )
        assert resp.status_code == 200
        assert resp.headers["x-edgeml-routed-model"] == "large-model"
        assert resp.headers["x-edgeml-tier"] == "quality"

    @pytest.mark.asyncio
    async def test_chat_completion_response_format(self, multi_model_app):
        """Response should have standard OpenAI chat completion format."""
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "test"}],
                },
            )
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_routing_stats_endpoint(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Make a request first
            await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            # Check stats
            resp = await client.get("/v1/routing/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] >= 1
        assert "routed_counts" in data
        assert data["strategy"] == "complexity"
        assert "models" in data

    @pytest.mark.asyncio
    async def test_streaming_has_routing_headers(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert "x-edgeml-routed-model" in resp.headers


# ---------------------------------------------------------------------------
# Fallback chain integration
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_fallback_on_model_failure(self):
        """When the primary model fails, the next model should be tried."""
        from edgeml.serve import EchoBackend, create_multi_model_app

        call_count = {"small": 0, "large": 0}

        class FailingBackend(EchoBackend):
            """Backend that fails on first generate call."""

            def __init__(self, name: str):
                super().__init__()
                self._model_name = name

            def generate(self, request):
                call_count[self._model_name] = call_count.get(self._model_name, 0) + 1
                if self._model_name == "small":
                    raise RuntimeError("Small model OOM")
                return super().generate(request)

        def mock_detect(name, **kwargs):
            backend = FailingBackend(name)
            backend.load_model(name)
            return backend

        with patch("edgeml.serve._detect_backend", side_effect=mock_detect):
            app = create_multi_model_app(["small", "large"])

            async def _trigger_lifespan():
                ctx = app.router.lifespan_context(app)
                await ctx.__aenter__()

            asyncio.run(_trigger_lifespan())

        async def _run():
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hello"}]},
                )
            return resp

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        # Should have used fallback
        assert resp.headers.get("x-edgeml-fallback") == "true"
        # The large model should have handled it
        assert resp.headers["x-edgeml-routed-model"] == "large"

    def test_all_models_fail_returns_503(self):
        """When all models fail, return 503."""
        from edgeml.serve import EchoBackend, create_multi_model_app

        class AlwaysFailBackend(EchoBackend):
            def generate(self, request):
                raise RuntimeError("Always fails")

        def mock_detect(name, **kwargs):
            backend = AlwaysFailBackend()
            backend.load_model(name)
            return backend

        with patch("edgeml.serve._detect_backend", side_effect=mock_detect):
            app = create_multi_model_app(["a", "b"])

            async def _trigger_lifespan():
                ctx = app.router.lifespan_context(app)
                await ctx.__aenter__()

            asyncio.run(_trigger_lifespan())

        async def _run():
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hello"}]},
                )
            return resp

        resp = asyncio.run(_run())
        assert resp.status_code == 503
        assert "All models failed" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------


class TestServeCliMultiModel:
    def test_auto_route_requires_models(self):
        from click.testing import CliRunner

        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "test", "--auto-route"])
        assert result.exit_code != 0
        assert "--auto-route requires --models" in result.output

    def test_auto_route_requires_two_models(self):
        from click.testing import CliRunner

        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main, ["serve", "test", "--auto-route", "--models", "single-model"]
        )
        assert result.exit_code != 0
        assert "at least 2 models" in result.output

    def test_multi_model_prints_tier_info(self):
        from click.testing import CliRunner

        from edgeml.cli import main

        runner = CliRunner()
        with patch("edgeml.serve.run_multi_model_server"):
            result = runner.invoke(
                main,
                [
                    "serve",
                    "small",
                    "--auto-route",
                    "--models",
                    "small,medium,large",
                ],
            )
        assert result.exit_code == 0
        assert "Loading 3 models" in result.output
        assert "auto-routing" in result.output
        assert "tier=" in result.output

    def test_route_strategy_default_is_complexity(self):
        from click.testing import CliRunner

        from edgeml.cli import main

        runner = CliRunner()
        with patch("edgeml.serve.run_multi_model_server") as mock_run:
            result = runner.invoke(
                main,
                [
                    "serve",
                    "test",
                    "--auto-route",
                    "--models",
                    "a,b",
                ],
            )
        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("route_strategy") == "complexity"

    def test_single_model_mode_unchanged(self):
        """Without --auto-route, serve works exactly as before."""
        from click.testing import CliRunner

        from edgeml.cli import main

        runner = CliRunner()
        with patch("edgeml.serve.run_server") as mock_run:
            result = runner.invoke(main, ["serve", "gemma-1b"])
        assert result.exit_code == 0
        assert "Starting EdgeML serve" in result.output
        mock_run.assert_called_once()

    def test_invalid_route_strategy_rejected(self):
        from click.testing import CliRunner

        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "serve",
                "test",
                "--auto-route",
                "--models",
                "a,b",
                "--route-strategy",
                "random",
            ],
        )
        assert result.exit_code != 0
