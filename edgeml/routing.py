"""
Query routing for multi-model serving.

Routes incoming queries to the most appropriate model based on estimated
complexity.  Simple queries (greetings, short factual questions) go to
the smallest/fastest model; complex queries (code generation, multi-step
reasoning) go to the largest/most capable model.

The complexity heuristic is pure Python — no ML model required.

Usage::

    from edgeml.routing import QueryRouter, ModelInfo

    models = {
        "smollm-360m": ModelInfo(name="smollm-360m", tier="fast", param_b=0.36),
        "phi-4-mini":  ModelInfo(name="phi-4-mini",  tier="balanced", param_b=3.8),
        "llama-3.2-3b": ModelInfo(name="llama-3.2-3b", tier="quality", param_b=3.0),
    }
    router = QueryRouter(models, strategy="complexity")
    chosen = router.route(messages)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

# Tiers ordered from smallest to largest capability.
TIER_ORDER: list[str] = ["fast", "balanced", "quality"]


@dataclass
class ModelInfo:
    """Metadata about a loaded model used for routing decisions."""

    name: str
    tier: str = "balanced"
    param_b: float = 0.0  # parameter count in billions (for display)
    loaded: bool = True

    @property
    def tier_index(self) -> int:
        """Numeric rank of the tier (higher = more capable)."""
        try:
            return TIER_ORDER.index(self.tier)
        except ValueError:
            return 1  # default to balanced


@dataclass
class RoutingDecision:
    """Result of a routing decision with metadata for telemetry."""

    model_name: str
    complexity_score: float
    tier: str
    strategy: str
    fallback_chain: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Complexity heuristic signals
# ---------------------------------------------------------------------------

# Words that indicate simple / conversational queries.
_SIMPLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(hi|hello|hey|howdy|greetings|yo|sup)\b", re.IGNORECASE),
    re.compile(r"\b(thanks|thank you|bye|goodbye|see ya)\b", re.IGNORECASE),
    re.compile(r"\b(what time|what day|what year|how old)\b", re.IGNORECASE),
    re.compile(r"\bwhat is (a |an |the )?\w+\b", re.IGNORECASE),
    re.compile(r"\bdefine \w+\b", re.IGNORECASE),
]

# Words/phrases that indicate complex queries.
_COMPLEX_PATTERNS: list[re.Pattern[str]] = [
    # Code generation
    re.compile(
        r"\b(write|implement|code|program|function|class|algorithm|refactor|debug)\b",
        re.IGNORECASE,
    ),
    # Reasoning
    re.compile(
        r"\b(explain why|reason|analyze|compare|contrast|evaluate|critique)\b",
        re.IGNORECASE,
    ),
    # Multi-step
    re.compile(
        r"\b(step by step|step-by-step|first .* then|multi-step)\b",
        re.IGNORECASE,
    ),
    # Math / logic
    re.compile(
        r"\b(prove|derive|calculate|compute|integral|derivative|equation|theorem)\b",
        re.IGNORECASE,
    ),
    # Creative / long-form
    re.compile(
        r"\b(write a (story|essay|article|poem|report|document))\b",
        re.IGNORECASE,
    ),
    # Technical terms
    re.compile(
        r"\b(API|REST|GraphQL|microservice|kubernetes|docker|terraform|CICD|CI/CD)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(neural network|transformer|attention|backpropagation|gradient)\b",
        re.IGNORECASE,
    ),
]

# Technical / uncommon vocabulary (presence raises complexity).
_TECHNICAL_WORDS: set[str] = {
    "asynchronous",
    "concurrency",
    "parallelism",
    "mutex",
    "semaphore",
    "deadlock",
    "recursion",
    "polymorphism",
    "inheritance",
    "abstraction",
    "encapsulation",
    "middleware",
    "serialization",
    "deserialization",
    "latency",
    "throughput",
    "idempotent",
    "deterministic",
    "stochastic",
    "heuristic",
    "eigenvalue",
    "convolution",
    "embedding",
    "tokenization",
    "quantization",
    "optimization",
    "regularization",
    "normalization",
    "hyperparameter",
    "inference",
    "fine-tuning",
    "federated",
    "differential",
    "cryptographic",
    "authentication",
    "authorization",
    "orchestration",
    "containerization",
}


def _word_count(text: str) -> int:
    """Count whitespace-delimited tokens."""
    return len(text.split())


def _estimate_complexity(
    text: str,
    system_prompt: str = "",
    turn_count: int = 1,
) -> float:
    """Estimate query complexity on a 0.0 (trivial) to 1.0 (complex) scale.

    Signals used:
    1. Token/word count — short queries tend to be simpler.
    2. Vocabulary complexity — presence of technical terms.
    3. Pattern matching — greeting vs. code/reasoning patterns.
    4. System prompt length — longer system prompts imply complex tasks.
    5. Multi-turn context — deeper conversations are generally harder.

    Returns a float in [0.0, 1.0].
    """
    signals: list[float] = []

    # 1. Length signal (0-1, log-scaled)
    wc = _word_count(text)
    # 1 word → ~0.0, 10 words → ~0.33, 50 words → ~0.56, 200 words → ~0.77
    length_score = min(math.log1p(wc) / math.log1p(200), 1.0)
    signals.append(length_score * 0.20)

    # 2. Simple pattern match (each match lowers complexity)
    simple_hits = sum(1 for p in _SIMPLE_PATTERNS if p.search(text))
    simple_penalty = min(simple_hits * 0.15, 0.30)
    signals.append(-simple_penalty)

    # 3. Complex pattern match (each match raises complexity)
    complex_hits = sum(1 for p in _COMPLEX_PATTERNS if p.search(text))
    complex_boost = min(complex_hits * 0.12, 0.40)
    signals.append(complex_boost)

    # 4. Technical vocabulary ratio
    words_lower = {w.lower().strip(".,;:!?()[]{}\"'") for w in text.split()}
    tech_count = len(words_lower & _TECHNICAL_WORDS)
    tech_score = min(tech_count / 5.0, 1.0) * 0.15
    signals.append(tech_score)

    # 5. Code indicators (backticks, indentation patterns)
    code_indicators = text.count("```") + text.count("def ") + text.count("class ")
    code_score = min(code_indicators / 3.0, 1.0) * 0.10
    signals.append(code_score)

    # 6. System prompt length (longer → more complex task)
    if system_prompt:
        sys_wc = _word_count(system_prompt)
        sys_score = min(math.log1p(sys_wc) / math.log1p(500), 1.0) * 0.10
        signals.append(sys_score)

    # 7. Multi-turn depth (deeper → harder)
    turn_score = min((turn_count - 1) / 10.0, 1.0) * 0.05
    signals.append(turn_score)

    # Base complexity (avoids everything summing to exactly 0)
    base = 0.25

    raw = base + sum(signals)
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------


class QueryRouter:
    """Routes queries to the best model based on complexity.

    Parameters
    ----------
    models:
        Dict mapping model name to ``ModelInfo``.  Models should be
        ordered from smallest to largest (or tagged with tiers).
    strategy:
        Routing strategy.  Currently only ``"complexity"`` is supported.
    thresholds:
        Two-element tuple ``(low, high)`` defining tier boundaries.
        Complexity in ``[0, low)`` → fast, ``[low, high)`` → balanced,
        ``[high, 1.0]`` → quality.
    """

    def __init__(
        self,
        models: dict[str, ModelInfo],
        strategy: str = "complexity",
        thresholds: tuple[float, float] = (0.3, 0.7),
    ) -> None:
        if not models:
            raise ValueError("At least one model must be provided")
        if strategy != "complexity":
            raise ValueError(
                f"Unknown routing strategy '{strategy}'. "
                "Supported strategies: complexity"
            )

        self.models = models
        self.strategy = strategy
        self.thresholds = thresholds

        # Build tier → model mapping (pick first model per tier)
        self._tier_models: dict[str, str] = {}
        for name, info in models.items():
            if info.tier not in self._tier_models:
                self._tier_models[info.tier] = name

        # Ordered list of model names from smallest to largest tier
        self._ordered_models: list[str] = sorted(
            models.keys(),
            key=lambda n: models[n].tier_index,
        )

        logger.info(
            "QueryRouter initialised: strategy=%s, models=%s, thresholds=%s",
            strategy,
            list(models.keys()),
            thresholds,
        )

    def route(self, messages: list[dict[str, str]]) -> RoutingDecision:
        """Determine which model should handle this request.

        Parameters
        ----------
        messages:
            OpenAI-style message list (dicts with ``role`` and ``content``).

        Returns
        -------
        RoutingDecision with the chosen model name, complexity score,
        tier, and fallback chain.
        """
        # Extract relevant text
        user_text = ""
        system_prompt = ""
        turn_count = 0

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_text = content  # use the last user message
                turn_count += 1

        complexity = _estimate_complexity(
            user_text,
            system_prompt=system_prompt,
            turn_count=turn_count,
        )

        # Map complexity to tier
        low, high = self.thresholds
        if complexity < low:
            target_tier = "fast"
        elif complexity < high:
            target_tier = "balanced"
        else:
            target_tier = "quality"

        # Find the best model for the target tier
        model_name = self._resolve_model(target_tier)

        # Build fallback chain: all models from current tier upward
        fallback_chain = self._build_fallback_chain(model_name)

        decision = RoutingDecision(
            model_name=model_name,
            complexity_score=round(complexity, 4),
            tier=target_tier,
            strategy=self.strategy,
            fallback_chain=fallback_chain,
        )

        logger.debug(
            "Routing decision: complexity=%.3f, tier=%s, model=%s",
            complexity,
            target_tier,
            model_name,
        )

        return decision

    def _resolve_model(self, target_tier: str) -> str:
        """Find the best available model for a tier, falling back upward."""
        # Try exact tier match
        if target_tier in self._tier_models:
            return self._tier_models[target_tier]

        # Fall back to the next larger tier
        try:
            target_idx = TIER_ORDER.index(target_tier)
        except ValueError:
            target_idx = 0

        for tier in TIER_ORDER[target_idx:]:
            if tier in self._tier_models:
                return self._tier_models[tier]

        # Last resort: fall back downward
        for tier in reversed(TIER_ORDER[:target_idx]):
            if tier in self._tier_models:
                return self._tier_models[tier]

        # Absolute fallback: first model
        return self._ordered_models[0]

    def _build_fallback_chain(self, primary: str) -> list[str]:
        """Build ordered list of fallback models (excluding primary).

        Order: models with higher capability than primary first, then lower.
        """
        primary_idx = self.models[primary].tier_index
        higher = [
            n
            for n in self._ordered_models
            if n != primary and self.models[n].tier_index > primary_idx
        ]
        lower = [
            n
            for n in self._ordered_models
            if n != primary and self.models[n].tier_index < primary_idx
        ]
        # Same-tier models that aren't the primary
        same = [
            n
            for n in self._ordered_models
            if n != primary and self.models[n].tier_index == primary_idx
        ]
        return higher + same + lower

    def get_fallback(self, failed_model: str) -> Optional[str]:
        """Return the next model to try after ``failed_model`` fails.

        Tries models with higher capability first, then lower.
        Returns ``None`` if no fallback is available.
        """
        failed_idx = self.models.get(failed_model, ModelInfo(name="")).tier_index
        # Try higher-tier models first
        for name in self._ordered_models:
            if name != failed_model and self.models[name].tier_index > failed_idx:
                return name
        # Then try same-tier
        for name in self._ordered_models:
            if name != failed_model and self.models[name].tier_index == failed_idx:
                return name
        # Then try lower-tier
        for name in reversed(self._ordered_models):
            if name != failed_model and self.models[name].tier_index < failed_idx:
                return name
        return None


def assign_tiers(
    model_names: list[str],
    thresholds: tuple[float, float] = (0.3, 0.7),
) -> dict[str, ModelInfo]:
    """Auto-assign tiers to an ordered list of model names.

    Assumes ``model_names`` is ordered from smallest to largest.
    Splits into three tiers: fast, balanced, quality.

    For 1 model:  all traffic goes to it (balanced).
    For 2 models: first is fast, second is quality.
    For 3+:       even split across fast/balanced/quality.
    """
    n = len(model_names)
    if n == 0:
        return {}

    if n == 1:
        return {model_names[0]: ModelInfo(name=model_names[0], tier="balanced")}

    if n == 2:
        return {
            model_names[0]: ModelInfo(name=model_names[0], tier="fast"),
            model_names[1]: ModelInfo(name=model_names[1], tier="quality"),
        }

    # 3+ models: divide into thirds
    result: dict[str, ModelInfo] = {}
    fast_end = n // 3
    quality_start = n - (n // 3)

    for i, name in enumerate(model_names):
        if i < fast_end:
            tier = "fast"
        elif i >= quality_start:
            tier = "quality"
        else:
            tier = "balanced"
        result[name] = ModelInfo(name=name, tier=tier)

    return result
