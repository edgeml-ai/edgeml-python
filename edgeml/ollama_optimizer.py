"""Backward compatibility — use ``edgeml.model_optimizer`` instead."""

from edgeml.model_optimizer import (  # noqa: F401
    BYTES_PER_PARAM,
    KV_CACHE_PER_1K_TOKENS,
    MODEL_SIZES,
    QUANT_SPEED_FACTORS,
    MemoryStrategy,
    ModelOptimizer,
    ModelRecommendation,
    QuantOffloadResult,
    SpeedEstimate,
    _kv_cache_gb,
    _model_memory_gb,
    _total_memory_gb,
)

# Backward compat alias
OllamaOptimizer = ModelOptimizer
