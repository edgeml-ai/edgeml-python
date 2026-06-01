"""sherpa-onnx model and voice-catalog helpers."""

from octomil.runtime.engines.sherpa.catalog import (
    _KOKORO_VOICES,
    ResolvedVoiceCatalog,
    catalog_for_model,
    fallback_catalog_for_artifact,
    is_sherpa_tts_model,
    resolve_default_voice_label,
    resolve_voice_catalog,
    resolve_voice_sid,
)

TIER = "supported"

__all__ = [
    "TIER",
    "ResolvedVoiceCatalog",
    "catalog_for_model",
    "fallback_catalog_for_artifact",
    "is_sherpa_tts_model",
    "resolve_default_voice_label",
    "resolve_voice_catalog",
    "resolve_voice_sid",
    "_KOKORO_VOICES",
]
