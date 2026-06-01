"""sherpa-onnx TTS catalog helpers.

sherpa-onnx (k2-fsa) ships VITS/Piper/Kokoro TTS models packaged as ONNX.
Production TTS synthesis now runs through the native OCT_* ABI runtime; this
module remains as the shared model/catalog/voice helper layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from octomil.errors import OctomilError, OctomilErrorCode

# Supported TTS models -- name -> (family, default voice).
# ``family`` is retained as catalog metadata for planner/listing behavior;
# synthesis itself is native-only and does not instantiate Python sherpa-onnx
# configs.
# Voice catalogs are model-specific; the second tuple element is the documented
# default voice label. Pocket has no native voice catalog — its "voices" are
# reference profiles published by the planner — so the default voice for Pocket
# models is the empty string.
_SHERPA_TTS_MODELS: dict[str, tuple[str, str]] = {
    "kokoro-82m": ("kokoro", "af_bella"),
    # Legacy v0.19 bundle, retained as an explicit-pin id alongside
    # ``kokoro-82m``. ``af_bella`` is at sid=1 in v0.19's voices.bin
    # and is the default for both bundles.
    "kokoro-en-v0_19": ("kokoro", "af_bella"),
    "piper-en-amy": ("vits", "amy"),
    "piper-en-ryan": ("vits", "ryan"),
    # PocketTTS — int8-quantized few-shot voice-cloning engine. The
    # planner is responsible for selecting this id; clients should
    # call ``@app/<slug>/tts`` instead of pinning the runtime model.
    "pocket-tts-int8": ("pocket", ""),
}


def _default_voice(model_name: str) -> str:
    entry = _SHERPA_TTS_MODELS.get(model_name.lower())
    return entry[1] if entry else ""


# Per-artifact Kokoro voice catalogs. Position in the tuple ==
# sherpa-onnx speaker id in the corresponding voices.bin.
#
# IMPORTANT: voice ordering is bundle-specific. A 28-name "modern"
# Kokoro catalog (af_heart, am_echo, …) is NOT interchangeable with
# the 11-name kokoro-en-v0_19 catalog the SDK currently ships —
# sherpa-onnx clamps out-of-range sids to 0, so a mismatched table
# silently aliases every "missing" voice to the default speaker.
#
# These tables are *legacy fallbacks*. The authoritative source is a
# ``voices.txt`` sidecar under the prepared artifact directory,
# materialized from the static recipe's ``voice_manifest`` field.
# The fallback only fires when a sidecar is absent — e.g. an
# artifact someone hand-staged before voices.txt materialization
# shipped — and is keyed by model id rather than a global "kokoro =
# these N names" assumption.

# kokoro-en-v0_19 — the legacy English-only bundle, still resolvable
# under the explicit ``kokoro-en-v0_19`` model id.
_KOKORO_EN_V0_19_VOICES: tuple[str, ...] = (
    "af",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
)

# kokoro-multi-lang-v1_0 — the bundle ``kokoro-82m`` resolves to as
# of the v1.0 cutover. 53 speakers across 8 languages; ordering is
# pinned to upstream's ``scripts/kokoro/v1.0/generate_voices_bin.py``
# so the fallback can never silently drift from voices.bin.
_KOKORO_MULTI_LANG_V1_0_VOICES: tuple[str, ...] = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
    "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel",
    "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex",
    "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
)  # fmt: skip

# Per-model legacy fallback catalog. Used ONLY when no voices.txt
# sidecar is present. Keep tightly scoped: an unknown model id
# falls through to "fail loudly" so callers can't accidentally
# inherit some other artifact's catalog.
_LEGACY_KOKORO_FALLBACK_CATALOGS: dict[str, tuple[str, ...]] = {
    "kokoro-82m": _KOKORO_MULTI_LANG_V1_0_VOICES,
    "kokoro-en-v0_19": _KOKORO_EN_V0_19_VOICES,
}

# Back-compat alias. Old import path
# ``octomil.runtime.engines.sherpa._KOKORO_VOICES`` resolves to the
# active default artifact's catalog so external callers keep working,
# but the canonical accessor is ``catalog_for_model(model_name)``.
_KOKORO_VOICES: tuple[str, ...] = _KOKORO_MULTI_LANG_V1_0_VOICES


def catalog_for_model(model_name: str) -> tuple[str, ...]:
    """Return the legacy fallback voice catalog for ``model_name``.

    Empty tuple when the model has no declared catalog. Callers that
    need authoritative ordering should read ``voices.txt`` from the
    prepared artifact directory; this helper is the *fallback* used
    only when the sidecar is missing.

    Keying solely on ``model_name`` is unsafe when an old prepared
    dir from a previous artifact identity is still on disk (e.g. a
    pre-cutover ``kokoro-82m`` v0.19 dir whose voices.txt sidecar
    predates the manifest patch). The richer
    :func:`fallback_catalog_for_artifact` resolves this by reading
    the materialized ``VERSION`` sidecar / layout signals.
    """
    return _LEGACY_KOKORO_FALLBACK_CATALOGS.get(model_name.lower(), ())


def fallback_catalog_for_artifact(model_name: str, model_dir: str) -> tuple[str, ...]:
    """Pick a legacy fallback catalog using artifact-on-disk signals.

    Resolution order:

      1. ``VERSION`` sidecar materialized by the static recipe.
         ``kokoro-en-v0_19`` → 11-speaker catalog;
         ``kokoro-multi-lang-v1_0`` → 53-speaker catalog.
      2. Directory-shape inference. If neither layout signal is
         present (artifact looks neither v0.19 nor v1.0), or the
         artifact's identity contradicts the model id (e.g. a
         pre-cutover v0.19 artifact still parked under
         ``kokoro-82m``), return an empty tuple — the voice resolver
         then refuses the explicit voice path rather than silently
         aliasing names like ``bm_george`` to the wrong sid.
      3. As a last resort for sidecar-less artifacts whose layout
         IS clearly recognizable, fall back to the model-id catalog
         only when it agrees with the layout signal.
    """
    version_path = os.path.join(model_dir, "VERSION")
    declared_version = ""
    if os.path.isfile(version_path):
        try:
            with open(version_path, encoding="utf-8") as f:
                declared_version = f.read().strip()
        except OSError:
            declared_version = ""

    if declared_version == "kokoro-en-v0_19":
        return _KOKORO_EN_V0_19_VOICES
    if declared_version == "kokoro-multi-lang-v1_0":
        return _KOKORO_MULTI_LANG_V1_0_VOICES

    # No VERSION sidecar — infer from layout. v1.0 uniquely ships
    # dict/jieba.dict.utf8; v0.19 uniquely ships espeak-ng-data
    # WITHOUT the lexicon files (v1.0 ships both).
    has_dict = os.path.isfile(os.path.join(model_dir, "dict", "jieba.dict.utf8"))
    has_lexicon_us = os.path.isfile(os.path.join(model_dir, "lexicon-us-en.txt"))
    has_espeak = os.path.isfile(os.path.join(model_dir, "espeak-ng-data", "phontab"))

    if has_dict and has_lexicon_us:
        layout_catalog: tuple[str, ...] = _KOKORO_MULTI_LANG_V1_0_VOICES
    elif has_espeak and not has_dict and not has_lexicon_us:
        layout_catalog = _KOKORO_EN_V0_19_VOICES
    else:
        # Layout is ambiguous (or unrecognized). Refuse to guess —
        # an old pre-manifest dir under ``kokoro-82m`` could
        # otherwise inherit the v1.0 catalog and re-introduce the
        # silent aliasing bug.
        return ()

    # Cross-check against the model-id catalog. If they disagree,
    # treat the dir as ambiguous (operator-staged bytes that don't
    # match the canonical recipe).
    name_catalog = catalog_for_model(model_name)
    if name_catalog and name_catalog is not layout_catalog:
        return ()
    return layout_catalog


def _read_voice_manifest(model_dir: str) -> tuple[str, ...]:
    """Read ``voices.txt`` from ``model_dir`` and return the ordered
    list of speaker names. Returns an empty tuple when the sidecar
    is missing. Trims trailing whitespace and skips blank lines so a
    crash-truncated final newline doesn't shift speaker ids.
    """
    sidecar = os.path.join(model_dir, "voices.txt")
    if not os.path.exists(sidecar):
        return ()
    with open(sidecar, encoding="utf-8") as f:
        return tuple(line.strip() for line in f if line.strip())


def _read_artifact_version(model_dir: str) -> str:
    """Read the ``VERSION`` sidecar (or empty string)."""
    version_path = os.path.join(model_dir, "VERSION")
    if not os.path.isfile(version_path):
        return ""
    try:
        with open(version_path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


@dataclass(frozen=True)
class ResolvedVoiceCatalog:
    """Resolved projection of a TTS voice catalog.

    Single-source-of-truth for voice validation and listing
    (``list_speech_voices``). Position in ``voices`` matches
    sherpa-onnx speaker id; ``source`` is provenance — either
    ``"voices_txt"`` (read from the prepared artifact's sidecar),
    ``"static_recipe"`` (no sidecar yet, manifest came from the
    recipe), or ``""`` (no catalog known).
    """

    voices: tuple[str, ...]
    source: str = ""
    artifact_version: str = ""


# Process-lifetime cache for ``resolve_voice_catalog``. Cleared via
# ``release_voice_catalog_cache`` (cascaded from
# ``ExecutionKernel.release_warmed_backends``).
#
# Reviewer P1 fix: the cache key includes the mtime_ns of
# ``voices.txt`` and ``VERSION`` under the prepared dir.
# ``PrepareManager`` stores artifacts at a path derived from
# ``artifact_id`` — NOT digest — so a v2 prepare can overwrite v1
# contents IN PLACE at the same directory string. A path-only cache
# key would serve the v1 voice catalog after the v2 prepare lands,
# reopening the voice-drift class of bugs (listing rejects new
# voices, sid resolution maps against stale ordering).
# Stat-based invalidation: one syscall per lookup, content-
# addresses for free, no event-bus hookup needed. When either
# sidecar's mtime_ns changes, the cache key changes and the cached
# entry is unreachable.
_VoiceCatalogCacheKey = tuple[
    str,  # model_name
    Optional[str],  # prepared_model_dir
    Optional[int],  # voices.txt mtime_ns (None when absent)
    Optional[int],  # VERSION mtime_ns (None when absent)
    tuple[str, ...],  # static_recipe_manifest
    str,  # static_recipe_artifact_version
]
_VOICE_CATALOG_CACHE: dict[_VoiceCatalogCacheKey, "ResolvedVoiceCatalog"] = {}


def release_voice_catalog_cache() -> None:
    """Drop every cached voice-catalog resolution.

    Idempotent. Called from ``ExecutionKernel.release_warmed_backends``
    so the existing public "drop my caches" surface clears this too.
    Tests use it directly to reset between runs.
    """
    _VOICE_CATALOG_CACHE.clear()


def _stat_mtime_ns(path: str) -> Optional[int]:
    """Return ``stat.st_mtime_ns`` for ``path``, or ``None`` if absent.

    Cheap content-fingerprint primitive for the voice-catalog cache
    key — one syscall, no read. Returning ``None`` when the file is
    missing is meaningful: a sidecar-less prepared dir caches under
    a ``(None, None)`` mtime tuple, and any future prepare that
    materializes the sidecar lands in a different cache slot.
    """
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def resolve_voice_catalog(
    model_name: str,
    *,
    prepared_model_dir: Optional[str] = None,
    static_recipe_manifest: tuple[str, ...] = (),
    static_recipe_artifact_version: str = "",
) -> ResolvedVoiceCatalog:
    """Single resolver shared by synthesis, validation, and listing.

    Resolution order:

      1. ``voices.txt`` sidecar in ``prepared_model_dir`` —
         authoritative for the artifact actually on disk.
      2. ``VERSION`` + layout fallback under ``prepared_model_dir``
         (handles sidecar-less prepared dirs from before the
         manifest patch).
      3. ``static_recipe_manifest`` — used when no prepared dir
         exists yet (the listing path can preview the catalog
         without forcing a download).
      4. Model-id legacy fallback (``catalog_for_model``), only
         when no other signal is available.

    Returns an empty catalog (``voices=()``) when none of the above
    yields a result; callers translate that into a strict refusal
    for the explicit-voice path or a "no catalog known" listing.

    Cached for the lifetime of the process (or until
    :func:`release_voice_catalog_cache` is called). The cache key
    includes the mtime_ns of ``voices.txt`` and ``VERSION`` under
    ``prepared_model_dir`` so a re-prepare that overwrites the dir
    in place naturally invalidates the cache — the new mtimes
    produce a new cache key and the previous entry becomes
    unreachable. (PrepareManager keys artifacts by ``artifact_id``,
    not digest, so in-place overwrites do happen on version
    changes.)
    """
    voices_mtime: Optional[int] = None
    version_mtime: Optional[int] = None
    if prepared_model_dir:
        voices_mtime = _stat_mtime_ns(os.path.join(prepared_model_dir, "voices.txt"))
        version_mtime = _stat_mtime_ns(os.path.join(prepared_model_dir, "VERSION"))
    cache_key: _VoiceCatalogCacheKey = (
        model_name,
        prepared_model_dir,
        voices_mtime,
        version_mtime,
        static_recipe_manifest,
        static_recipe_artifact_version,
    )
    cached = _VOICE_CATALOG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = _resolve_voice_catalog_uncached(
        model_name,
        prepared_model_dir=prepared_model_dir,
        static_recipe_manifest=static_recipe_manifest,
        static_recipe_artifact_version=static_recipe_artifact_version,
    )
    _VOICE_CATALOG_CACHE[cache_key] = result
    return result


def _resolve_voice_catalog_uncached(
    model_name: str,
    *,
    prepared_model_dir: Optional[str] = None,
    static_recipe_manifest: tuple[str, ...] = (),
    static_recipe_artifact_version: str = "",
) -> ResolvedVoiceCatalog:
    """Inner resolver — see :func:`resolve_voice_catalog` for the contract.

    Split out so the public entry point owns caching and the body
    stays a pure function of its inputs.
    """
    if prepared_model_dir:
        sidecar = _read_voice_manifest(prepared_model_dir)
        if sidecar:
            return ResolvedVoiceCatalog(
                voices=sidecar,
                source="voices_txt",
                artifact_version=_read_artifact_version(prepared_model_dir),
            )
        layout = fallback_catalog_for_artifact(model_name, prepared_model_dir)
        if layout:
            return ResolvedVoiceCatalog(
                voices=layout,
                source="voices_txt",  # derived from artifact-on-disk signals
                artifact_version=_read_artifact_version(prepared_model_dir),
            )
        # Prepared dir exists but is ambiguous (layout + model id
        # disagree, or layout is unrecognized). Refuse to fall
        # through to the model-id catalog or the recipe manifest:
        # an old pre-manifest dir under the same model id would
        # otherwise inherit the WRONG catalog and re-introduce the
        # silent sid-aliasing bug.
        return ResolvedVoiceCatalog(voices=(), source="", artifact_version="")

    if static_recipe_manifest:
        return ResolvedVoiceCatalog(
            voices=static_recipe_manifest,
            source="static_recipe",
            artifact_version=static_recipe_artifact_version,
        )

    name_fallback = catalog_for_model(model_name)
    if name_fallback:
        return ResolvedVoiceCatalog(
            voices=name_fallback,
            source="static_recipe",
            artifact_version="",
        )

    return ResolvedVoiceCatalog(voices=(), source="", artifact_version="")


def is_sherpa_tts_model(model_name: str) -> bool:
    """Check if a model name refers to a sherpa-onnx TTS model.

    Means "known model id," not "installed and runnable." For runnable
    detection, the kernel asks PrepareManager whether a prepared
    artifact dir exists for ``(model, capability='tts')`` — there is no
    legacy "is staged" path.
    """
    return model_name.lower() in _SHERPA_TTS_MODELS


def resolve_voice_sid(
    model_name: str,
    voice: str,
    *,
    prepared_model_dir: str,
    explicit: bool = True,
) -> int:
    """Resolve a named catalog voice to its speaker id.

    This is the legacy backend's strict voice mapping factored into the
    catalog layer. Native TTS owns synthesis now, but list/preflight tests
    still need one resolver to prove advertised voices map to stable ids.
    """
    if not voice:
        return 0

    manifest = resolve_voice_catalog(model_name, prepared_model_dir=prepared_model_dir).voices
    if not manifest:
        if explicit:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"voice_not_supported_for_model: model {model_name!r} has no declared "
                    "voice catalog (no voices.txt sidecar, no built-in fallback). "
                    "Pass voice=None to use the default speaker, or run "
                    "client.prepare(model, capability='tts') to materialize the "
                    "artifact's voice manifest."
                ),
            )
        return 0

    target = voice.strip().lower()
    for idx, name in enumerate(manifest):
        if name.lower() == target:
            return idx

    if not explicit:
        return 0

    raise OctomilError(
        code=OctomilErrorCode.INVALID_INPUT,
        message=(
            f"voice_not_supported_for_model: voice {voice!r} is not in the speaker "
            f"catalog for model {model_name!r}. Supported voices: {', '.join(manifest)}."
        ),
    )


def resolve_default_voice_label(model_name: str, *, prepared_model_dir: str) -> str:
    """Return the effective default voice label for a prepared artifact."""
    manifest = resolve_voice_catalog(model_name, prepared_model_dir=prepared_model_dir).voices
    if manifest:
        return manifest[0]
    return _default_voice(model_name) or ""
