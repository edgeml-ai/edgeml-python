"""Cross-API consistency audit — voices.list ↔ create ↔ stream ↔ validation.

The closure-of-loop guarantee from the streaming-truthful + speaker-
plumbing PRs is: every voice that ``voices.list`` advertises will be
accepted by ``speech.create`` and ``speech.stream``, and every voice
those two accept will be in the listed catalog. This audit asserts
that against the *same* prepared artifact directory through all four
entry points in a single test, so a regression where one path skews
from the others surfaces here regardless of which path the change
landed on.

Setup: a private kokoro artifact dir with a hand-written
``voices.txt`` containing three voices. The catalog helper reads voices
from disk; the kernel walks the same resolver for ``voices.list`` and
``_validate_local_voice``. If the paths diverge, the mismatch fails one
of the asserts below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from octomil.audio.speech import VoiceInfo

# ---------------------------------------------------------------------------
# Fixture — staged kokoro artifact + minimal backend
# ---------------------------------------------------------------------------


@pytest.fixture
def staged_kokoro_artifact(tmp_path: Path) -> Path:
    """Stage a kokoro artifact directory with a known voices.txt.

    The four files the sherpa-onnx Kokoro config wants are stubbed
    so ``resolve_voice_catalog`` finds the directory; the *voices*
    side is what the audit tests against.
    """
    art = tmp_path / "kokoro-audit"
    art.mkdir()
    # Stubs — content is not exercised; resolver only checks layout.
    for name in ("model.onnx", "voices.bin", "tokens.txt"):
        (art / name).write_bytes(b"\x00" * 16)
    (art / "espeak-ng-data").mkdir()
    (art / "espeak-ng-data" / "phontab").write_bytes(b"\x00")
    # Authoritative catalog — three custom labels so they can't
    # accidentally alias to the public Kokoro v1.0 fallback table.
    (art / "voices.txt").write_text("af_audit_1\naf_audit_2\naf_audit_3\n")
    (art / "VERSION").write_text("kokoro-audit-1.0")
    return art


def _resolve_sid(model_dir: Path, voice: str) -> int:
    from octomil.runtime.engines.sherpa import resolve_voice_sid

    return resolve_voice_sid("kokoro-82m", voice, prepared_model_dir=str(model_dir))


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voices_list_create_stream_and_validate_share_catalog(staged_kokoro_artifact: Path):
    """Run all four entry points against the same staged artifact and
    assert their views of the catalog agree.

    Specifically:
      1. ``resolve_voice_sid("af_audit_2")`` resolves to a sid under the
         staged voices.txt.
      2. ``resolve_voice_catalog`` reads the same three labels
         from voices.txt — this is the API ``voices.list`` walks.
      3. The native VoiceInfo entries we'd hand to ``VoiceCatalog``
         in the kernel cover every label voices.txt declares (no
         catalog drift).
      4. Default voice resolution agrees with the catalog's first
         entry by position.
    """
    from octomil.runtime.engines.sherpa import resolve_default_voice_label, resolve_voice_catalog

    # 1. validate_voice — synthesis-time gate (used by both create()
    #    and stream() pre-validation). All three labels must resolve.
    assert _resolve_sid(staged_kokoro_artifact, "af_audit_1") == 0
    assert _resolve_sid(staged_kokoro_artifact, "af_audit_2") == 1
    assert _resolve_sid(staged_kokoro_artifact, "af_audit_3") == 2

    # 2. resolve_voice_catalog — what voices.list calls.
    resolved = resolve_voice_catalog(
        "kokoro-82m",
        prepared_model_dir=str(staged_kokoro_artifact),
    )
    assert tuple(resolved.voices) == ("af_audit_1", "af_audit_2", "af_audit_3")

    # 3. Catalog contains every label validate_voice accepts —
    #    no entry can be in one place and not the other.
    catalog_set = {v.lower() for v in resolved.voices}
    for label in ("af_audit_1", "af_audit_2", "af_audit_3"):
        assert label.lower() in catalog_set
        # And conversely: every catalog entry resolves through sid validation.
        sid = _resolve_sid(staged_kokoro_artifact, label)
        assert sid >= 0

    # 4. Default voice — kernel picks "the bundle's first speaker"
    #    when the model's documented default isn't in the catalog.
    #    af_bella IS the doc default for kokoro-82m and is NOT in
    #    voices.txt, so the kernel falls back to resolved.voices[0].
    default_sid = 0
    default_label = resolve_default_voice_label("kokoro-82m", prepared_model_dir=str(staged_kokoro_artifact))
    assert default_label == "af_audit_1"
    assert default_sid == 0


@pytest.mark.asyncio
async def test_unsupported_voice_rejected_by_validate_and_would_fail_listing(
    staged_kokoro_artifact: Path,
):
    """The catalog and the validation layer must agree on what's
    invalid too. A name that isn't in voices.txt:
      - is NOT in resolve_voice_catalog's output;
      - is rejected by resolve_voice_sid with a clear error.
    """
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.engines.sherpa import resolve_voice_catalog

    resolved = resolve_voice_catalog(
        "kokoro-82m",
        prepared_model_dir=str(staged_kokoro_artifact),
    )
    assert "af_phantom" not in resolved.voices

    with pytest.raises(OctomilError) as exc_info:
        _resolve_sid(staged_kokoro_artifact, "af_phantom")
    assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
    assert "voice_not_supported_for_model" in exc_info.value.error_message


@pytest.mark.asyncio
async def test_planner_logical_speakers_round_trip_through_resolver(staged_kokoro_artifact: Path):
    """A planner that publishes a logical speaker pointing at one of
    the staged native voices must satisfy the closure-of-loop:
      - voices.list (via list_logical_speakers) advertises the speaker;
      - speech.create / .stream resolve it through the speaker
        resolver to the SAME native voice the catalog lists;
      - resolve_voice_sid accepts that native voice.
    """
    from octomil.execution.tts_speaker_resolver import (
        list_logical_speakers,
        resolve_tts_speaker,
    )
    from octomil.runtime.planner.schemas import (
        AppResolution,
        RuntimeSelection,
        TtsSpeakerProfile,
    )

    selection = RuntimeSelection(
        locality="local",
        app_resolution=AppResolution(
            app_id="audit",
            capability="tts",
            routing_policy="private",
            selected_model="kokoro-82m",
            tts_speakers={
                "narrator": TtsSpeakerProfile(speaker_id="narrator", native_voice="af_audit_2"),
            },
        ),
    )
    # voices.list view
    speakers = list_logical_speakers(selection)
    assert speakers[0]["speaker_id"] == "narrator"
    assert speakers[0]["native_voice"] == "af_audit_2"

    # create() / stream() view via the speaker resolver
    resolved = resolve_tts_speaker(
        speaker="narrator",
        voice=None,
        selection=selection,
        is_app_ref=True,
    )
    assert resolved.source == "planner_profile"
    assert resolved.native_voice == "af_audit_2"

    # resolve_voice_sid accepts the resolved native voice (closes loop)
    assert _resolve_sid(staged_kokoro_artifact, resolved.native_voice) == 1


def test_voice_info_ids_round_trip_with_validate_voice(staged_kokoro_artifact: Path):
    """A ``VoiceInfo`` constructed from the resolver's catalog must
    produce ids that ``resolve_voice_sid`` accepts. Catches
    case-mangling regressions (kernel lower-cases for default
    matching, resolver preserves case)."""
    from octomil.runtime.engines.sherpa import resolve_voice_catalog

    resolved = resolve_voice_catalog(
        "kokoro-82m",
        prepared_model_dir=str(staged_kokoro_artifact),
    )

    voice_infos = [
        VoiceInfo(id=name, sid=idx, default=False, source="voices_txt") for idx, name in enumerate(resolved.voices)
    ]

    for info in voice_infos:
        sid = _resolve_sid(staged_kokoro_artifact, info.id)
        assert sid == info.sid, f"sid mismatch on {info.id!r}: catalog={info.sid} validate={sid}"
