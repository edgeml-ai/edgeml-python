"""AudioTranscriptions — speech-to-text API."""

from __future__ import annotations

from typing import Callable, Optional

from octomil._generated.message_role import MessageRole
from octomil._generated.model_capability import ModelCapability
from octomil.audio.types import ChunkDiagnostics, TranscriptionResult, TranscriptionSegment
from octomil.model_ref import ModelRef, ModelRefFactory
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    GenerationConfig,
    RuntimeContentPart,
    RuntimeMessage,
    RuntimeRequest,
    RuntimeResponse,
    SttOptions,
)


class AudioTranscriptions:
    """Audio transcription API.

    Wraps the underlying audio runtime to provide speech-to-text.

    Usage::

        result = await client.audio.transcriptions.create(
            audio=audio_bytes
        )
        print(result.text)
    """

    def __init__(
        self,
        runtime_resolver: Callable[[ModelRef], Optional[ModelRuntime]],
    ) -> None:
        self._runtime_resolver = runtime_resolver

    async def create(
        self,
        audio: bytes,
        *,
        model: Optional[ModelRef] = None,
        language: Optional[str] = None,
        response_format: Optional[str] = None,
        chunk_window_ms: Optional[int] = None,
        chunk_overlap_ms: Optional[int] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Raw audio data (WAV, MP3, etc.).
            model: Model reference. Defaults to transcription capability.
            language: Optional language hint (BCP 47 code, e.g. "en").
            response_format: Optional output format hint.
            chunk_window_ms: Optional fixed decode-window size for chunked
                transcription (native path only). ``None`` (default) runs
                a single full-buffer decode — byte-identical to the
                pre-v0.1.27 behaviour.
            chunk_overlap_ms: Optional overlap between consecutive decode
                windows. Ignored unless ``chunk_window_ms`` is set.

        Returns:
            TranscriptionResult with the transcribed text. On the native
            path the result also carries ``segments`` (with per-segment
            ``avg_logprob`` / ``no_speech_prob``), ``duration_ms``, and —
            when ``chunk_window_ms`` is set — ``chunk_diagnostics``.
        """
        ref = model or ModelRefFactory.capability(ModelCapability.TRANSCRIPTION)
        runtime = self._runtime_resolver(ref)
        if runtime is None:
            raise RuntimeError("No runtime available for transcription model")

        parts = [RuntimeContentPart.audio_part(audio, "audio/wav")]
        if language:
            parts.append(RuntimeContentPart.text_part(language))
        stt_options: Optional[SttOptions] = None
        if chunk_window_ms is not None or chunk_overlap_ms is not None:
            stt_options = SttOptions(
                chunk_window_ms=chunk_window_ms,
                chunk_overlap_ms=chunk_overlap_ms,
            )
        request = RuntimeRequest(
            messages=[RuntimeMessage(role=MessageRole.USER, parts=parts)],
            generation_config=GenerationConfig(max_tokens=0, temperature=0.0),
            stt_options=stt_options,
        )
        response = await runtime.run(request)
        return self._project_result(response, language)

    @staticmethod
    def _project_result(response: RuntimeResponse, language: Optional[str]) -> TranscriptionResult:
        """Project a ``RuntimeResponse`` onto the public result.

        Segments and chunk diagnostics are only populated on the native
        path; legacy / cloud runtimes leave the carriers as ``None`` so
        the public result keeps its empty-segments / ``None``-diagnostics
        defaults without raising.
        """
        segments: list[TranscriptionSegment] = list(response.stt_segments or [])
        diagnostics = response.stt_chunk_diagnostics
        if diagnostics is not None and not isinstance(diagnostics, ChunkDiagnostics):
            # Defensive: a runtime that hands back a None / unexpected
            # shape must not poison the public result. Only project a
            # real ``ChunkDiagnostics``; anything else is treated as
            # "no diagnostics" rather than surfaced verbatim.
            diagnostics = None
        return TranscriptionResult(
            text=response.text,
            language=language,
            segments=segments,
            duration_ms=int(response.stt_duration_ms or 0),
            chunk_diagnostics=diagnostics,
        )

    async def stream(
        self,
        audio: bytes,
        *,
        model: Optional[ModelRef] = None,
    ) -> list[TranscriptionSegment]:
        """Stream transcription segments.

        Args:
            audio: Raw audio data.
            model: Model reference. Defaults to transcription capability.

        Returns:
            List of transcription segments.
        """
        ref = model or ModelRefFactory.capability(ModelCapability.TRANSCRIPTION)
        runtime = self._runtime_resolver(ref)
        if runtime is None:
            raise RuntimeError("No runtime available for transcription model")

        request = RuntimeRequest(
            messages=[
                RuntimeMessage(
                    role=MessageRole.USER,
                    parts=[RuntimeContentPart.audio_part(audio, "audio/wav")],
                )
            ],
            generation_config=GenerationConfig(max_tokens=0, temperature=0.0),
        )
        segments: list[TranscriptionSegment] = []
        async for chunk in runtime.stream(request):
            if chunk.text:
                segments.append(TranscriptionSegment(text=chunk.text))
        return segments
