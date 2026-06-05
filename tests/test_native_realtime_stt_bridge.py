"""Realtime STT ABI-bridge unit tests (no runtime dylib required).

These lock the Python-side mirror of the v0.1.24 realtime STT ABI
(``OCT_EVENT_TRANSCRIPT_PARTIAL`` + ``oct_session_end_input``) against
drift from ``octomil-runtime/include/octomil/runtime.h``. They exercise
only the cffi cdef and the pure-Python ``NativeEvent`` view, so they run
everywhere — unlike the live struct-size parity test, which needs a
fetched minor-12 dylib (``oct_event_size()`` round-trip) and is covered
in ``test_runtime_native_loader.py``.
"""

from __future__ import annotations

import pytest

cffi = pytest.importorskip("cffi", reason="cffi extra not installed")

from octomil.runtime.native import loader as L  # noqa: E402


def _parsed_cdef():
    """Parse the loader cdef in isolation; raises on any C syntax error."""
    ffi = cffi.FFI()
    ffi.cdef(L._CDEF)
    return ffi


def test_abi_constants_match_runtime() -> None:
    """The mirrored ABI constants must match runtime.h exactly. A bump
    here without a matching runtime release breaks load-time gating, so
    these values are intentionally pinned."""
    assert L.OCT_EVENT_TRANSCRIPT_PARTIAL == 26
    # v0.1.24 bumped EVENT_VERSION 2->3 (TRANSCRIPT_PARTIAL); v0.1.25
    # bumped 3->4 (tail-additive transcript_segment decode diagnostics).
    assert L.OCT_EVENT_VERSION == 4
    # end_input is a new exported symbol (minor 12); declaring it in the
    # cdef means an older dylib lacking the export must be rejected.
    assert L._REQUIRED_ABI_MINOR == 12


def test_cdef_parses_and_declares_end_input() -> None:
    """The full cdef must parse, and the end_input symbol must be
    present. (Regression: an unescaped ``*/`` inside the end_input doc
    comment once truncated the block comment and broke the parse.)"""
    _parsed_cdef()  # raises on any C syntax error
    assert "oct_status_t oct_session_end_input(oct_session_t* session);" in L._CDEF


def test_transcript_partial_union_arm_layout() -> None:
    """The data.transcript_partial union arm must expose every field
    from runtime.h with assignable, round-tripping values."""
    ffi = _parsed_cdef()
    ev = ffi.new("oct_event_t*")
    p = ev.data.transcript_partial
    p.n_bytes = 12
    p.revision_id = 7
    p.start_ms = 100
    p.end_ms = 940
    p.stable_prefix_bytes = 5
    p.is_stable = 1
    assert (
        p.n_bytes,
        p.revision_id,
        p.start_ms,
        p.end_ms,
        p.stable_prefix_bytes,
        p.is_stable,
    ) == (12, 7, 100, 940, 5, 1)


def test_native_event_carries_partial_fields() -> None:
    """A decoded TRANSCRIPT_PARTIAL surfaces revision-aware fields and
    puts the provisional UTF-8 in ``text`` (same slot as segment/final)
    so callers can replace stale partials without confusing them with
    the authoritative final transcript."""
    ev = L.NativeEvent(
        type=L.OCT_EVENT_TRANSCRIPT_PARTIAL,
        version=L.OCT_EVENT_VERSION,
        monotonic_ns=0,
        user_data_ptr=0,
        text="the quick brown",
        partial_revision_id=3,
        partial_start_ms=0,
        partial_end_ms=1200,
        partial_stable_prefix_bytes=9,
        partial_is_stable=True,
    )
    assert ev.type == L.OCT_EVENT_TRANSCRIPT_PARTIAL
    assert ev.text == "the quick brown"
    assert ev.partial_revision_id == 3
    assert ev.partial_end_ms == 1200
    assert ev.partial_stable_prefix_bytes == 9
    assert ev.partial_is_stable is True


def test_native_event_partial_fields_default_to_zero() -> None:
    """Non-partial events must not carry stray partial state (defaults
    keep the revision counter unambiguous: 0 == 'not a partial')."""
    ev = L.NativeEvent(
        type=L.OCT_EVENT_TRANSCRIPT_FINAL,
        version=L.OCT_EVENT_VERSION,
        monotonic_ns=0,
        user_data_ptr=0,
        text="final transcript",
    )
    assert ev.partial_revision_id == 0
    assert ev.partial_is_stable is False
    assert ev.partial_stable_prefix_bytes == 0


def test_transcript_segment_carries_v4_decode_diagnostics() -> None:
    """The data.transcript_segment arm must expose the v0.1.25
    (EVENT_VERSION 4) tail-additive ``avg_logprob`` / ``no_speech_prob``
    floats with assignable, round-tripping values."""
    ffi = _parsed_cdef()
    ev = ffi.new("oct_event_t*")
    s = ev.data.transcript_segment
    s.segment_index = 2
    s.is_final = 1
    s.avg_logprob = -0.25
    s.no_speech_prob = 0.0125
    assert s.segment_index == 2
    assert s.is_final == 1
    assert abs(s.avg_logprob - (-0.25)) < 1e-6
    assert abs(s.no_speech_prob - 0.0125) < 1e-6


def test_native_event_carries_segment_diagnostics() -> None:
    """A decoded TRANSCRIPT_SEGMENT surfaces the v4 per-segment decode
    diagnostics; non-segment events leave them at the 0.0 default."""
    seg = L.NativeEvent(
        type=L.OCT_EVENT_TRANSCRIPT_SEGMENT,
        version=L.OCT_EVENT_VERSION,
        monotonic_ns=0,
        user_data_ptr=0,
        text="konnichiwa",
        segment_index=0,
        segment_is_final=True,
        segment_avg_logprob=-0.31,
        segment_no_speech_prob=0.02,
    )
    assert abs(seg.segment_avg_logprob - (-0.31)) < 1e-6
    assert abs(seg.segment_no_speech_prob - 0.02) < 1e-6

    final = L.NativeEvent(
        type=L.OCT_EVENT_TRANSCRIPT_FINAL,
        version=L.OCT_EVENT_VERSION,
        monotonic_ns=0,
        user_data_ptr=0,
        text="final",
    )
    assert final.segment_avg_logprob == 0.0
    assert final.segment_no_speech_prob == 0.0
