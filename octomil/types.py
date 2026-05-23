"""Public type re-exports for the Octomil SDK.

Provides a convenient import path for route metadata types::

    from octomil.types import RouteMetadata, RouteExecution, RouteModel

Generated contract types (Pydantic v2 models produced by
datamodel-code-generator from the OpenAPI spec in octomil-contracts) are
re-exported here as stable aliases::

    from octomil.types import DesiredState
    from octomil.types import ObservedState
    from octomil.types import DeviceSyncRequest, DeviceSyncResponse
    from octomil.types import TelemetryBatch, TelemetryEvent
    from octomil.types import ChatTurnRequest, ChatTurnResult
    from octomil.types import AudioSpeechRequest, AudioSpeechResult

These types reflect the wire-format shapes defined in octomil-contracts and
are always imported from the generated layer — never hand-edited here.

Note: SDK method return types (e.g. ``OctomilControl.fetch_desired_state``,
``OctomilControl.sync``) currently return ``dict[str, Any]`` for backwards
compatibility.  These re-exports allow callers to type-annotate against the
structured Pydantic models and construct request payloads via the canonical
type rather than raw dicts.  The raw-dict facade signatures are intentionally
preserved — do not tighten them here.
"""

from __future__ import annotations

from octomil._generated.types import (
    AudioSpeechRequest,
    AudioSpeechResult,
    ChatTurnRequest,
    ChatTurnResult,
    DesiredState,
    DeviceSyncRequest,
    DeviceSyncResponse,
    ObservedState,
    TelemetryBatch,
)
from octomil._generated.types import (
    Event as TelemetryEvent,
)
from octomil.execution.route_metadata_mapper import (
    ArtifactCache,
    FallbackInfo,
    PlannerInfo,
    RouteArtifact,
    RouteExecution,
    RouteMetadata,
    RouteModel,
    RouteModelRequested,
    RouteModelResolved,
    RouteReason,
)

__all__ = [
    # Route metadata types
    "ArtifactCache",
    "FallbackInfo",
    "PlannerInfo",
    "RouteArtifact",
    "RouteExecution",
    "RouteMetadata",
    "RouteModel",
    "RouteModelRequested",
    "RouteModelResolved",
    "RouteReason",
    # Generated contract types (transport layer)
    # Device state protocol
    "DesiredState",
    "DeviceSyncRequest",
    "DeviceSyncResponse",
    "ObservedState",
    # Telemetry wire types
    "TelemetryBatch",
    "TelemetryEvent",
    # Chat API wire types
    "ChatTurnRequest",
    "ChatTurnResult",
    # Audio / speech wire types
    "AudioSpeechRequest",
    "AudioSpeechResult",
]
