"""Public type re-exports for the Octomil SDK.

Provides a convenient import path for route metadata types::

    from octomil.types import RouteMetadata, RouteExecution, RouteModel

Generated contract types (Pydantic v2 models produced by
datamodel-code-generator from the OpenAPI spec in octomil-contracts) are
re-exported here as stable aliases::

    from octomil.types import DesiredState
"""

from __future__ import annotations

from octomil._generated.types import DesiredState
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
    "DesiredState",
]
