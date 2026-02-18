"""
EdgeML Python SDK.

Serve, deploy, and observe ML models on edge devices.

Primary SDK code lives in `edgeml/python/edgeml`.
"""

from .client import Client
from .python.edgeml import (
    EdgeML,
    EdgeMLClientError,
    ExperimentsAPI,
    FederatedAnalyticsAPI,
    Federation,
    FederatedClient,
    ModelRegistry,
    RolloutsAPI,
    DeviceAuthClient,
    compute_state_dict_delta,
    apply_filters,
    DataKind,
    DeltaFilter,
    FilterRegistry,
    FilterResult,
    ECKeyPair,
    SecAggClient,
    SecAggConfig,
    SecAggPlusClient,
    SecAggPlusConfig,
    SECAGG_PLUS_MOD_RANGE,
    HKDF_INFO_PAIRWISE_MASK,
    HKDF_INFO_SHARE_ENCRYPTION,
    HKDF_INFO_SELF_MASK,
)

__all__ = [
    "Client",
    "EdgeML",
    "EdgeMLClientError",
    "Federation",
    "FederatedClient",
    "ModelRegistry",
    "RolloutsAPI",
    "ExperimentsAPI",
    "FederatedAnalyticsAPI",
    "compute_state_dict_delta",
    "apply_filters",
    "DeviceAuthClient",
    "DataKind",
    "DeltaFilter",
    "FilterRegistry",
    "FilterResult",
    "ECKeyPair",
    "SecAggClient",
    "SecAggConfig",
    "SecAggPlusClient",
    "SecAggPlusConfig",
    "SECAGG_PLUS_MOD_RANGE",
    "HKDF_INFO_PAIRWISE_MASK",
    "HKDF_INFO_SHARE_ENCRYPTION",
    "HKDF_INFO_SELF_MASK",
]
