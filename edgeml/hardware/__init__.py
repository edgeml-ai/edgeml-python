"""Hardware detection subsystem for edgeml."""

from __future__ import annotations

from ._base import GPUBackend, GPUBackendRegistry, get_gpu_registry, reset_gpu_registry
from ._types import CPUInfo, GPUDetectionResult, GPUInfo, GPUMemory, HardwareProfile
from ._unified import UnifiedDetector, detect_hardware

__all__ = [
    "CPUInfo",
    "GPUDetectionResult",
    "GPUInfo",
    "GPUMemory",
    "HardwareProfile",
    "GPUBackend",
    "GPUBackendRegistry",
    "get_gpu_registry",
    "reset_gpu_registry",
    "UnifiedDetector",
    "detect_hardware",
]
