"""Inference backend selection."""
from __future__ import annotations

from .service import Click, InferenceService, InferResult

_INSTANCE: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """Return the process-wide InferenceService, built lazily per config."""
    global _INSTANCE
    if _INSTANCE is None:
        from .. import config

        backend = config.INFERENCE_BACKEND.lower()
        if backend == "mock":
            from .mock import MockInferenceService

            _INSTANCE = MockInferenceService()
        elif backend == "ptv3":
            # Real model path — enabled once basic functionality is stable.
            from .ptv3_local import Ptv3InferenceService

            _INSTANCE = Ptv3InferenceService()
        else:
            raise ValueError(f"Unknown INFERENCE_BACKEND '{backend}'")
    return _INSTANCE


__all__ = ["Click", "InferenceService", "InferResult", "get_inference_service"]
