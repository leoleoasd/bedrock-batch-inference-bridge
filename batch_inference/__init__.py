"""Bedrock batch inference SDK with boto3-compatible API."""

from .client import BatchInferenceClient
from .exceptions import (
    AlreadySetupError,
    BatchInferenceError,
    BatchJobError,
    InvocationError,
    NotSetupError,
)
from .models import BatchConfig, StreamingBody

__all__ = [
    "BatchInferenceClient",
    "BatchConfig",
    "StreamingBody",
    "BatchInferenceError",
    "BatchJobError",
    "InvocationError",
    "NotSetupError",
    "AlreadySetupError",
]

__version__ = "0.1.0"
