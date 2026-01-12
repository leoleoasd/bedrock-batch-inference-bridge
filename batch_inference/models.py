"""Data models for batch inference SDK."""

import asyncio
import io
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchRequest:
    """Represents a single request in a batch."""

    record_id: str
    body: bytes
    content_type: str
    accept: str
    future: asyncio.Future = field(repr=False)

    def to_jsonl_record(self) -> dict[str, Any]:
        """Convert to JSONL record format for Bedrock batch inference."""
        import json

        return {
            "recordId": self.record_id,
            "modelInput": json.loads(self.body.decode("utf-8")),
        }


class StreamingBody:
    """Mimics boto3's StreamingBody for response compatibility."""

    def __init__(self, data: bytes):
        self._data = data
        self._stream = io.BytesIO(data)

    def read(self, amt: int | None = None) -> bytes:
        """Read data from the stream."""
        if amt is None:
            return self._stream.read()
        return self._stream.read(amt)

    def close(self) -> None:
        """Close the stream."""
        self._stream.close()

    def __enter__(self) -> "StreamingBody":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


@dataclass
class BatchResponse:
    """Response matching boto3 invoke_model response structure."""

    body: StreamingBody
    content_type: str = "application/json"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict matching boto3 response format."""
        return {
            "body": self.body,
            "contentType": self.content_type,
        }


@dataclass
class BatchConfig:
    """Configuration for batch inference."""

    model_id: str
    s3_input_uri: str
    s3_output_uri: str
    role_arn: str
    region: str = "us-west-2"
    min_batch_size: int = 100  # Bedrock minimum is 100
    max_batch_size: int = 1000
    batch_timeout: float = 60.0
    poll_interval: float = 5.0
    job_timeout_hours: int = 24
    job_name_prefix: str = "batch-inference"
    bedrock_min_records: int = 100  # Bedrock's hard minimum, pad with dummy if needed
