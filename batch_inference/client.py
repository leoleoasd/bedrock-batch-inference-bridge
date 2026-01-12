"""Main client for batch inference SDK."""

import logging
from typing import Any

import aioboto3

from .exceptions import AlreadySetupError, NotSetupError
from .models import BatchConfig
from .queue import RequestQueue
from .scheduler import BatchScheduler
from .submitter import BatchSubmitter

logger = logging.getLogger(__name__)


class BatchInferenceClient:
    """Client for batch inference with boto3-compatible invoke_model API.

    Example:
        client = BatchInferenceClient()

        await client.setup(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            s3_input_uri="s3://my-bucket/input/",
            s3_output_uri="s3://my-bucket/output/",
            role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole",
            min_batch_size=5,
            max_batch_size=50,
            batch_timeout=30.0,
        )

        response = await client.invoke_model(
            body=json.dumps({"messages": [...]}),
            contentType="application/json",
        )

        await client.close()
    """

    def __init__(self, session: aioboto3.Session | None = None):
        self._session = session or aioboto3.Session()
        self._config: BatchConfig | None = None
        self._queue: RequestQueue | None = None
        self._scheduler: BatchScheduler | None = None
        self._submitter: BatchSubmitter | None = None
        self._is_setup = False

    async def setup(
        self,
        model_id: str,
        s3_input_uri: str,
        s3_output_uri: str,
        role_arn: str,
        region: str = "us-west-2",
        min_batch_size: int = 100,
        max_batch_size: int = 1000,
        batch_timeout: float = 60.0,
        poll_interval: float = 5.0,
        job_timeout_hours: int = 24,
        job_name_prefix: str = "batch-inference",
    ) -> None:
        """Configure and start the batch inference client.

        Args:
            model_id: Bedrock model ID to use for all requests.
            s3_input_uri: S3 URI for batch input files (e.g., s3://bucket/input/).
            s3_output_uri: S3 URI for batch output files (e.g., s3://bucket/output/).
            role_arn: IAM role ARN with permissions for batch inference.
            region: AWS region (default "us-west-2").
            min_batch_size: Minimum requests before time-based submission (default 100).
            max_batch_size: Submit immediately when this many requests queued (default 1000).
            batch_timeout: Seconds to wait before submitting if >= min_batch_size (default 60).
            poll_interval: Seconds between job status polls (default 5).
            job_timeout_hours: Maximum hours for batch job (default 24).
            job_name_prefix: Prefix for batch job names (default "batch-inference").
        """
        if self._is_setup:
            raise AlreadySetupError()

        self._config = BatchConfig(
            model_id=model_id,
            s3_input_uri=s3_input_uri,
            s3_output_uri=s3_output_uri,
            role_arn=role_arn,
            region=region,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
            poll_interval=poll_interval,
            job_timeout_hours=job_timeout_hours,
            job_name_prefix=job_name_prefix,
        )

        self._submitter = BatchSubmitter(
            config=self._config,
            session=self._session,
        )

        self._queue = RequestQueue()
        self._scheduler = BatchScheduler(
            config=self._config,
            queue=self._queue,
            submitter=self._submitter,
        )

        # Wire up size change notifications
        self._queue._on_size_change = self._scheduler.notify_size_change

        # Start the scheduler
        await self._scheduler.start()

        self._is_setup = True
        logger.info(
            f"BatchInferenceClient setup complete: model={model_id}, "
            f"min={min_batch_size}, max={max_batch_size}, timeout={batch_timeout}s"
        )

    async def invoke_model(
        self,
        body: bytes | str,
        contentType: str,
        accept: str = "application/json",
        modelId: str | None = None,
        trace: str | None = None,
        guardrailIdentifier: str | None = None,
        guardrailVersion: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit an inference request (boto3-compatible signature).

        This method matches the boto3 bedrock-runtime invoke_model signature.
        The modelId parameter is accepted but ignored (model is set in setup()).

        Args:
            body: Request payload in JSON format (bytes or str).
            contentType: MIME type of input (must be application/json).
            accept: Desired response MIME type (default application/json).
            modelId: Ignored - model is configured in setup().
            trace: Ignored for batch inference.
            guardrailIdentifier: Ignored for batch inference.
            guardrailVersion: Ignored for batch inference.

        Returns:
            Response dict matching boto3 invoke_model response:
            {
                "body": StreamingBody,  # Read with .read()
                "contentType": "application/json",
            }
        """
        if not self._is_setup or self._queue is None:
            raise NotSetupError()

        # Convert str to bytes if needed
        if isinstance(body, str):
            body = body.encode("utf-8")

        # Queue the request and get a future
        future = await self._queue.put(
            body=body,
            content_type=contentType,
            accept=accept,
        )

        # Wait for the result
        return await future

    async def close(self) -> None:
        """Flush pending requests and stop the client."""
        logger.info("Closing BatchInferenceClient")
        if self._scheduler:
            await self._scheduler.stop()

        self._is_setup = False
        logger.info("BatchInferenceClient closed")

    async def __aenter__(self) -> "BatchInferenceClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
