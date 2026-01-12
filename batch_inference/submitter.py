"""Batch job submission and lifecycle management."""

import asyncio
import json
import logging
import uuid

import aioboto3

from .exceptions import BatchJobError, InvocationError
from .models import BatchConfig, BatchRequest, StreamingBody
from .s3 import S3Manager

logger = logging.getLogger(__name__)


class BatchSubmitter:
    """Handles submission and monitoring of Bedrock batch inference jobs."""

    def __init__(
        self,
        config: BatchConfig,
        session: aioboto3.Session | None = None,
    ):
        self._config = config
        self._session = session or aioboto3.Session()
        self._s3_manager = S3Manager(session=self._session, region=config.region)

    async def submit_batch(self, requests: list[BatchRequest]) -> None:
        """Submit a batch of requests and resolve their futures when complete."""
        if not requests:
            return

        job_id = str(uuid.uuid4())[:8]
        real_request_count = len(requests)

        # Pad with dummy requests if below Bedrock's minimum
        dummy_count = 0
        if len(requests) < self._config.bedrock_min_records:
            dummy_count = self._config.bedrock_min_records - len(requests)
            dummy_requests = self._create_dummy_requests(dummy_count)
            requests = requests + dummy_requests
            logger.info(
                f"Padded batch with {dummy_count} dummy requests "
                f"({real_request_count} real + {dummy_count} dummy = {len(requests)} total)"
            )

        logger.info(f"Submitting batch job_id={job_id} with {len(requests)} requests ({real_request_count} real)")

        try:
            # Upload input to S3
            input_uri = await self._s3_manager.upload_batch_input(
                requests=requests,
                s3_input_uri=self._config.s3_input_uri,
                job_id=job_id,
            )
            logger.debug(f"Uploaded batch input to {input_uri}")

            # Create batch job
            job_arn = await self._create_job(input_uri, job_id)
            logger.info(f"Created batch job: {job_arn}")

            # Poll for completion
            final_status = await self._poll_job(job_arn)
            logger.info(f"Batch job {job_arn} completed with status: {final_status}")

            if final_status != "Completed":
                error = BatchJobError(job_arn, final_status)
                logger.error(f"Batch job failed: {error}")
                for request in requests:
                    if request.future is not None and not request.future.done():
                        request.future.set_exception(error)
                return

            # Download and parse results
            results = await self._s3_manager.download_batch_output(
                s3_output_uri=self._config.s3_output_uri,
                job_arn=job_arn,
            )
            logger.info(f"Downloaded {len(results)} results from S3")

            # Resolve futures
            self._resolve_futures(requests, results)
            logger.info(f"Batch job_id={job_id} completed successfully")

        except Exception as e:
            logger.exception(f"Batch job_id={job_id} failed with exception: {e}")
            # On any error, reject all pending futures (skip dummies)
            for request in requests:
                if request.future is not None and not request.future.done():
                    request.future.set_exception(e)

    def _create_dummy_requests(self, count: int) -> list[BatchRequest]:
        """Create dummy requests to pad batch to Bedrock's minimum."""
        dummy_requests = []
        dummy_body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "x"}]
        }).encode("utf-8")

        for i in range(count):
            # Use special prefix for dummy record IDs
            record_id = f"__dummy__{uuid.uuid4()}"
            # Dummy requests don't have futures - they're just padding
            dummy_requests.append(BatchRequest(
                record_id=record_id,
                body=dummy_body,
                content_type="application/json",
                accept="application/json",
                future=None,  # type: ignore
            ))

        return dummy_requests

    async def _create_job(self, input_uri: str, job_id: str) -> str:
        """Create a batch inference job and return the job ARN."""
        async with self._session.client("bedrock", region_name=self._config.region) as bedrock:
            response = await bedrock.create_model_invocation_job(
                jobName=f"{self._config.job_name_prefix}-{job_id}",
                roleArn=self._config.role_arn,
                modelId=self._config.model_id,
                inputDataConfig={
                    "s3InputDataConfig": {
                        "s3Uri": input_uri,
                    }
                },
                outputDataConfig={
                    "s3OutputDataConfig": {
                        "s3Uri": self._config.s3_output_uri,
                    }
                },
                timeoutDurationInHours=self._config.job_timeout_hours,
            )
            return response["jobArn"]

    async def _poll_job(self, job_arn: str) -> str:
        """Poll job status until completion. Returns final status."""
        terminal_statuses = {"Completed", "Failed", "Stopped", "Expired"}
        poll_count = 0
        last_status = None

        logger.info(f"Polling job {job_arn} (interval={self._config.poll_interval}s)")

        async with self._session.client("bedrock", region_name=self._config.region) as bedrock:
            while True:
                response = await bedrock.get_model_invocation_job(
                    jobIdentifier=job_arn
                )
                status = response["status"]
                poll_count += 1

                # Log status changes at INFO level, same status at DEBUG
                if status != last_status:
                    logger.info(f"Job status: {status} (poll #{poll_count})")
                    last_status = status
                else:
                    logger.debug(f"Poll #{poll_count}: status={status}")

                if status in terminal_statuses:
                    logger.info(f"Job finished with status: {status} after {poll_count} polls")
                    return status

                await asyncio.sleep(self._config.poll_interval)

    def _resolve_futures(
        self,
        requests: list[BatchRequest],
        results: dict,
    ) -> None:
        """Match results to requests and resolve futures."""
        success_count = 0
        error_count = 0

        for request in requests:
            # Skip dummy requests (no future)
            if request.future is None or request.record_id.startswith("__dummy__"):
                continue

            if request.future.done():
                continue

            record_id = request.record_id
            result = results.get(record_id)

            if result is None:
                # No result found for this record
                logger.warning(f"No result found for record_id={record_id}")
                request.future.set_exception(
                    InvocationError(
                        record_id=record_id,
                        error_code="ResultNotFound",
                        error_message="No result found in batch output",
                    )
                )
                error_count += 1
                continue

            # Check for error in result
            if "error" in result:
                error = result["error"]
                logger.warning(f"Invocation error for record_id={record_id}: {error}")
                request.future.set_exception(
                    InvocationError(
                        record_id=record_id,
                        error_code=error.get("errorCode", "Unknown"),
                        error_message=error.get("errorMessage", "Unknown error"),
                    )
                )
                error_count += 1
                continue

            # Build response matching boto3 format
            model_output = result.get("modelOutput", {})
            body_bytes = json.dumps(model_output).encode("utf-8")

            response = {
                "body": StreamingBody(body_bytes),
                "contentType": "application/json",
            }

            request.future.set_result(response)
            success_count += 1

        logger.debug(f"Resolved futures: {success_count} success, {error_count} errors")
