"""S3 operations for batch inference."""

import json
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import aioboto3

from .models import BatchRequest

logger = logging.getLogger(__name__)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse S3 URI into bucket and key."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


class S3Manager:
    """Manages S3 operations for batch inference."""

    def __init__(self, session: aioboto3.Session | None = None, region: str = "us-west-2"):
        self._session = session or aioboto3.Session()
        self._region = region

    async def upload_batch_input(
        self,
        requests: list[BatchRequest],
        s3_input_uri: str,
        job_id: str,
    ) -> str:
        """Upload batch input JSONL to S3.

        Returns the full S3 URI of the uploaded file.
        """
        bucket, base_key = parse_s3_uri(s3_input_uri)

        # Generate unique filename with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"batch_{job_id}_{timestamp}.jsonl"

        # Ensure base_key ends with /
        if base_key and not base_key.endswith("/"):
            base_key += "/"

        key = f"{base_key}{filename}"

        # Build JSONL content
        lines = []
        for request in requests:
            record = request.to_jsonl_record()
            lines.append(json.dumps(record, separators=(",", ":")))

        content = "\n".join(lines)

        async with self._session.client("s3", region_name=self._region) as s3:
            await s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=content.encode("utf-8"),
                ContentType="application/jsonl",
            )

        s3_uri = f"s3://{bucket}/{key}"
        logger.debug(f"Uploaded {len(requests)} records to {s3_uri}")
        return s3_uri

    async def download_batch_output(
        self,
        s3_output_uri: str,
        job_arn: str,
    ) -> dict[str, Any]:
        """Download and parse batch output from S3.

        Args:
            s3_output_uri: Base S3 URI for output files
            job_arn: Full Bedrock job ARN (e.g., arn:aws:bedrock:region:account:model-invocation-job/xyz123)

        Returns a dict mapping record_id to model output.
        """
        bucket, base_key = parse_s3_uri(s3_output_uri)

        # Extract Bedrock job ID from ARN (last part after /)
        # ARN format: arn:aws:bedrock:region:account:model-invocation-job/xyz123
        bedrock_job_id = job_arn.split("/")[-1]

        # Bedrock output structure: s3_output_uri/<bedrock-job-id>/input_filename.jsonl.out
        async with self._session.client("s3", region_name=self._region) as s3:
            prefix = base_key.rstrip("/") + "/" + bedrock_job_id + "/"

            logger.info(f"Listing objects in s3://{bucket}/{prefix}")
            response = await s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
            )

            results: dict[str, Any] = {}
            files_processed = 0

            for obj in response.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".jsonl.out"):
                    # Download and parse the output file
                    logger.debug(f"Downloading output file: s3://{bucket}/{key}")
                    get_response = await s3.get_object(Bucket=bucket, Key=key)
                    body = await get_response["Body"].read()
                    content = body.decode("utf-8")

                    for line in content.strip().split("\n"):
                        if line:
                            record = json.loads(line)
                            record_id = record.get("recordId")
                            if record_id:
                                results[record_id] = record

                    files_processed += 1

            logger.debug(f"Downloaded {files_processed} output files, {len(results)} records")
            return results
