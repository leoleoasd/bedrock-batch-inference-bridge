"""Integration test for batch inference SDK using real AWS resources."""

import asyncio
import json
import logging
import os
import random

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from batch_inference import BatchInferenceClient

# Load config from .env
load_dotenv()

REGION = os.getenv("AWS_REGION", "us-west-2")
S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = os.getenv("S3_PREFIX", "batch_inference_sdk_test")
MODEL_ID = os.environ["MODEL_ID"]
ROLE_ARN = os.environ["ROLE_ARN"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


async def test_real_batch_inference():
    """Test with real AWS Bedrock batch inference."""
    print("=" * 60)
    print("Integration Test: Real AWS Bedrock Batch Inference")
    print("=" * 60)
    print(f"Region:    {REGION}")
    print(f"Bucket:    {S3_BUCKET}")
    print(f"Model:     {MODEL_ID}")
    print(f"Role:      {ROLE_ARN}")
    print("=" * 60)
    print()

    client = BatchInferenceClient()

    await client.setup(
        model_id=MODEL_ID,
        s3_input_uri=f"s3://{S3_BUCKET}/{S3_PREFIX}/input/",
        s3_output_uri=f"s3://{S3_BUCKET}/{S3_PREFIX}/output/",
        role_arn=ROLE_ARN,
        region=REGION,
        min_batch_size=100,   # Bedrock minimum is 100
        max_batch_size=120,
        batch_timeout=10.0,
        poll_interval=30.0,
        job_name_prefix="sdk-test",
    )

    # Bedrock requires minimum 100 records per batch
    num_requests = 150
    print(f"Submitting {num_requests} requests...")
    async def task_wrapper(i):
        await asyncio.sleep(random.random() * 0.1)
        return await client.invoke_model(
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": f"Reply with only the number {i}"}
                ]
            }),
            contentType="application/json",
        )
    tasks = [task_wrapper(i) for i in range(num_requests)]

    print("Waiting for batch to complete...")
    responses = await tqdm.gather(*tasks)

    print()
    print("=" * 60)
    print(f"Results: {len(responses)} responses received")
    print("=" * 60)

    # Show first 5 samples
    for i, response in enumerate(responses[:5]):
        body = response["body"].read()
        result = json.loads(body)
        content = result.get("content", [])
        text = ""
        for block in content:
            if block.get("type") == "text":
                text = block.get("text", "")
                break
        print(f"Response {i}: {text[:100]}")

    if len(responses) > 5:
        print(f"... and {len(responses) - 5} more")

    await client.close()

    print()
    print("Integration test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_real_batch_inference())
