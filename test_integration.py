"""Integration test for batch inference SDK using real AWS resources."""

import argparse
import asyncio
import json
import logging
import os
import random

import aioboto3
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from batch_inference import BatchInferenceClient

# Load config from .env
load_dotenv()

REGION = os.getenv("AWS_REGION", "us-west-2")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "batch_inference_sdk_test")
MODEL_ID = os.getenv("MODEL_ID", "")
ROLE_ARN = os.getenv("ROLE_ARN", "")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


async def test_real_batch_inference(
    debug_mode: bool = False,
    litellm_mode: bool = False,
    litellm_model: str | None = None,
    litellm_api_base: str | None = None,
):
    """Test with real AWS Bedrock batch inference.
    
    Args:
        debug_mode: If True, use direct invoke_model calls instead of batch inference.
        litellm_mode: If True, use litellm acompletion instead of batch inference.
        litellm_model: Model name for litellm (overrides LITELLM_MODEL env var).
        litellm_api_base: API base URL for litellm (overrides LITELLM_API_BASE env var).
    """
    # Verify AWS identity using aioboto3 (skip for litellm mode)
    session = aioboto3.Session()
    if not litellm_mode:
        print("Verifying AWS identity...")
        async with session.client("sts", region_name=REGION) as sts:
            identity = await sts.get_caller_identity()
            print(f"Account: {identity['Account']}")
            print(f"Arn: {identity['Arn']}")
            print()

    # Determine litellm model and api_base to use
    llm_model = litellm_model or LITELLM_MODEL
    llm_api_base = litellm_api_base or LITELLM_API_BASE

    print("=" * 60)
    if litellm_mode:
        print("Integration Test: LITELLM MODE (litellm acompletion)")
    elif debug_mode:
        print("Integration Test: DEBUG MODE (Direct invoke_model)")
    else:
        print("Integration Test: Real AWS Bedrock Batch Inference")
    print("=" * 60)
    print(f"Region:    {REGION}")
    if litellm_mode:
        print(f"LiteLLM:   {llm_model}")
        if llm_api_base:
            print(f"API Base:  {llm_api_base}")
    else:
        print(f"Model:     {MODEL_ID}")
    if not debug_mode and not litellm_mode:
        print(f"Bucket:    {S3_BUCKET}")
        print(f"Role:      {ROLE_ARN}")
    print(f"Debug:     {debug_mode}")
    print(f"LiteLLM:   {litellm_mode}")
    print("=" * 60)
    print()

    client = BatchInferenceClient(session=session)

    if litellm_mode:
        # LiteLLM mode: only litellm_model required
        if not llm_model:
            raise ValueError(
                "litellm_model is required. Set LITELLM_MODEL env var or use --litellm-model"
            )
        await client.setup(
            model_id=MODEL_ID or "not-used",
            litellm_mode=True,
            litellm_model=llm_model,
            litellm_api_base=llm_api_base,
        )
        num_requests = 5  # Fewer requests in litellm mode
    elif debug_mode:
        # Debug mode: only model_id and region required
        await client.setup(
            model_id=MODEL_ID,
            region=REGION,
            debug_mode=True,
        )
        num_requests = 5  # Fewer requests in debug mode
    else:
        # Batch mode: requires S3 and role
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
        num_requests = 150  # Bedrock requires minimum 100 records per batch

    print(f"Submitting {num_requests} requests...")

    async def task_wrapper(i):
        if not debug_mode and not litellm_mode:
            await asyncio.sleep(random.random() * 0.1)
        return await client.invoke_model(
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "messages": [
                    {"role": "user", "content": f"Reply with only the number {i}"}
                ]
            }),
            contentType="application/json",
        )

    tasks = [task_wrapper(i) for i in range(num_requests)]

    print("Waiting for responses...")
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
    parser = argparse.ArgumentParser(description="Integration test for batch inference SDK")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug mode (direct invoke_model instead of batch inference)",
    )
    parser.add_argument(
        "--litellm",
        action="store_true",
        help="Use litellm mode (litellm acompletion instead of batch inference)",
    )
    parser.add_argument(
        "--litellm-model",
        type=str,
        default=None,
        help="Model name for litellm (e.g., 'gpt-4o', 'anthropic/claude-3-opus')",
    )
    parser.add_argument(
        "--litellm-api-base",
        type=str,
        default=None,
        help="API base URL for litellm (e.g., 'http://127.0.0.1:30000/v1')",
    )
    args = parser.parse_args()

    asyncio.run(test_real_batch_inference(
        debug_mode=args.debug,
        litellm_mode=args.litellm,
        litellm_model=args.litellm_model,
        litellm_api_base=args.litellm_api_base,
    ))
