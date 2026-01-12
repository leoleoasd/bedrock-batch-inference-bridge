# Batch Inference SDK

A Python SDK that provides a boto3-compatible `invoke_model` API while transparently batching requests and using AWS Bedrock batch inference under the hood.

## Features

- **boto3-compatible API**: Drop-in replacement for `bedrock-runtime.invoke_model()`
- **Automatic batching**: Requests are batched based on configurable size and timeout thresholds
- **Parallel job execution**: Multiple batch jobs can run concurrently
- **Async-first design**: Built on asyncio for high concurrency

## Installation

```bash
pip install batch-inference
```

Or with uv:

```bash
uv add batch-inference
```

## Quick Start

```python
import asyncio
import json
from batch_inference import BatchInferenceClient

async def main():
    client = BatchInferenceClient()

    await client.setup(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        s3_input_uri="s3://my-bucket/batch-input/",
        s3_output_uri="s3://my-bucket/batch-output/",
        role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole",
        region="us-west-2",
        min_batch_size=100,
        max_batch_size=1000,
        batch_timeout=60.0,
    )

    # Use exactly like boto3's invoke_model
    response = await client.invoke_model(
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello!"}]
        }),
        contentType="application/json",
    )

    result = json.loads(response["body"].read())
    print(result["content"][0]["text"])

    await client.close()

asyncio.run(main())
```

## Concurrent Requests

The SDK automatically batches concurrent requests:

```python
import asyncio
import json
from batch_inference import BatchInferenceClient

async def main():
    client = BatchInferenceClient()

    await client.setup(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        s3_input_uri="s3://my-bucket/input/",
        s3_output_uri="s3://my-bucket/output/",
        role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole",
        max_batch_size=100,
        batch_timeout=30.0,
    )

    async def ask(question: str):
        response = await client.invoke_model(
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": question}]
            }),
            contentType="application/json",
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    # Submit 500 requests concurrently - they get batched automatically
    questions = [f"What is {i} + {i}?" for i in range(500)]
    tasks = [ask(q) for q in questions]
    answers = await asyncio.gather(*tasks)

    for q, a in zip(questions[:5], answers[:5]):
        print(f"Q: {q}\nA: {a}\n")

    await client.close()

asyncio.run(main())
```

## Configuration

### `setup()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | str | required | Bedrock model ID or inference profile ARN |
| `s3_input_uri` | str | required | S3 URI for batch input files (e.g., `s3://bucket/input/`) |
| `s3_output_uri` | str | required | S3 URI for batch output files (e.g., `s3://bucket/output/`) |
| `role_arn` | str | required | IAM role ARN with Bedrock and S3 permissions |
| `region` | str | `us-west-2` | AWS region |
| `min_batch_size` | int | `100` | Minimum requests before time-based submission |
| `max_batch_size` | int | `1000` | Submit immediately when this many requests queued |
| `batch_timeout` | float | `60.0` | Seconds to wait before submitting pending requests |
| `poll_interval` | float | `5.0` | Seconds between job status polls |
| `job_timeout_hours` | int | `24` | Maximum hours for batch job |
| `job_name_prefix` | str | `batch-inference` | Prefix for batch job names |

### Batching Behavior

- When `max_batch_size` requests are queued, a batch is submitted immediately
- When `batch_timeout` elapses with pending requests, they are submitted (padded to 100 if needed, as Bedrock requires minimum 100 records)
- Multiple batches can run in parallel

## `invoke_model()` Parameters

The method signature matches boto3's `bedrock-runtime.invoke_model()`:

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `body` | bytes/str | Yes | JSON payload with prompt and inference params |
| `contentType` | str | Yes | Must be `application/json` |
| `accept` | str | No | Response MIME type (default `application/json`) |
| `modelId` | str | No | Ignored - model is set in `setup()` |

### Response Structure

```python
{
    "body": StreamingBody,      # Read with .read()
    "contentType": "application/json",
}
```

## IAM Permissions

The IAM role needs permissions for:

- `bedrock:CreateModelInvocationJob`
- `bedrock:GetModelInvocationJob`
- `s3:PutObject` on input bucket
- `s3:GetObject`, `s3:ListBucket` on output bucket

Example policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateModelInvocationJob",
                "bedrock:GetModelInvocationJob"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": ["s3:PutObject"],
            "Resource": "arn:aws:s3:::my-bucket/input/*"
        },
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": [
                "arn:aws:s3:::my-bucket",
                "arn:aws:s3:::my-bucket/output/*"
            ]
        }
    ]
}
```

## Running Tests

```bash
# Copy and configure .env
cp .env.example .env
# Edit .env with your AWS config

# Install dev dependencies
uv sync --extra dev

# Run integration test
uv run python test_integration.py
```

## License

MIT
