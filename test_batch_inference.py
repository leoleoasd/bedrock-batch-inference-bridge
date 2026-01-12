"""Test script for batch inference SDK."""

import asyncio
import json
import logging
from unittest.mock import MagicMock

from batch_inference import BatchInferenceClient

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


async def test_batching_logic():
    """Test that requests are batched correctly."""
    print("Testing batching logic...")

    submitted_batches: list[list] = []

    client = BatchInferenceClient()

    await client.setup(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        s3_input_uri="s3://test-bucket/input/",
        s3_output_uri="s3://test-bucket/output/",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        min_batch_size=2,
        max_batch_size=5,
        batch_timeout=1.0,
    )

    # Replace submit_batch with mock after setup
    async def mock_submit_batch(requests):
        submitted_batches.append(requests)
        for req in requests:
            response = {
                "body": MagicMock(read=lambda: json.dumps({
                    "content": [{"type": "text", "text": "Response"}]
                }).encode()),
                "contentType": "application/json",
            }
            req.future.set_result(response)

    client._submitter.submit_batch = mock_submit_batch

    # Submit 3 requests (should trigger after timeout since >= min_batch_size)
    tasks = [
        client.invoke_model(
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": f"Test {i}"}]
            }),
            contentType="application/json",
        )
        for i in range(3)
    ]

    responses = await asyncio.gather(*tasks)

    assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}"
    assert len(submitted_batches) == 1, f"Expected 1 batch, got {len(submitted_batches)}"
    print(f"  Received {len(responses)} responses in {len(submitted_batches)} batch(es)")

    await client.close()
    print("  PASSED: Batching logic works correctly")


async def test_max_batch_trigger():
    """Test that max_batch_size triggers immediate submission."""
    print("Testing max batch trigger...")

    submit_count = 0

    client = BatchInferenceClient()

    await client.setup(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        s3_input_uri="s3://test-bucket/input/",
        s3_output_uri="s3://test-bucket/output/",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        min_batch_size=2,
        max_batch_size=5,
        batch_timeout=60.0,  # Long timeout - should not trigger
    )

    async def mock_submit_batch(requests):
        nonlocal submit_count
        submit_count += 1
        for req in requests:
            response = {
                "body": MagicMock(read=lambda: b'{"content": "ok"}'),
                "contentType": "application/json",
            }
            req.future.set_result(response)

    client._submitter.submit_batch = mock_submit_batch

    # Submit exactly max_batch_size requests
    tasks = [
        client.invoke_model(
            body=json.dumps({"messages": [{"role": "user", "content": f"Test {i}"}]}),
            contentType="application/json",
        )
        for i in range(5)
    ]

    responses = await asyncio.gather(*tasks)

    assert len(responses) == 5, f"Expected 5 responses, got {len(responses)}"
    assert submit_count >= 1, "Batch should have been submitted"
    print(f"  Submitted {submit_count} batch(es)")

    await client.close()
    print("  PASSED: Max batch trigger works")


async def test_request_queue():
    """Test RequestQueue directly."""
    print("Testing RequestQueue...")

    from batch_inference.queue import RequestQueue

    size_changes: list[int] = []

    def on_size_change(size: int):
        size_changes.append(size)

    queue = RequestQueue(on_size_change=on_size_change)

    # Add requests
    futures = []
    for i in range(3):
        future = await queue.put(
            body=f"test {i}".encode(),
            content_type="application/json",
            accept="application/json",
        )
        futures.append(future)

    assert await queue.size() == 3, "Queue should have 3 items"
    assert size_changes == [1, 2, 3], f"Size changes should be [1, 2, 3], got {size_changes}"

    # Drain queue
    requests = await queue.drain()
    assert len(requests) == 3, "Should drain 3 requests"
    assert await queue.size() == 0, "Queue should be empty"
    assert size_changes[-1] == 0, "Last size change should be 0"

    print("  PASSED: RequestQueue works correctly")


async def test_streaming_body():
    """Test StreamingBody mimics boto3."""
    print("Testing StreamingBody...")

    from batch_inference.models import StreamingBody

    data = b'{"test": "data"}'
    body = StreamingBody(data)

    # Test full read
    result = body.read()
    assert result == data, f"Full read should return all data, got {result}"
    assert body.read() == b"", "After full read, should be empty"

    # Reset with new body
    body = StreamingBody(data)
    assert body.read(5) == b'{"tes', "Partial read should work"
    assert body.read() == b't": "data"}', "Remaining read should work"

    # Test context manager
    with StreamingBody(data) as b:
        result = b.read()
        assert result == data

    print("  PASSED: StreamingBody works correctly")


async def test_jsonl_format():
    """Test JSONL record format."""
    print("Testing JSONL format...")

    from batch_inference.models import BatchRequest

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    request = BatchRequest(
        record_id="test-123",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}]
        }).encode(),
        content_type="application/json",
        accept="application/json",
        future=future,
    )

    record = request.to_jsonl_record()

    assert record["recordId"] == "test-123"
    assert "modelInput" in record
    assert record["modelInput"]["max_tokens"] == 100

    print("  PASSED: JSONL format is correct")


async def test_s3_uri_parsing():
    """Test S3 URI parsing."""
    print("Testing S3 URI parsing...")

    from batch_inference.s3 import parse_s3_uri

    bucket, key = parse_s3_uri("s3://my-bucket/path/to/file.jsonl")
    assert bucket == "my-bucket"
    assert key == "path/to/file.jsonl"

    bucket, key = parse_s3_uri("s3://bucket/")
    assert bucket == "bucket"
    assert key == ""

    try:
        parse_s3_uri("https://invalid.com/path")
        assert False, "Should raise ValueError for non-S3 URI"
    except ValueError:
        pass

    print("  PASSED: S3 URI parsing works")


async def test_context_manager():
    """Test async context manager."""
    print("Testing context manager...")

    async with BatchInferenceClient() as client:
        await client.setup(
            model_id="test-model",
            s3_input_uri="s3://bucket/in/",
            s3_output_uri="s3://bucket/out/",
            role_arn="arn:aws:iam::123:role/Test",
            min_batch_size=1,
            max_batch_size=10,
            batch_timeout=0.5,
        )

        async def mock_submit_batch(requests):
            for req in requests:
                req.future.set_result({
                    "body": MagicMock(read=lambda: b"{}"),
                    "contentType": "application/json"
                })

        client._submitter.submit_batch = mock_submit_batch

        response = await client.invoke_model(
            body=b'{"messages": []}',
            contentType="application/json",
        )
        assert "body" in response

    print("  PASSED: Context manager works")


async def test_not_setup_error():
    """Test error when calling invoke_model before setup."""
    print("Testing NotSetupError...")

    from batch_inference import NotSetupError

    client = BatchInferenceClient()

    try:
        await client.invoke_model(body=b"{}", contentType="application/json")
        assert False, "Should raise NotSetupError"
    except NotSetupError:
        pass

    print("  PASSED: NotSetupError raised correctly")


async def main():
    """Run all tests."""
    print("=" * 50)
    print("Batch Inference SDK Tests")
    print("=" * 50)
    print()

    await test_streaming_body()
    await test_jsonl_format()
    await test_s3_uri_parsing()
    await test_request_queue()
    await test_not_setup_error()
    await test_context_manager()
    await test_max_batch_trigger()
    await test_batching_logic()

    print()
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
