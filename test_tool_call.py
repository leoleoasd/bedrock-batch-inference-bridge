"""Test tool calling with litellm mode."""

import asyncio
import json

from batch_inference import BatchInferenceClient


async def test_tool_call():
    """Test tool calling with litellm mode."""
    client = BatchInferenceClient()

    await client.setup(
        model_id="not-used",
        litellm_mode=True,
        litellm_model="hosted_vllm/Qwen/Qwen3-32B",
        litellm_api_base="http://127.0.0.1:30000/v1",
    )

    # Define a simple tool in Bedrock format
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    ]

    # First request - should trigger tool use
    print("=" * 60)
    print("Test 1: Initial request with tools")
    print("=" * 60)
    
    response = await client.invoke_model(
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "tools": tools,
            "messages": [
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ]
        }),
        contentType="application/json",
    )

    result = json.loads(response["body"].read())
    print(f"Stop reason: {result.get('stop_reason')}")
    print(f"Content blocks: {len(result.get('content', []))}")
    
    for block in result.get("content", []):
        if block.get("type") == "text":
            print(f"Text: {block.get('text', '')[:200]}")
        elif block.get("type") == "tool_use":
            print(f"Tool use: {block.get('name')}")
            print(f"  ID: {block.get('id')}")
            print(f"  Input: {json.dumps(block.get('input', {}))}")

    # If we got a tool_use, simulate the tool result and continue
    tool_use_blocks = [b for b in result.get("content", []) if b.get("type") == "tool_use"]
    
    if tool_use_blocks:
        print()
        print("=" * 60)
        print("Test 2: Continue with tool result")
        print("=" * 60)
        
        # Build the follow-up request with tool result
        messages = [
            {"role": "user", "content": "What's the weather like in San Francisco?"},
            {"role": "assistant", "content": result.get("content", [])},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_blocks[0].get("id"),
                        "content": json.dumps({
                            "temperature": 72,
                            "unit": "fahrenheit",
                            "condition": "sunny"
                        })
                    }
                ]
            }
        ]

        response2 = await client.invoke_model(
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "tools": tools,
                "messages": messages
            }),
            contentType="application/json",
        )

        result2 = json.loads(response2["body"].read())
        print(f"Stop reason: {result2.get('stop_reason')}")
        
        for block in result2.get("content", []):
            if block.get("type") == "text":
                print(f"Text: {block.get('text', '')[:500]}")

    await client.close()
    print()
    print("Tool call test completed!")


if __name__ == "__main__":
    asyncio.run(test_tool_call())
