"""Main client for batch inference SDK."""

import json
import logging
from typing import Any

import aioboto3

from .exceptions import AlreadySetupError, NotSetupError
from .models import BatchConfig, StreamingBody
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
        s3_input_uri: str = "",
        s3_output_uri: str = "",
        role_arn: str = "",
        region: str = "us-west-2",
        min_batch_size: int = 100,
        max_batch_size: int = 1000,
        batch_timeout: float = 60.0,
        poll_interval: float = 5.0,
        job_timeout_hours: int = 24,
        job_name_prefix: str = "batch-inference",
        debug_mode: bool = False,
        litellm_mode: bool = False,
        litellm_model: str = "",
        litellm_api_base: str = "",
    ) -> None:
        """Configure and start the batch inference client.

        Args:
            model_id: Bedrock model ID to use for all requests.
            s3_input_uri: S3 URI for batch input files (e.g., s3://bucket/input/).
                Not required in debug_mode or litellm_mode.
            s3_output_uri: S3 URI for batch output files (e.g., s3://bucket/output/).
                Not required in debug_mode or litellm_mode.
            role_arn: IAM role ARN with permissions for batch inference.
                Not required in debug_mode or litellm_mode.
            region: AWS region (default "us-west-2").
            min_batch_size: Minimum requests before time-based submission (default 100).
            max_batch_size: Submit immediately when this many requests queued (default 1000).
            batch_timeout: Seconds to wait before submitting if >= min_batch_size (default 60).
            poll_interval: Seconds between job status polls (default 5).
            job_timeout_hours: Maximum hours for batch job (default 24).
            job_name_prefix: Prefix for batch job names (default "batch-inference").
            debug_mode: If True, use direct invoke_model calls instead of batch
                inference. Useful for testing without S3/batch job setup.
            litellm_mode: If True, use litellm acompletion instead of batch inference.
                Transforms Bedrock request format to OpenAI format automatically.
            litellm_model: Model name for litellm (e.g., "gpt-4o", "anthropic/claude-3-opus").
                Required if litellm_mode is True.
            litellm_api_base: API base URL for litellm (e.g., "https://hosted-vllm-api.co").
                Optional, used for custom endpoints like vLLM or other OpenAI-compatible APIs.
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
            debug_mode=debug_mode,
            litellm_mode=litellm_mode,
            litellm_model=litellm_model,
            litellm_api_base=litellm_api_base,
        )

        # In litellm mode, skip batch infrastructure setup
        if litellm_mode:
            if not litellm_model:
                raise ValueError("litellm_model is required when litellm_mode is True")
            logger.info(
                f"BatchInferenceClient setup in LITELLM MODE: model={litellm_model}, "
                "using litellm acompletion calls"
            )
            self._is_setup = True
            return

        # In debug mode, skip batch infrastructure setup
        if debug_mode:
            logger.info(
                f"BatchInferenceClient setup in DEBUG MODE: model={model_id}, "
                "using direct invoke_model calls"
            )
            self._is_setup = True
            return

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
        if not self._is_setup or self._config is None:
            raise NotSetupError()

        # Convert str to bytes if needed
        if isinstance(body, str):
            body = body.encode("utf-8")

        # Litellm mode: use litellm acompletion
        if self._config.litellm_mode:
            return await self._invoke_model_litellm(body)

        # Debug mode: call bedrock-runtime invoke_model directly
        if self._config.debug_mode:
            return await self._invoke_model_direct(body, contentType, accept)

        # Batch mode: queue the request
        if self._queue is None:
            raise NotSetupError()

        # Queue the request and get a future
        future = await self._queue.put(
            body=body,
            content_type=contentType,
            accept=accept,
        )

        # Wait for the result
        return await future

    async def _invoke_model_direct(
        self,
        body: bytes,
        content_type: str,
        accept: str,
    ) -> dict[str, Any]:
        """Invoke model directly using bedrock-runtime (debug mode).

        Args:
            body: Request payload in bytes.
            content_type: MIME type of input.
            accept: Desired response MIME type.

        Returns:
            Response dict matching boto3 invoke_model response.
        """
        assert self._config is not None

        async with self._session.client(
            "bedrock-runtime", region_name=self._config.region
        ) as bedrock_runtime:
            response = await bedrock_runtime.invoke_model(
                modelId=self._config.model_id,
                body=body,
                contentType=content_type,
                accept=accept,
            )

            # Read the response body and wrap it in our StreamingBody
            response_body = await response["body"].read()
            return {
                "body": StreamingBody(response_body),
                "contentType": response.get("contentType", "application/json"),
            }

    async def _invoke_model_litellm(
        self,
        body: bytes,
    ) -> dict[str, Any]:
        """Invoke model using litellm acompletion (litellm mode).

        Transforms Bedrock Claude request format to OpenAI format,
        calls litellm, and transforms the response back.

        Args:
            body: Request payload in Bedrock format (bytes).

        Returns:
            Response dict matching boto3 invoke_model response.
        """
        assert self._config is not None

        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for litellm_mode. "
                "Install it with: pip install litellm"
            )

        # Parse Bedrock request
        bedrock_request = json.loads(body.decode("utf-8"))

        # Transform Bedrock format to OpenAI format
        openai_kwargs = self._bedrock_to_openai(bedrock_request)

        # Add api_base if configured
        if self._config.litellm_api_base:
            openai_kwargs["api_base"] = self._config.litellm_api_base

        # Call litellm
        response = await litellm.acompletion(
            model=self._config.litellm_model,
            **openai_kwargs,
        )

        # Transform response back to Bedrock format
        bedrock_response = self._openai_to_bedrock(response)

        # Wrap in StreamingBody for compatibility
        response_bytes = json.dumps(bedrock_response).encode("utf-8")
        return {
            "body": StreamingBody(response_bytes),
            "contentType": "application/json",
        }

    def _bedrock_to_openai(self, bedrock_request: dict[str, Any]) -> dict[str, Any]:
        """Transform Bedrock Claude request format to OpenAI format.

        Args:
            bedrock_request: Request in Bedrock Claude format.

        Returns:
            Request kwargs for OpenAI/litellm format.
        """
        openai_kwargs: dict[str, Any] = {}

        # Messages - format is similar but need to handle content blocks
        if "messages" in bedrock_request:
            openai_messages = []
            for msg in bedrock_request["messages"]:
                openai_msg: dict[str, Any] = {"role": msg["role"]}

                # Handle content - can be string or list of content blocks
                content = msg.get("content", "")
                if isinstance(content, str):
                    openai_msg["content"] = content
                    openai_messages.append(openai_msg)
                elif isinstance(content, list):
                    # Convert content blocks to OpenAI format
                    parts = []
                    tool_calls = []
                    tool_results = []
                    
                    for block in content:
                        block_type = block.get("type")
                        
                        if block_type == "text":
                            parts.append({"type": "text", "text": block.get("text", "")})
                        elif block_type == "image":
                            # Handle image blocks
                            source = block.get("source", {})
                            if source.get("type") == "base64":
                                parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                                    }
                                })
                        elif block_type == "tool_use":
                            # Bedrock tool_use -> OpenAI tool_calls
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                }
                            })
                        elif block_type == "tool_result":
                            # Bedrock tool_result -> OpenAI tool message
                            tool_results.append({
                                "tool_use_id": block.get("tool_use_id", ""),
                                "content": block.get("content", ""),
                            })
                    
                    # Handle tool_calls in assistant messages
                    if tool_calls and msg["role"] == "assistant":
                        openai_msg["tool_calls"] = tool_calls
                        # Content can be empty or text for assistant with tool_calls
                        if parts:
                            if len(parts) == 1 and parts[0].get("type") == "text":
                                openai_msg["content"] = parts[0]["text"]
                            else:
                                openai_msg["content"] = parts
                        else:
                            openai_msg["content"] = None
                        openai_messages.append(openai_msg)
                    # Handle tool_results - convert to separate tool messages
                    elif tool_results:
                        for result in tool_results:
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": result["tool_use_id"],
                                "content": result["content"] if isinstance(result["content"], str) else json.dumps(result["content"]),
                            }
                            openai_messages.append(tool_msg)
                    else:
                        # Regular content
                        if len(parts) == 1 and parts[0].get("type") == "text":
                            openai_msg["content"] = parts[0]["text"]
                        elif parts:
                            openai_msg["content"] = parts
                        else:
                            openai_msg["content"] = ""
                        openai_messages.append(openai_msg)
                else:
                    openai_msg["content"] = ""
                    openai_messages.append(openai_msg)

            openai_kwargs["messages"] = openai_messages

        # System prompt - in Bedrock it can be a separate field
        if "system" in bedrock_request:
            system_content = bedrock_request["system"]
            if isinstance(system_content, str):
                openai_kwargs["messages"] = [
                    {"role": "system", "content": system_content}
                ] + openai_kwargs.get("messages", [])
            elif isinstance(system_content, list):
                # Handle list of system content blocks
                text_parts = [
                    block.get("text", "")
                    for block in system_content
                    if block.get("type") == "text"
                ]
                openai_kwargs["messages"] = [
                    {"role": "system", "content": "\n".join(text_parts)}
                ] + openai_kwargs.get("messages", [])

        # Max tokens
        if "max_tokens" in bedrock_request:
            openai_kwargs["max_tokens"] = bedrock_request["max_tokens"]

        # Temperature
        if "temperature" in bedrock_request:
            openai_kwargs["temperature"] = bedrock_request["temperature"]

        # Top P
        if "top_p" in bedrock_request:
            openai_kwargs["top_p"] = bedrock_request["top_p"]

        # Stop sequences
        if "stop_sequences" in bedrock_request:
            openai_kwargs["stop"] = bedrock_request["stop_sequences"]

        # Tools - convert Bedrock tool format to OpenAI format
        if "tools" in bedrock_request:
            openai_tools = []
            for tool in bedrock_request["tools"]:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    }
                }
                openai_tools.append(openai_tool)
            openai_kwargs["tools"] = openai_tools

        # Tool choice - convert Bedrock tool_choice to OpenAI format
        if "tool_choice" in bedrock_request:
            bedrock_choice = bedrock_request["tool_choice"]
            if isinstance(bedrock_choice, dict):
                choice_type = bedrock_choice.get("type", "auto")
                if choice_type == "auto":
                    openai_kwargs["tool_choice"] = "auto"
                elif choice_type == "any":
                    openai_kwargs["tool_choice"] = "required"
                elif choice_type == "tool":
                    openai_kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": bedrock_choice.get("name", "")}
                    }
            else:
                openai_kwargs["tool_choice"] = bedrock_choice

        # Top K - not directly supported in OpenAI, skip
        # anthropic_version - skip

        return openai_kwargs

    def _openai_to_bedrock(self, response: Any) -> dict[str, Any]:
        """Transform OpenAI/litellm response to Bedrock Claude format.

        Args:
            response: Response from litellm acompletion.

        Returns:
            Response in Bedrock Claude format.
        """
        choice = response.choices[0]
        message = choice.message

        # Build content blocks
        content: list[dict[str, Any]] = []
        
        # Add text content if present
        if message.content:
            content.append({
                "type": "text",
                "text": message.content,
            })

        # Handle tool calls - convert OpenAI tool_calls to Bedrock tool_use blocks
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_use_block: dict[str, Any] = {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                }
                # Parse arguments from JSON string
                try:
                    tool_use_block["input"] = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_use_block["input"] = {"raw": tool_call.function.arguments}
                content.append(tool_use_block)

        # Build Bedrock-style response
        bedrock_response: dict[str, Any] = {
            "id": response.id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": response.model,
            "stop_reason": self._openai_finish_reason_to_bedrock(choice.finish_reason),
        }

        # Add usage if available
        if response.usage:
            bedrock_response["usage"] = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return bedrock_response

    def _openai_finish_reason_to_bedrock(self, finish_reason: str | None) -> str:
        """Convert OpenAI finish_reason to Bedrock stop_reason."""
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "content_filtered",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
        }
        return mapping.get(finish_reason or "", "end_turn")

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
