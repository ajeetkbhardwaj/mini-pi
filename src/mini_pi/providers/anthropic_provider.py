from __future__ import annotations

import os
import uuid
from typing import Any, AsyncIterator

import anthropic

from mini_pi.agent.types import (
    Block,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from mini_pi.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelClient,
    StreamChunk,
)


class AnthropicClient(ModelClient):
    def __init__(
        self,
        model_id: str = "claude-3-7-sonnet-20250219",
        api_key: str | None = None,
    ) -> None:
        self._model_id = model_id
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Please set the environment variable or pass api_key."
            )
        self.client = anthropic.AsyncAnthropic(api_key=resolved_key)

    @property
    def model_id(self) -> str:
        return self._model_id

    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        messages = self._to_anthropic_messages(request.messages)
        model = request.model or self._model_id

        tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool.get("input_schema", {}),
            }
            for tool in request.tools
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield StreamChunk(delta_text=delta.text)
                    elif delta.type == "thinking_delta":
                        yield StreamChunk(delta_thinking=delta.thinking)
                    elif delta.type == "input_json_delta":
                        yield StreamChunk(
                            delta_tool_calls=[
                                {"index": getattr(event, "index", 0), "arguments": delta.partial_json}
                            ]
                        )
                elif event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        yield StreamChunk(
                            delta_tool_calls=[
                                {
                                    "index": getattr(event, "index", 0),
                                    "id": block.id,
                                    "name": block.name,
                                }
                            ]
                        )
                elif event.type == "message_delta":
                    if hasattr(event, "usage") and event.usage:
                        yield StreamChunk(
                            output_tokens=getattr(event.usage, "output_tokens", 0)
                        )
                elif event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        yield StreamChunk(
                            input_tokens=getattr(event.message.usage, "input_tokens", 0)
                        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        full_text = ""
        full_thinking = ""
        in_tok = 0
        out_tok = 0
        think_tok = 0
        raw_tools: dict[int, dict[str, str]] = {}

        async for chunk in self.stream_complete(request):
            if chunk.delta_text:
                full_text += chunk.delta_text
            if chunk.delta_thinking:
                full_thinking += chunk.delta_thinking
            if chunk.input_tokens:
                in_tok = chunk.input_tokens
            if chunk.output_tokens:
                out_tok = chunk.output_tokens
            if chunk.thinking_tokens:
                think_tok = chunk.thinking_tokens

            for tc in chunk.delta_tool_calls:
                idx = tc.get("index", 0)
                if idx not in raw_tools:
                    raw_tools[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.get("id"):
                    raw_tools[idx]["id"] += tc["id"]
                if tc.get("name"):
                    raw_tools[idx]["name"] += tc["name"]
                if tc.get("arguments"):
                    raw_tools[idx]["arguments"] += tc["arguments"]

        blocks: list[Block] = []
        if full_thinking.strip():
            blocks.append(ThinkingBlock(thinking=full_thinking.strip()))
        if full_text.strip():
            blocks.append(TextBlock(text=full_text.strip()))

        import json
        for tc_data in raw_tools.values():
            parsed_args: dict[str, Any] = {}
            raw_arg = tc_data["arguments"]
            if raw_arg:
                try:
                    parsed_args = json.loads(raw_arg)
                except json.JSONDecodeError:
                    parsed_args = {"raw": raw_arg}
            blocks.append(
                ToolUseBlock(
                    id=tc_data["id"] or str(uuid.uuid4()),
                    name=tc_data["name"],
                    input=parsed_args,
                )
            )

        if think_tok == 0 and full_thinking:
            think_tok = max(1, len(full_thinking.split()) * 4 // 3)

        msg = Message(id=str(uuid.uuid4()), role="assistant", content=blocks)
        return CompletionResponse(
            message=msg,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thinking_tokens=think_tok,
        )

    def _to_anthropic_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "user":
                user_blocks = [
                    {"type": "text", "text": b.text}
                    for b in message.content
                    if isinstance(b, TextBlock)
                ]
                if user_blocks:
                    if converted and converted[-1]["role"] == "user":
                        converted[-1]["content"].extend(user_blocks)
                    else:
                        converted.append({"role": "user", "content": user_blocks})

            elif message.role == "assistant":
                asst_blocks: list[dict[str, Any]] = []
                for b in message.content:
                    if isinstance(b, TextBlock):
                        asst_blocks.append({"type": "text", "text": b.text})
                    elif isinstance(b, ToolUseBlock):
                        asst_blocks.append(
                            {
                                "type": "tool_use",
                                "id": b.id,
                                "name": b.name,
                                "input": b.input,
                            }
                        )
                if asst_blocks:
                    converted.append({"role": "assistant", "content": asst_blocks})

            elif message.role == "tool":
                tool_blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": b.tool_use_id,
                        "content": b.content,
                        "is_error": b.is_error,
                    }
                    for b in message.content
                    if isinstance(b, ToolResultBlock)
                ]
                if tool_blocks:
                    if converted and converted[-1]["role"] == "user":
                        converted[-1]["content"].extend(tool_blocks)
                    else:
                        converted.append({"role": "user", "content": tool_blocks})

        return converted
