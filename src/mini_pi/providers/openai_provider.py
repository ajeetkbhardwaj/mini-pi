from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, AsyncIterator

import openai

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


class StreamTagFilter:
    """Isolates <think>...</think> tags on incoming text stream chunks."""

    def __init__(self, open_tag: str = "<think>", close_tag: str = "</think>") -> None:
        self.open_tag = open_tag
        self.close_tag = close_tag
        self.in_think = False
        self.buf = ""

    def feed(self, chunk: str) -> list[tuple[bool, str]]:
        self.buf += chunk
        res: list[tuple[bool, str]] = []
        while self.buf:
            if not self.in_think:
                if self.open_tag in self.buf:
                    before, _, self.buf = self.buf.partition(self.open_tag)
                    if before:
                        res.append((False, before))
                    self.in_think = True
                else:
                    prefix_match = False
                    for i in range(1, len(self.open_tag)):
                        if self.open_tag.startswith(self.buf[-i:]):
                            safe_len = len(self.buf) - i
                            if safe_len > 0:
                                res.append((False, self.buf[:safe_len]))
                                self.buf = self.buf[safe_len:]
                            prefix_match = True
                            break
                    if not prefix_match:
                        res.append((False, self.buf))
                        self.buf = ""
                    break
            else:
                if self.close_tag in self.buf:
                    before, _, self.buf = self.buf.partition(self.close_tag)
                    if before:
                        res.append((True, before))
                    self.in_think = False
                else:
                    prefix_match = False
                    for i in range(1, len(self.close_tag)):
                        if self.close_tag.startswith(self.buf[-i:]):
                            safe_len = len(self.buf) - i
                            if safe_len > 0:
                                res.append((True, self.buf[:safe_len]))
                                self.buf = self.buf[safe_len:]
                            prefix_match = True
                            break
                    if not prefix_match:
                        res.append((True, self.buf))
                        self.buf = ""
                    break
        return res

    def flush(self) -> list[tuple[bool, str]]:
        res: list[tuple[bool, str]] = []
        if self.buf:
            res.append((self.in_think, self.buf))
            self.buf = ""
        return res


class OpenAIClient(ModelClient):
    def __init__(
        self,
        model_id: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model_id = model_id
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy-key-for-local")
        self.client = openai.AsyncOpenAI(
            api_key=resolved_key,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        messages = self._to_openai_messages(request.system, request.messages)
        model = request.model or self._model_id

        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in request.tools
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            stream = await self.client.chat.completions.create(
                **kwargs, stream_options={"include_usage": True}
            )
        except Exception:
            stream = await self.client.chat.completions.create(**kwargs)

        tag_filter = StreamTagFilter()

        if hasattr(stream, "choices"):
            async def _single():
                yield stream
            iterator = _single()
        else:
            iterator = stream

        async for chunk in iterator:
            # Check for usage info
            if getattr(chunk, "usage", None):
                u = chunk.usage
                in_tok = getattr(u, "prompt_tokens", 0) or 0
                out_tok = getattr(u, "completion_tokens", 0) or 0
                details = getattr(u, "completion_tokens_details", None)
                think_tok = getattr(details, "reasoning_tokens", 0) if details else 0
                yield StreamChunk(
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    thinking_tokens=think_tok or 0,
                )

            if not getattr(chunk, "choices", None):
                continue

            choice = chunk.choices[0]
            choice_msg = getattr(choice, "message", None)
            choice_delta = getattr(choice, "delta", None)
            if choice_msg is not None and isinstance(getattr(choice_msg, "content", None), str):
                delta = choice_msg
            elif choice_delta is not None:
                delta = choice_delta
            else:
                delta = None

            if not delta:
                continue

            # 1. Native reasoning_content
            for attr in ("reasoning_content", "reasoning", "thinking"):
                reasoning = getattr(delta, attr, None)
                if isinstance(reasoning, str) and reasoning:
                    yield StreamChunk(delta_thinking=reasoning)
                    break

            # 2. Text content with <think> tag isolation
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                for is_think, text_seg in tag_filter.feed(content):
                    if is_think:
                        yield StreamChunk(delta_thinking=text_seg)
                    else:
                        yield StreamChunk(delta_text=text_seg)

            # 3. Tool call fragments
            if getattr(delta, "tool_calls", None):
                raw_calls = []
                for tc in delta.tool_calls:
                    tc_dict: dict[str, Any] = {"index": getattr(tc, "index", 0)}
                    if getattr(tc, "id", None):
                        tc_dict["id"] = tc.id
                    if getattr(tc, "function", None):
                        fn = tc.function
                        if getattr(fn, "name", None):
                            tc_dict["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            tc_dict["arguments"] = fn.arguments
                    raw_calls.append(tc_dict)
                yield StreamChunk(delta_tool_calls=raw_calls)

        # Flush any remaining buffer in tag filter
        for is_think, text_seg in tag_filter.flush():
            if is_think:
                yield StreamChunk(delta_thinking=text_seg)
            else:
                yield StreamChunk(delta_text=text_seg)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        full_text = ""
        full_thinking = ""
        raw_tools: dict[int, dict[str, str]] = {}
        in_tok = 0
        out_tok = 0
        think_tok = 0

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
        if out_tok == 0 and (full_text or full_thinking):
            out_tok = max(1, len((full_text + full_thinking).split()) * 4 // 3)

        msg = Message(id=str(uuid.uuid4()), role="assistant", content=blocks)
        return CompletionResponse(
            message=msg,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thinking_tokens=think_tok,
        )

    def _to_openai_messages(
        self, system_prompt: str, messages: list[Message]
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        if system_prompt:
            converted.append({"role": "system", "content": system_prompt})

        for message in messages:
            if message.role == "user":
                text = message.text
                if text:
                    converted.append({"role": "user", "content": text})

            elif message.role == "assistant":
                text = message.text or None
                tool_calls_payload = []
                for tc in message.tool_calls:
                    tool_calls_payload.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.input),
                            },
                        }
                    )
                msg_dict: dict[str, Any] = {"role": "assistant"}
                if text:
                    msg_dict["content"] = text
                if tool_calls_payload:
                    msg_dict["tool_calls"] = tool_calls_payload
                converted.append(msg_dict)

            elif message.role == "tool":
                for res in message.tool_results:
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": res.tool_use_id,
                            "content": res.content,
                        }
                    )

        return converted
