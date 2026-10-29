from __future__ import annotations

import json
import uuid
from typing import AsyncIterator, Callable

from mini_pi.agent.types import Message, TextBlock, ThinkingBlock, ToolUseBlock
from mini_pi.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelClient,
    StreamChunk,
)


class MockModelClient(ModelClient):
    """A deterministic mock client for offline testing and verification."""

    def __init__(
        self,
        responses: list[Message] | None = None,
        generator: Callable[[CompletionRequest], Message] | None = None,
        model_id: str = "mock-model",
    ) -> None:
        self._responses = list(responses or [])
        self._generator = generator
        self._model_id = model_id
        self.call_history: list[CompletionRequest] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        self.call_history.append(request)

        if self._generator:
            msg = self._generator(request)
        elif self._responses:
            msg = self._responses.pop(0)
        else:
            msg = Message(
                id=str(uuid.uuid4()),
                role="assistant",
                content=[TextBlock(text="Mock default answer")],
            )

        # Stream thinking if present
        for b in msg.content:
            if isinstance(b, ThinkingBlock):
                for word in b.thinking.split():
                    yield StreamChunk(delta_thinking=word + " ")
            elif isinstance(b, TextBlock):
                for word in b.text.split():
                    yield StreamChunk(delta_text=word + " ")
            elif isinstance(b, ToolUseBlock):
                yield StreamChunk(
                    delta_tool_calls=[
                        {
                            "index": 0,
                            "id": b.id,
                            "name": b.name,
                            "arguments": json.dumps(b.input),
                        }
                    ]
                )

        yield StreamChunk(input_tokens=10, output_tokens=10, thinking_tokens=5 if msg.thinking else 0)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        full_text = ""
        full_thinking = ""
        in_tok = 10
        out_tok = 10
        think_tok = 0
        raw_tools: list[ToolUseBlock] = []

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
                raw_arg = tc.get("arguments", "{}")
                try:
                    args = json.loads(raw_arg)
                except Exception:
                    args = {}
                raw_tools.append(
                    ToolUseBlock(
                        id=tc.get("id", str(uuid.uuid4())),
                        name=tc.get("name", ""),
                        input=args,
                    )
                )

        blocks = []
        if full_thinking.strip():
            blocks.append(ThinkingBlock(thinking=full_thinking.strip()))
        if full_text.strip():
            blocks.append(TextBlock(text=full_text.strip()))
        blocks.extend(raw_tools)

        msg = Message(id=str(uuid.uuid4()), role="assistant", content=blocks)
        return CompletionResponse(
            message=msg,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thinking_tokens=think_tok,
        )
