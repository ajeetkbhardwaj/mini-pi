from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from mini_pi.agent.types import Message


@dataclass(frozen=True)
class CompletionRequest:
    system: str
    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 8192


@dataclass(frozen=True)
class CompletionResponse:
    message: Message
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


@dataclass
class StreamChunk:
    delta_text: str = ""
    delta_thinking: str = ""
    delta_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


class ModelClient(abc.ABC):
    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        ...

    @abc.abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        ...

    @abc.abstractmethod
    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        ...
