from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


Block = Annotated[
    Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock],
    Field(discriminator="type"),
]


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Role
    content: list[Block] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    parent_id: str | None = None

    @property
    def text(self) -> str:
        """Concatenated text of all text blocks in the message."""
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def thinking(self) -> str:
        """Concatenated reasoning/thinking text from all thinking blocks."""
        return "\n".join(b.thinking for b in self.content if isinstance(b, ThinkingBlock))

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        """All tool use blocks in the message."""
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def tool_results(self) -> list[ToolResultBlock]:
        """All tool result blocks in the message."""
        return [b for b in self.content if isinstance(b, ToolResultBlock)]

    @classmethod
    def user(cls, text: str, parent_id: str | None = None) -> Message:
        return cls(role="user", content=[TextBlock(text=text)], parent_id=parent_id)

    @classmethod
    def assistant(
        cls,
        text: str = "",
        thinking: str = "",
        tool_calls: list[ToolUseBlock] | None = None,
        parent_id: str | None = None,
    ) -> Message:
        blocks: list[Block] = []
        if thinking:
            blocks.append(ThinkingBlock(thinking=thinking))
        if text:
            blocks.append(TextBlock(text=text))
        if tool_calls:
            blocks.extend(tool_calls)
        return cls(role="assistant", content=blocks, parent_id=parent_id)

    @classmethod
    def tool_result(
        cls,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
        parent_id: str | None = None,
    ) -> Message:
        return cls(
            role="tool",
            content=[
                ToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=content,
                    is_error=is_error,
                )
            ],
            parent_id=parent_id,
        )


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict[str, Any]


class ToolResult(BaseModel):
    tool_call_id: str
    output: str
    is_error: bool = False
