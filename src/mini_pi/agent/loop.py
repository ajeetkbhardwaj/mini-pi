from __future__ import annotations

import inspect
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from mini_pi.agent.types import (
    Block,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from mini_pi.providers.base import CompletionRequest, ModelClient
from mini_pi.tools.registry import ToolContext, ToolRegistry


@dataclass
class AgentConfig:
    system_prompt: str
    max_turns: int = 40
    temperature: float = 0.0


class AgentLoop:
    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        config: AgentConfig,
        cwd: str | Path = ".",
        on_event: Callable[[dict[str, Any]], None] | None = None,
        tool_interceptor: (
            Callable[[str, dict[str, Any]], Awaitable[tuple[bool, str]]] | None
        ) = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config
        self.cwd = Path(cwd).resolve()
        self.on_event = on_event or (lambda event: None)
        self.tool_interceptor = tool_interceptor
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_thinking_tokens = 0

    async def run(
        self, history: list[Message], user_text: str
    ) -> list[Message]:
        parent_id = history[-1].id if history else None
        user_message = Message.user(user_text, parent_id=parent_id)

        history.append(user_message)
        self._emit("agent_start", {"user_message_id": user_message.id, "prompt": user_text})

        for turn in range(self.config.max_turns):
            self._emit("turn_start", {"turn": turn + 1})

            request = CompletionRequest(
                system=self.config.system_prompt,
                messages=history,
                tools=self.tools.schemas(),
                temperature=self.config.temperature,
            )

            turn_thinking = ""
            turn_text = ""
            raw_tools: dict[int, dict[str, str]] = {}
            in_tok = 0
            out_tok = 0
            think_tok = 0

            # Stream response chunks from model
            if hasattr(self.model, "stream_complete"):
                async for chunk in self.model.stream_complete(request):
                    if chunk.delta_thinking:
                        turn_thinking += chunk.delta_thinking
                        self._emit(
                            "thinking_chunk",
                            {"delta": chunk.delta_thinking, "accumulated": turn_thinking},
                        )
                    if chunk.delta_text:
                        turn_text += chunk.delta_text
                        self._emit(
                            "text_chunk",
                            {"delta": chunk.delta_text, "accumulated": turn_text},
                        )
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
                if turn_thinking.strip():
                    blocks.append(ThinkingBlock(thinking=turn_thinking.strip()))
                if turn_text.strip():
                    blocks.append(TextBlock(text=turn_text.strip()))

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

                if think_tok == 0 and turn_thinking:
                    think_tok = max(1, len(turn_thinking.split()) * 4 // 3)
                if out_tok == 0 and (turn_text or turn_thinking):
                    out_tok = max(1, len((turn_text + turn_thinking).split()) * 4 // 3)

                assistant_message = Message(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content=blocks,
                    parent_id=history[-1].id,
                )
            else:
                response = await self.model.complete(request=request)
                in_tok = response.input_tokens
                out_tok = response.output_tokens
                think_tok = response.thinking_tokens
                assistant_message = response.message
                assistant_message.parent_id = history[-1].id

            self.total_input_tokens += in_tok
            self.total_output_tokens += out_tok
            self.total_thinking_tokens += think_tok
            history.append(assistant_message)

            self._emit(
                "message_end",
                {
                    "message": assistant_message.model_dump(),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "thinking_tokens": think_tok,
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                    "total_thinking_tokens": self.total_thinking_tokens,
                },
            )

            tool_calls = assistant_message.tool_calls
            if not tool_calls:
                self._emit("agent_end", {"reason": "final_response"})
                self._emit("agent_settled", {})
                return history

            for call in tool_calls:
                result_message = await self._execute_tool(call, history[-1].id)
                history.append(result_message)
                self._emit("tool_result", {"result": result_message.model_dump()})

        self._emit("agent_end", {"reason": "max_turns_exceeded"})
        self._emit("agent_settled", {})
        raise RuntimeError(f"Agent exceeded maximum turns ({self.config.max_turns})")

    async def _execute_tool(self, call: ToolUseBlock, parent_id: str) -> Message:
        ctx = ToolContext(cwd=self.cwd)
        self._emit(
            "tool_execution_start",
            {"tool": call.name, "input": call.input, "call_id": call.id},
        )

        # Check safety interceptor if registered
        if self.tool_interceptor:
            try:
                allowed, reason = await self.tool_interceptor(call.name, call.input)
                if not allowed:
                    output = f"Execution blocked by safety policy: {reason}"
                    self._emit(
                        "tool_execution_end",
                        {"tool": call.name, "is_error": True, "output": output},
                    )
                    return Message.tool_result(
                        tool_use_id=call.id,
                        content=output,
                        is_error=True,
                        parent_id=parent_id,
                    )
            except Exception as e:
                output = f"Safety interceptor error: {e}"
                return Message.tool_result(
                    tool_use_id=call.id,
                    content=output,
                    is_error=True,
                    parent_id=parent_id,
                )

        try:
            output = await self.tools.execute(call.name, call.input, ctx)
            is_error = False
        except Exception as error:
            output = f"Tool error ({type(error).__name__}): {error}"
            is_error = True

        self._emit(
            "tool_execution_end",
            {
                "tool": call.name,
                "is_error": is_error,
                "output_preview": output[:300] if output else "",
            },
        )

        return Message.tool_result(
            tool_use_id=call.id,
            content=output,
            is_error=is_error,
            parent_id=parent_id,
        )

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"type": event_type, **payload}
        try:
            res = self.on_event(event)
            if inspect.isawaitable(res):
                import asyncio
                asyncio.create_task(res)
        except Exception:
            pass
