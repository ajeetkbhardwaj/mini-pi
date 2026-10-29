from __future__ import annotations

from typing import Awaitable, Callable

from mini_pi.agent.types import Message, TextBlock


async def default_summarizer(messages: list[Message]) -> str:
    """A fallback rule-based summarizer when no LLM summarizer is provided."""
    summary_lines: list[str] = []
    for idx, msg in enumerate(messages, 1):
        if msg.role == "user":
            summary_lines.append(f"- User requested: {msg.text[:120]}")
        elif msg.role == "assistant":
            calls = [c.name for c in msg.tool_calls]
            if calls:
                summary_lines.append(f"- Assistant called tools: {', '.join(calls)}")
            elif msg.text:
                summary_lines.append(f"- Assistant concluded: {msg.text[:100]}")
    return "\n".join(summary_lines) or "Previous session turns condensed."


async def compact_history(
    history: list[Message],
    summarize: Callable[[list[Message]], Awaitable[str]] | None = None,
    keep_recent: int = 10,
) -> list[Message]:
    """Condense older history into a summary message while preserving recent turns."""
    if len(history) <= keep_recent:
        return list(history)

    old = history[:-keep_recent]
    recent = history[-keep_recent:]

    if summarize is not None:
        summary_text = await summarize(old)
    else:
        summary_text = await default_summarizer(old)

    summary_message = Message.user(
        f"Earlier session summary:\n\n{summary_text}\n\n"
        "Continue from the most recent messages below."
    )

    if recent:
        recent[0].parent_id = summary_message.id

    return [summary_message, *recent]

