from __future__ import annotations

from pathlib import Path
import pytest

from mini_pi.agent.loop import AgentConfig, AgentLoop
from mini_pi.agent.types import Message, TextBlock, ThinkingBlock
from mini_pi.providers.base import CompletionRequest, StreamChunk
from mini_pi.providers.mock_provider import MockModelClient
from mini_pi.providers.openai_provider import StreamTagFilter
from mini_pi.tools.registry import ToolRegistry


def test_stream_tag_filter_split_boundary():
    filt = StreamTagFilter()
    chunks = ["<th", "ink>analyzing problem</th", "ink>Here is the solution."]
    events = []
    for c in chunks:
        events.extend(filt.feed(c))
    events.extend(filt.flush())

    thinking_segments = [txt for is_think, txt in events if is_think]
    text_segments = [txt for is_think, txt in events if not is_think]

    assert "".join(thinking_segments) == "analyzing problem"
    assert "".join(text_segments) == "Here is the solution."


@pytest.mark.asyncio
async def test_agent_loop_streaming_events(tmp_path: Path):
    mock = MockModelClient(
        responses=[
            Message.assistant(
                text="The code is updated.",
                thinking="I should check the files first.",
            )
        ]
    )

    tools = ToolRegistry()
    config = AgentConfig(system_prompt="Test")
    agent = AgentLoop(model=mock, tools=tools, config=config, cwd=tmp_path)

    events_received = []
    agent.on_event = lambda e: events_received.append(e)

    history = []
    result = await agent.run(history, "Please update code.")

    event_types = [e["type"] for e in events_received]
    assert "thinking_chunk" in event_types
    assert "text_chunk" in event_types
    assert "message_end" in event_types

    thinking_chunks = [e["delta"] for e in events_received if e["type"] == "thinking_chunk"]
    text_chunks = [e["delta"] for e in events_received if e["type"] == "text_chunk"]

    assert "".join(thinking_chunks).strip() == "I should check the files first."
    assert "".join(text_chunks).strip() == "The code is updated."
    assert result[-1].thinking == "I should check the files first."
    assert result[-1].text == "The code is updated."

