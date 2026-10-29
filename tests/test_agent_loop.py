from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from mini_pi.agent.loop import AgentConfig, AgentLoop
from mini_pi.agent.types import Message, TextBlock, ToolUseBlock
from mini_pi.providers.local_provider import LocalOpenAIClient
from mini_pi.providers.mock_provider import MockModelClient
from mini_pi.tools.read import ReadTool
from mini_pi.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_agent_loop_simple_response(tmp_path: Path):
    mock_model = MockModelClient(
        responses=[
            Message.assistant("I am ready to help."),
        ]
    )
    tools = ToolRegistry()
    config = AgentConfig(system_prompt="You are an assistant.")
    agent = AgentLoop(model=mock_model, tools=tools, config=config, cwd=tmp_path)

    events = []
    agent.on_event = lambda e: events.append(e["type"])

    history = []
    result = await agent.run(history, "Hello!")

    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].text == "Hello!"
    assert result[1].role == "assistant"
    assert result[1].text == "I am ready to help."
    assert "agent_start" in events
    assert "agent_end" in events


@pytest.mark.asyncio
async def test_agent_loop_tool_execution(tmp_path: Path):
    # Setup test file
    test_file = tmp_path / "data.txt"
    test_file.write_text("special_secret_code_123", encoding="utf-8")

    # Turn 1: Model calls read
    call = ToolUseBlock(id="call_read_1", name="read", input={"path": "data.txt"})
    msg_1 = Message.assistant(text="Reading file...", tool_calls=[call])

    # Turn 2: Model finishes
    msg_2 = Message.assistant(text="I found special_secret_code_123")

    mock_model = MockModelClient(responses=[msg_1, msg_2])

    tools = ToolRegistry()
    tools.register(ReadTool())

    config = AgentConfig(system_prompt="Test agent")
    agent = AgentLoop(model=mock_model, tools=tools, config=config, cwd=tmp_path)

    history = []
    result = await agent.run(history, "Find secret")

    assert len(result) == 4
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    assert result[2].role == "tool"
    assert "special_secret_code_123" in result[2].tool_results[0].content
    assert result[3].role == "assistant"
    assert "special_secret_code_123" in result[3].text


@pytest.mark.asyncio
async def test_agent_loop_with_local_provider(tmp_path: Path, monkeypatch):
    # Test local OpenAI provider integration (LMStudio/Ollama/llama.cpp)
    local_client = LocalOpenAIClient(backend="lmstudio", model_id="local-model")

    mock_choice = MagicMock()
    mock_choice.message.content = "Local LMStudio model response"
    mock_choice.message.tool_calls = None

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 15
    mock_completion.usage.completion_tokens = 8

    # Patch OpenAI API call
    monkeypatch.setattr(
        local_client._inner_client.client.chat.completions,
        "create",
        AsyncMock(return_value=mock_completion),
    )

    tools = ToolRegistry()
    config = AgentConfig(system_prompt="System instructions")
    agent = AgentLoop(model=local_client, tools=tools, config=config, cwd=tmp_path)

    history = []
    result = await agent.run(history, "Explain codebase with local model")

    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    assert result[1].text == "Local LMStudio model response"
