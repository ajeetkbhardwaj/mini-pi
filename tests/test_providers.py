from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from mini_pi.agent.types import Message, TextBlock, ToolResultBlock, ToolUseBlock
from mini_pi.providers.anthropic_provider import AnthropicClient
from mini_pi.providers.base import CompletionRequest
from mini_pi.providers.local_provider import LOCAL_PRESETS, LocalOpenAIClient
from mini_pi.providers.openai_provider import OpenAIClient


def test_anthropic_message_conversion(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AnthropicClient()

    messages = [
        Message.user("Hello"),
        Message.assistant(
            text="Checking",
            tool_calls=[ToolUseBlock(id="call_1", name="read", input={"path": "a.txt"})],
        ),
        Message.tool_result(tool_use_id="call_1", content="file contents"),
    ]

    anthropic_msgs = client._to_anthropic_messages(messages)
    assert len(anthropic_msgs) == 3
    assert anthropic_msgs[0]["role"] == "user"
    assert anthropic_msgs[1]["role"] == "assistant"
    assert anthropic_msgs[2]["role"] == "user"
    assert anthropic_msgs[2]["content"][0]["type"] == "tool_result"
    assert anthropic_msgs[2]["content"][0]["tool_use_id"] == "call_1"


def test_openai_message_conversion():
    client = OpenAIClient(api_key="test-key")

    messages = [
        Message.user("Hello"),
        Message.assistant(
            text="Checking",
            tool_calls=[ToolUseBlock(id="call_1", name="read", input={"path": "a.txt"})],
        ),
        Message.tool_result(tool_use_id="call_1", content="file contents"),
    ]

    openai_msgs = client._to_openai_messages("System instructions", messages)
    assert len(openai_msgs) == 4
    assert openai_msgs[0]["role"] == "system"
    assert openai_msgs[1]["role"] == "user"
    assert openai_msgs[2]["role"] == "assistant"
    assert openai_msgs[2]["tool_calls"][0]["function"]["name"] == "read"
    assert openai_msgs[3]["role"] == "tool"
    assert openai_msgs[3]["tool_call_id"] == "call_1"
    assert openai_msgs[3]["content"] == "file contents"


def test_local_provider_presets(monkeypatch):
    monkeypatch.delenv("LOCAL_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    lmstudio_client = LocalOpenAIClient(backend="lmstudio")
    assert lmstudio_client.base_url == "http://localhost:1234/v1"
    assert lmstudio_client.model_id == "local-model"

    ollama_client = LocalOpenAIClient(backend="ollama")
    assert ollama_client.base_url == "http://localhost:11434/v1"
    assert ollama_client.model_id == "qwen2.5-coder"

    llamacpp_client = LocalOpenAIClient(backend="llamacpp")
    assert llamacpp_client.base_url == "http://localhost:8080/v1"
    assert llamacpp_client.model_id == "default"


@pytest.mark.asyncio
async def test_openai_reasoning_and_thinking_extraction(monkeypatch):
    client = OpenAIClient(api_key="test-key")

    mock_msg = MagicMock()
    mock_msg.content = "<think>Let me calculate 2+2=4.</think>\nThe answer is 4."
    mock_msg.reasoning_content = None
    mock_msg.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_details = MagicMock()
    mock_details.reasoning_tokens = 12

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.completion_tokens_details = mock_details

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage

    monkeypatch.setattr(client.client.chat.completions, "create", AsyncMock(return_value=mock_resp))

    req = CompletionRequest(system="", messages=[Message.user("2+2?")])
    res = await client.complete(req)

    assert res.message.thinking == "Let me calculate 2+2=4."
    assert res.message.text == "The answer is 4."
    assert res.thinking_tokens == 12
    assert res.input_tokens == 10
    assert res.output_tokens == 20
