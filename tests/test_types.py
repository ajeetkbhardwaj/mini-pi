from __future__ import annotations

from mini_pi.agent.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def test_message_creation_and_properties():
    msg = Message.user("Hello world")
    assert msg.role == "user"
    assert msg.text == "Hello world"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextBlock)


def test_thinking_block_creation_and_properties():
    asst = Message.assistant(text="Final answer", thinking="Reasoning about the problem...")
    assert asst.text == "Final answer"
    assert asst.thinking == "Reasoning about the problem..."
    assert len(asst.content) == 2
    assert isinstance(asst.content[0], ThinkingBlock)
    assert isinstance(asst.content[1], TextBlock)

    json_str = asst.model_dump_json()
    loaded = Message.model_validate_json(json_str)
    assert loaded.thinking == "Reasoning about the problem..."
    assert loaded.text == "Final answer"


def test_tool_use_and_result_blocks():
    call_block = ToolUseBlock(id="call_1", name="read", input={"path": "README.md"})
    asst_msg = Message.assistant(text="Let me check", tool_calls=[call_block])
    assert asst_msg.role == "assistant"
    assert asst_msg.text == "Let me check"
    assert len(asst_msg.tool_calls) == 1
    assert asst_msg.tool_calls[0].name == "read"

    res_msg = Message.tool_result(tool_use_id="call_1", content="File content here")
    assert res_msg.role == "tool"
    assert len(res_msg.tool_results) == 1
    assert res_msg.tool_results[0].content == "File content here"
    assert not res_msg.tool_results[0].is_error


def test_serialization_and_deserialization():
    orig = Message(
        role="assistant",
        content=[
            TextBlock(text="Thinking..."),
            ToolUseBlock(id="call_123", name="bash", input={"command": "ls -la"}),
        ],
    )
    json_data = orig.model_dump_json()
    loaded = Message.model_validate_json(json_data)

    assert loaded.id == orig.id
    assert loaded.role == orig.role
    assert len(loaded.content) == 2
    assert isinstance(loaded.content[0], TextBlock)
    assert isinstance(loaded.content[1], ToolUseBlock)
    assert loaded.content[1].name == "bash"
    assert loaded.content[1].input == {"command": "ls -la"}

