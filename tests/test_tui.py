from __future__ import annotations

from pathlib import Path
import pytest

from mini_pi.agent.loop import AgentConfig, AgentLoop
from mini_pi.agent.types import Message
from mini_pi.providers.local_provider import LocalOpenAIClient
from mini_pi.providers.mock_provider import MockModelClient
from mini_pi.session.store import JsonlSessionStore
from mini_pi.tools.registry import ToolRegistry
from mini_pi.tui.app import MiniPiApp, ModelSelectModal, SLASH_COMMANDS


@pytest.fixture
def tui_app(tmp_path: Path) -> MiniPiApp:
    model = LocalOpenAIClient(backend="lmstudio", model_id="local-model")
    tools = ToolRegistry()
    config = AgentConfig(system_prompt="Test TUI")
    agent = AgentLoop(model=model, tools=tools, config=config, cwd=tmp_path)
    store = JsonlSessionStore(tmp_path / "session.jsonl")
    initial_history = [Message.user("Hello"), Message.assistant("Hi! How can I help?")]
    return MiniPiApp(agent=agent, session_store=store, initial_history=initial_history)


def test_tui_initial_state(tui_app: MiniPiApp):
    assert len(tui_app.history) == 2
    assert tui_app.agent.model.model_id == "local-model"
    assert "/help" in SLASH_COMMANDS
    assert "/models" in SLASH_COMMANDS
    assert "/model " in SLASH_COMMANDS
    assert "/connect " in SLASH_COMMANDS


@pytest.mark.asyncio
async def test_tui_slash_commands_handling(tui_app: MiniPiApp):
    # Test model switching
    tui_app._switch_model("qwen2.5-coder:7b")
    assert tui_app.agent.model.model_id == "qwen2.5-coder:7b"

    # Test connect URL
    await tui_app._connect_url("http://localhost:1234/v1")
    assert tui_app.agent.model.base_url == "http://localhost:1234/v1"

    # Test switch provider
    await tui_app._switch_provider("ollama")
    assert tui_app.agent.model.backend == "ollama"
    assert tui_app.agent.model.base_url == "http://localhost:11434/v1"

    # Test compact command
    await tui_app._handle_slash_command("/compact")
    assert len(tui_app.history) > 0

    # Test clear session command
    await tui_app._handle_slash_command("/clear-session")
    assert len(tui_app.history) == 0

    # Test tokens command
    tui_app.total_in = 100
    tui_app.total_out = 50
    tui_app.total_think = 35
    await tui_app._handle_slash_command("/tokens")
    assert tui_app.total_think == 35


def test_model_select_modal_instantiation():
    modal = ModelSelectModal("lmstudio", "local-model", "http://localhost:1234/v1")
    assert modal.current_provider == "lmstudio"
    assert modal.current_model == "local-model"
    assert modal.current_url == "http://localhost:1234/v1"

