from __future__ import annotations

from pathlib import Path
import pytest

from mini_pi.extensions.api import ExtensionAPI
from mini_pi.extensions.safety import register as register_safety


@pytest.mark.asyncio
async def test_extension_api_events():
    api = ExtensionAPI()
    received = []

    def handler(payload, ctx):
        received.append(payload)
        return "handled"

    api.on("custom_event", handler)
    result = await api.emit("custom_event", {"status": "ok"})

    assert len(received) == 1
    assert received[0] == {"status": "ok"}
    assert result == "handled"


@pytest.mark.asyncio
async def test_safety_gate_denial():
    api = ExtensionAPI()
    # Configure confirmation callback to reject dangerous commands
    api.context.confirm_callback = lambda title, msg: False

    await register_safety(api, auto_allow=False)

    # Benign command
    res_benign = await api.emit(
        "tool_call",
        {"tool": "bash", "input": {"command": "ls -l"}},
    )
    assert res_benign is None

    # Dangerous command
    res_danger = await api.emit(
        "tool_call",
        {"tool": "bash", "input": {"command": "rm -rf /tmp/test"}},
    )
    assert res_danger is not None
    assert res_danger.get("block") is True
    assert "Execution blocked" in res_danger.get("reason", "")

