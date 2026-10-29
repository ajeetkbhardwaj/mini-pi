from __future__ import annotations

from pathlib import Path
import pytest

from mini_pi.agent.types import Message
from mini_pi.session.compact import compact_history
from mini_pi.session.store import JsonlSessionStore
from mini_pi.session.tree import SessionTree


def test_session_store(tmp_path: Path):
    store_file = tmp_path / "session.jsonl"
    store = JsonlSessionStore(store_file)

    msg1 = Message.user("First turn")
    msg2 = Message.assistant("First answer")

    store.append(msg1)
    store.append(msg2)

    loaded = store.load()
    assert len(loaded) == 2
    assert loaded[0].text == "First turn"
    assert loaded[1].text == "First answer"

    store.clear()
    assert len(store.load()) == 0


def test_session_tree_branching():
    tree = SessionTree()

    # Turn 1
    m1 = tree.append(Message.user("Hello"))
    # Turn 2 (Branch A)
    m2 = tree.append(Message.assistant("A1"))
    m3 = tree.append(Message.assistant("A2"))

    # Active path should be m1 -> m2 -> m3
    path_a = tree.active_path()
    assert [m.id for m in path_a] == [m1.id, m2.id, m3.id]

    # Switch back to m1 and create Branch B
    tree.switch_to(m1.id)
    m4 = tree.append(Message.assistant("B1"))

    path_b = tree.active_path()
    assert [m.id for m in path_b] == [m1.id, m4.id]

    # Switch back to Branch A
    tree.switch_to(m3.id)
    assert [m.id for m in tree.active_path()] == [m1.id, m2.id, m3.id]


@pytest.mark.asyncio
async def test_compact_history():
    messages = [
        Message.user(f"Turn {i}") if i % 2 == 1 else Message.assistant(f"Ans {i}")
        for i in range(1, 15)
    ]
    assert len(messages) == 14

    compacted = await compact_history(messages, keep_recent=4)
    assert len(compacted) == 5  # 1 summary + 4 recent
    assert "Earlier session summary" in compacted[0].text
    assert compacted[-1].text == "Ans 14"

