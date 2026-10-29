from __future__ import annotations

from mini_pi.agent.types import Message


class SessionTree:
    def __init__(self) -> None:
        self.messages: dict[str, Message] = {}
        self.active_leaf_id: str | None = None

    def append(self, message: Message) -> Message:
        if message.parent_id is None and self.active_leaf_id is not None:
            message.parent_id = self.active_leaf_id

        self.messages[message.id] = message
        self.active_leaf_id = message.id
        return message

    def switch_to(self, message_id: str) -> None:
        if message_id not in self.messages:
            raise KeyError(f"Unknown message id: {message_id}")
        self.active_leaf_id = message_id

    def active_path(self) -> list[Message]:
        path: list[Message] = []
        cursor = self.active_leaf_id

        visited = set()
        while cursor and cursor not in visited:
            visited.add(cursor)
            message = self.messages.get(cursor)
            if not message:
                break
            path.append(message)
            cursor = message.parent_id

        return list(reversed(path))

    def leaves(self) -> list[str]:
        parent_ids = {m.parent_id for m in self.messages.values() if m.parent_id}
        return [mid for mid in self.messages if mid not in parent_ids]

    def children(self, message_id: str) -> list[Message]:
        return [m for m in self.messages.values() if m.parent_id == message_id]

    @classmethod
    def from_messages(cls, messages: list[Message]) -> SessionTree:
        tree = cls()
        for msg in messages:
            tree.messages[msg.id] = msg
            tree.active_leaf_id = msg.id
        return tree

