from __future__ import annotations

from pathlib import Path

from mini_pi.agent.types import Message


class JsonlSessionStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, message: Message) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(message.model_dump_json() + "\n")

    def load(self) -> list[Message]:
        if not self.path.exists():
            return []

        messages: list[Message] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line_str = line.strip()
                if line_str:
                    try:
                        messages.append(Message.model_validate_json(line_str))
                    except Exception:
                        pass

        return messages

    def save_all(self, messages: list[Message]) -> None:
        """Overwrite the session store with a new message list."""
        with self.path.open("w", encoding="utf-8") as file:
            for message in messages:
                file.write(message.model_dump_json() + "\n")

    def clear(self) -> None:
        if self.path.exists():
            self.path.write_text("", encoding="utf-8")

