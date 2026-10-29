from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mini_pi.tools.registry import Tool


@dataclass
class ExtensionContext:
    cwd: Path
    confirm_callback: Callable[[str, str], bool] | None = None

    async def confirm(self, title: str, message: str) -> bool:
        if self.confirm_callback:
            return self.confirm_callback(title, message)
        try:
            answer = input(f"\n[SECURITY CONFIRMATION]\n{title}\n{message}\nAllow execution? [y/N]: ")
            return answer.strip().lower() in {"y", "yes"}
        except (EOFError, KeyboardInterrupt):
            return False


class ExtensionAPI:
    def __init__(self, cwd: Path | None = None) -> None:
        self.cwd = cwd or Path.cwd()
        self.handlers: dict[str, list[Callable]] = {}
        self.tools: list[Tool] = []
        self.commands: dict[str, Callable] = {}
        self.context = ExtensionContext(cwd=self.cwd)

    def on(self, event: str, handler: Callable) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def register_tool(self, tool: Tool) -> None:
        self.tools.append(tool)

    def register_command(self, name: str, handler: Callable) -> None:
        self.commands[name.lstrip("/")] = handler

    async def emit(self, event: str, payload: dict[str, Any]) -> Any:
        result = None
        for handler in self.handlers.get(event, []):
            if inspect.iscoroutinefunction(handler):
                res = await handler(payload, self.context)
            else:
                res = handler(payload, self.context)
            if res is not None:
                result = res
        return result

