from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class ToolContext:
    cwd: Path
    abort_event: Any | None = None


class Tool(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the tool as invoked by LLMs."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Documentation for the model explaining what the tool does and when to use it."""
        ...

    @abc.abstractmethod
    def args_model(self) -> type[BaseModel]:
        """Pydantic model representing expected arguments."""
        ...

    @abc.abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Execute the tool asynchronously and return text result."""
        ...

    def schema(self) -> dict[str, Any]:
        """JSON schema representation for model providers."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model().model_json_schema(),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: '{name}' (available: {list(self._tools.keys())})")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def execute(
        self, name: str, arguments: dict[str, Any], ctx: ToolContext
    ) -> str:
        tool = self.get(name)
        return await tool.run(arguments, ctx)

