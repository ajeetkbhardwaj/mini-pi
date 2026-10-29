from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mini_pi.tools.path_guard import validate_workspace_path
from mini_pi.tools.registry import Tool, ToolContext


class WriteArgs(BaseModel):
    path: str = Field(description="File path, relative to the project root")
    content: str = Field(description="Full text content to write to the file")


class WriteTool(Tool):
    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return "Create a new file or completely replace an existing file."

    def args_model(self) -> type[BaseModel]:
        return WriteArgs

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        data = WriteArgs.model_validate(args)
        path = validate_workspace_path(ctx.cwd, data.path, allow_sensitive=False)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data.content, encoding="utf-8")
        return f"Wrote {data.path} ({len(data.content)} characters)"

