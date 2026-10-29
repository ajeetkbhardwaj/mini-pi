from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mini_pi.tools.path_guard import validate_workspace_path
from mini_pi.tools.registry import Tool, ToolContext


class ReadArgs(BaseModel):
    path: str = Field(description="File path, relative to the project root")
    offset: int | None = Field(
        default=None, ge=1, description="Optional 1-based start line number"
    )
    limit: int | None = Field(
        default=None, gt=0, description="Optional maximum number of lines to read"
    )


class ReadTool(Tool):
    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return (
            "Read a UTF-8 text file. Returns numbered lines so later edits "
            "can reference exact text."
        )

    def args_model(self) -> type[BaseModel]:
        return ReadArgs

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        data = ReadArgs.model_validate(args)
        path = validate_workspace_path(ctx.cwd, data.path, allow_sensitive=False)

        if not path.exists():
            raise FileNotFoundError(f"No such file: {data.path}")
        if path.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {data.path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        start = (data.offset - 1) if data.offset else 0
        end = (start + data.limit) if data.limit else len(lines)
        selected_lines = lines[start:end]

        return "\n".join(
            f"{number:5d} | {line}"
            for number, line in enumerate(selected_lines, start=start + 1)
        )

