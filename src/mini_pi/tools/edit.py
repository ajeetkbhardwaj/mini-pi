from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mini_pi.tools.path_guard import validate_workspace_path
from mini_pi.tools.registry import Tool, ToolContext


class EditArgs(BaseModel):
    path: str = Field(description="File path, relative to the project root")
    old_text: str = Field(min_length=1, description="Exact text block to replace")
    new_text: str = Field(description="Replacement text block")


class EditTool(Tool):
    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return (
            "Replace an exact, unique block of text in an existing file. "
            "old_text must match exactly."
        )

    def args_model(self) -> type[BaseModel]:
        return EditArgs

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        data = EditArgs.model_validate(args)
        path = validate_workspace_path(ctx.cwd, data.path, allow_sensitive=False)

        if not path.is_file():
            raise FileNotFoundError(f"No such file: {data.path}")

        original = path.read_text(encoding="utf-8")
        occurrences = original.count(data.old_text)

        if occurrences == 0:
            raise ValueError(f"old_text not found in {data.path}")
        if occurrences > 1:
            raise ValueError(
                f"old_text matched {occurrences} times in {data.path}; "
                "include more surrounding context to disambiguate"
            )

        updated = original.replace(data.old_text, data.new_text, 1)
        path.write_text(updated, encoding="utf-8")

        return f"Edited {data.path}"

