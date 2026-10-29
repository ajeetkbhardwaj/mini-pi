from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from mini_pi.tools.registry import Tool, ToolContext


def truncate(text: str, limit: int = 20_000) -> str:
    if len(text) <= limit:
        return text
    truncated_count = len(text) - limit
    return text[:limit] + f"\n... [{truncated_count} characters truncated]"


class BashArgs(BaseModel):
    command: str = Field(description="Shell command to run in the project directory")
    timeout_seconds: float = Field(
        default=60.0, gt=0, le=600, description="Execution timeout in seconds"
    )


class BashTool(Tool):
    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return (
            "Run a shell command in the project directory. "
            "Returns stdout, stderr, and exit code."
        )

    def args_model(self) -> type[BaseModel]:
        return BashArgs

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        data = BashArgs.model_validate(args)

        try:
            process = await asyncio.create_subprocess_shell(
                data.command,
                cwd=str(ctx.cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=data.timeout_seconds,
            )
            returncode = process.returncode
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return f"Command timed out after {data.timeout_seconds} seconds."
        except Exception as e:
            return f"Failed to execute command: {type(e).__name__}: {e}"

        stdout_text = stdout_bytes.decode(encoding="utf-8", errors="replace")
        stderr_text = stderr_bytes.decode(encoding="utf-8", errors="replace")

        output = (
            f"Exit code: {returncode}\n"
            f"STDOUT:\n{stdout_text}\n"
            f"STDERR:\n{stderr_text}"
        )

        return truncate(output)

