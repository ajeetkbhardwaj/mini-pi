from __future__ import annotations

from typing import Any

from mini_pi.extensions.api import ExtensionAPI, ExtensionContext

DANGEROUS_PATTERNS = (
    "rm -rf",
    "sudo ",
    "git push --force",
    "git push -f",
    "git reset --hard",
    "chmod -R 777",
    "mkfs",
    "> /dev/",
)


async def register(pi: ExtensionAPI, auto_allow: bool = False) -> None:
    async def gate(payload: dict[str, Any], ctx: ExtensionContext) -> dict[str, Any] | None:
        tool_name = payload.get("tool")
        args = payload.get("input", {})

        if tool_name == "bash":
            command = args.get("command", "")
            for pattern in DANGEROUS_PATTERNS:
                if pattern in command:
                    if auto_allow:
                        return None
                    allowed = await ctx.confirm(
                        "Potentially Destructive Command",
                        f"Command contains dangerous pattern '{pattern}':\n  {command}",
                    )
                    if not allowed:
                        return {
                            "block": True,
                            "reason": f"Execution blocked: user denied '{pattern}' in bash command",
                        }

        return None

    pi.on("tool_call", gate)

