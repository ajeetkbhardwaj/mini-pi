from __future__ import annotations

from pathlib import Path

PROTECTED_NAMES = {".git", ".env", "node_modules", "id_rsa", "id_ed25519"}


def validate_workspace_path(
    cwd: Path,
    path_str: str,
    allow_sensitive: bool = False,
    allow_outside: bool = False,
) -> Path:
    """Validate that path_str stays within workspace and does not touch protected files."""
    resolved_cwd = cwd.resolve()
    target = (cwd / path_str).resolve()

    if not allow_outside:
        try:
            rel = target.relative_to(resolved_cwd)
        except ValueError:
            raise PermissionError(
                f"Path '{path_str}' escapes workspace directory '{cwd}'"
            )

        if not allow_sensitive:
            for part in rel.parts:
                if part in PROTECTED_NAMES or part.startswith(".env."):
                    raise PermissionError(
                        f"Access to protected path '{part}' is restricted"
                    )

    return target

