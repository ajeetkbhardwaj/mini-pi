from __future__ import annotations

from pathlib import Path

BASE_SYSTEM_PROMPT = """\
You are a careful terminal coding agent.

Rules:
1. Read relevant files before editing them.
2. Make the smallest change that solves the task.
3. Use edit for existing files and write only for new files.
4. Run tests or relevant commands after meaningful changes.
5. Explain what changed and how you verified it.
6. Never invent file contents; inspect the repository first.
7. Ask before destructive or irreversible operations.
"""


def load_context_files(cwd: Path) -> str:
    """Load instructions from global and workspace AGENTS.md files."""
    parts: list[str] = []

    global_agents = Path.home() / ".mini-pi" / "AGENTS.md"
    project_agents = cwd / "AGENTS.md"

    for path in (global_agents, project_agents):
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
                parts.append(f"# Context file: {path.name}\n\n{text}")
            except Exception:
                pass

    return "\n\n".join(parts)


def build_system_prompt(cwd: Path, extra_instructions: str = "") -> str:
    """Build layered effective system prompt."""
    context = load_context_files(cwd)
    prompt_parts = [BASE_SYSTEM_PROMPT.strip()]

    if context:
        prompt_parts.append(context)

    if extra_instructions.strip():
        prompt_parts.append(f"# Additional Instructions\n\n{extra_instructions.strip()}")

    return "\n\n".join(prompt_parts)

