from __future__ import annotations

from pathlib import Path


def get_prompts_dirs(cwd: Path | None = None) -> list[Path]:
    dirs = [
        Path.home() / ".mini-pi" / "prompts",
    ]
    if cwd:
        dirs.extend([
            cwd / "prompts",
            cwd / ".mini-pi" / "prompts",
        ])
    return dirs


def list_templates(cwd: Path | None = None) -> dict[str, str]:
    """Scan and list prompt templates."""
    templates: dict[str, str] = {}
    for base_dir in get_prompts_dirs(cwd):
        if base_dir.is_dir():
            for file_path in base_dir.glob("*.md"):
                try:
                    templates[file_path.stem] = file_path.read_text(encoding="utf-8")
                except Exception:
                    pass
    return templates


def load_template(name: str, cwd: Path | None = None) -> str:
    """Load a specific prompt template by name."""
    clean_name = name.removeprefix("/").strip()
    templates = list_templates(cwd)
    if clean_name not in templates:
        raise FileNotFoundError(
            f"Unknown template '/{clean_name}'. Available: {list(templates.keys())}"
        )
    return templates[clean_name]

