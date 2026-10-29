from __future__ import annotations

from pathlib import Path


def get_skills_dirs(cwd: Path | None = None) -> list[Path]:
    dirs = [
        Path.home() / ".mini-pi" / "skills",
    ]
    if cwd:
        dirs.extend([
            cwd / "skills",
            cwd / ".mini-pi" / "skills",
        ])
    return dirs


def list_skills(cwd: Path | None = None) -> dict[str, str]:
    """Scan for available skills and return mapping of name -> SKILL.md content."""
    skills: dict[str, str] = {}
    for base_dir in get_skills_dirs(cwd):
        if base_dir.is_dir():
            for skill_dir in base_dir.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.is_file():
                        try:
                            skills[skill_dir.name] = skill_file.read_text(encoding="utf-8")
                        except Exception:
                            pass
    return skills


def load_skill(name: str, cwd: Path | None = None) -> str:
    """Load the content of a specific skill by name."""
    skills = list_skills(cwd)
    clean_name = name.removeprefix("/skill:").removeprefix("skill:").strip()
    if clean_name not in skills:
        raise KeyError(f"Unknown skill '{clean_name}'. Available: {list(skills.keys())}")
    return skills[clean_name]


def get_skill_summary(cwd: Path | None = None) -> list[dict[str, str]]:
    """Return catalog of available skills with summary for prompt injection."""
    catalog: list[dict[str, str]] = []
    for name, content in list_skills(cwd).items():
        first_line = "No description"
        for line in content.splitlines():
            line_s = line.strip("# \t")
            if line_s:
                first_line = line_s
                break
        catalog.append({"name": name, "summary": first_line})
    return catalog

