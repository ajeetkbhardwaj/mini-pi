from __future__ import annotations

from pathlib import Path

from mini_pi.skills.loader import list_skills, load_skill
from mini_pi.skills.templates import list_templates, load_template


def test_skills_loader(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "deploy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Deploy Workflow\n1. Deploy app", encoding="utf-8")

    skills = list_skills(tmp_path)
    assert "deploy" in skills
    assert "Deploy Workflow" in load_skill("deploy", tmp_path)


def test_templates_loader(tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "explain.md").write_text("Explain this repository in depth.", encoding="utf-8")

    templates = list_templates(tmp_path)
    assert "explain" in templates
    assert "Explain this repository" in load_template("explain", tmp_path)

