"""Skill loader. A skill is a folder in skills/ with a SKILL.md file plus optional resources.

Agents only see each skill's name and description up front. They call the `load_skill`
tool to read the full instructions when they need them (progressive disclosure).
"""

import importlib.util
import re
from functools import lru_cache
from types import ModuleType

from screener.config import SKILLS_DIR

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _frontmatter(text: str) -> dict[str, str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    pairs = (line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def list_skills() -> dict[str, str]:
    """Skill name -> description."""
    skills = {}
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        meta = _frontmatter(skill_md.read_text())
        skills[meta.get("name", skill_md.parent.name)] = meta.get("description", "")
    return skills


def skills_catalogue(names: list[str]) -> str:
    """Short catalogue for a system prompt."""
    available = list_skills()
    return "\n".join(f"- {name}: {available[name]}" for name in names if name in available)


def load_skill(name: str) -> str:
    """SKILL.md body plus any extra .md resources in the skill folder."""
    folder = SKILLS_DIR / name
    skill_md = folder / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError (f"Unknown skill '{name}'. Available: {', '.join(list_skills())}")
    parts = [FRONTMATTER_RE.sub("", skill_md.read_text())]
    for resource in sorted(folder.glob("*.md")):
        if resource.name != "SKILL.md":
            parts.append(f"\n## Resource: {resource.name}\n\n{resource.read_text()}")
    return "\n".join(parts)


@lru_cache
def load_skill_script(skill: str, module: str) -> ModuleType:
    """Import a Python script bundled with a skill, e.g. scoring-rubric/compute_score.py."""
    path = SKILLS_DIR / skill / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"skills.{skill}.{module}", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded
