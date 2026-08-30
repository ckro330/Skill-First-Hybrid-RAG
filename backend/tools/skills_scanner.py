from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


@dataclass
class SkillRecord:
    name: str
    description: str
    path: str
    negative_triggers: str = ""

    @property
    def id(self) -> str:
        """技能 ID，即技能目录名，例如 skills/web-search/SKILL.md -> web-search。"""
        return Path(self.path).parent.name


def _parse_frontmatter(text: str) -> dict[str, Any]:
    match = FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}
    data = yaml.safe_load(match.group(1)) or {}
    return {str(key): value for key, value in data.items()}


def _stringify(value: Any) -> str:
    """把 frontmatter 字段规范成字符串，支持单行字符串或 YAML 列表。"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "；".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "")


def scan_skills(base_dir: Path) -> list[SkillRecord]:
    skills_dir = base_dir / "skills"
    records: list[SkillRecord] = []
    if not skills_dir.exists():
        return records

    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8")
        meta = _parse_frontmatter(text)
        records.append(
            SkillRecord(
                name=str(meta.get("name", skill_file.parent.name)),
                description=str(meta.get("description", "No description")),
                path=str(skill_file.relative_to(base_dir)).replace("\\", "/"),
                negative_triggers=_stringify(meta.get("negative_triggers", "")),
            )
        )
    return records


def build_snapshot(skills: list[SkillRecord]) -> str:
    lines = [
        "<skills>",
        "  <summary>Available local skills that the agent can inspect with read_file.</summary>",
    ]
    for skill in skills:
        lines.extend(
            [
                f'  <skill name="{skill.name}" path="{skill.path}">',
                f"    <description>{skill.description}</description>",
                "  </skill>",
            ]
        )
    lines.append("</skills>")
    return "\n".join(lines) + "\n"


def refresh_snapshot(base_dir: Path) -> Path:
    snapshot_path = base_dir / "SKILLS_SNAPSHOT.md"
    snapshot_path.write_text(
        build_snapshot(scan_skills(base_dir)),
        encoding="utf-8",
    )
    return snapshot_path
