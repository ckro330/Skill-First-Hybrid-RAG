from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.skills_scanner import FRONTMATTER_PATTERN, _parse_frontmatter


@dataclass
class LoadedSkill:
    """加载后的 Skill：frontmatter 元数据 + SKILL.md 正文（Instructions）。"""

    skill_id: str
    name: str
    description: str
    instructions: str
    full_text: str


class SkillLoader:
    """进入 Skill 后加载 SKILL.md（方案第五、六节）。"""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def load(self, skill_id: str | None) -> LoadedSkill | None:
        if not skill_id:
            return None
        safe_id = Path(skill_id).name
        if not safe_id or safe_id in {".", ".."}:
            return None
        path = self.base_dir / "skills" / safe_id / "SKILL.md"
        if not path.exists():
            return None

        text = path.read_text(encoding="utf-8")
        meta: dict[str, str] = {}
        body = text
        match = FRONTMATTER_PATTERN.match(text)
        if match:
            parsed = _parse_frontmatter(text)
            meta = {str(key): str(value) for key, value in parsed.items()}
            body = text[match.end() :]

        return LoadedSkill(
            skill_id=safe_id,
            name=meta.get("name", safe_id),
            description=meta.get("description", ""),
            instructions=body.strip(),
            full_text=text,
        )
