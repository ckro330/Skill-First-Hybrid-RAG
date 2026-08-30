"""一次性迁移 backend/memory/MEMORY.md 到 PostgreSQL/pgvector 长期记忆库。

用法（在 backend 目录下执行）：
    python -m scripts.import_memory_md            # 迁移到数据库
    python -m scripts.import_memory_md --dry-run  # 只打印解析结果，不写库

需要先配置 DATABASE_URL 并启动 PostgreSQL（会自动建表 + pgvector 扩展）。
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config import get_settings  # noqa: E402
from memory.memory_manager import MemoryManager  # noqa: E402

SECTION_RE = re.compile(r"^##\s+(.*)$")
SUBSECTION_RE = re.compile(r"^###\s+(.*)$")
BULLET_RE = re.compile(r"^[-*]\s+(.*)$")


def parse_memory_md(text: str) -> list[dict]:
    """把 MEMORY.md 解析为记忆记录列表。

    - 项目背景 -> profile
    - 经验教训（### 小节） -> episodic
    - 用户偏好（- 条目） -> preference
    """
    records: list[dict] = []
    section: str | None = None
    profile_lines: list[str] = []
    lesson_title: str | None = None
    lesson_lines: list[str] = []

    def flush_lesson() -> None:
        nonlocal lesson_title, lesson_lines
        if lesson_title is None:
            return
        body = "\n".join(lesson_lines).strip()
        if body:
            records.append(
                {
                    "content": f"{lesson_title}\n{body}",
                    "type": "episodic",
                    "importance": 0.6,
                }
            )
        lesson_title = None
        lesson_lines = []

    def flush_profile() -> None:
        nonlocal profile_lines
        body = "\n".join(profile_lines).strip()
        if body:
            records.append(
                {"content": body, "type": "profile", "importance": 0.7}
            )
        profile_lines = []

    for raw in text.splitlines():
        line = raw.rstrip()
        sec_match = SECTION_RE.match(line)
        sub_match = SUBSECTION_RE.match(line)
        if sec_match:
            flush_lesson()
            flush_profile()
            section = sec_match.group(1).strip()
            continue
        if sub_match:
            flush_lesson()
            lesson_title = sub_match.group(1).strip()
            continue
        if section == "项目背景":
            profile_lines.append(line)
            continue
        if section == "经验教训":
            if lesson_title is not None:
                lesson_lines.append(line)
            continue
        if section == "用户偏好":
            bullet = BULLET_RE.match(line)
            if bullet:
                content = bullet.group(1).strip()
                if content:
                    records.append(
                        {
                            "content": content,
                            "type": "preference",
                            "importance": 0.8,
                        }
                    )
            continue

    flush_lesson()
    flush_profile()
    return records


async def import_memories(
    memory_path: Path,
    user_id: str,
    dry_run: bool,
) -> int:
    text = memory_path.read_text(encoding="utf-8")
    records = parse_memory_md(text)

    if dry_run:
        for idx, record in enumerate(records, start=1):
            preview = record["content"].replace("\n", " ")[:120]
            print(
                f"[{idx}] ({record['type']}, importance={record['importance']}) {preview}"
            )
        print(f"\n共解析出 {len(records)} 条记忆")
        return 0

    if not records:
        print("未解析出任何记忆")
        return 0

    manager = MemoryManager(backend_dir)
    manager.initialize()
    if not manager.is_available:
        print(
            "PostgreSQL 不可用：请先配置 DATABASE_URL 并启动 PostgreSQL，再运行导入。"
        )
        return 1

    ok = 0
    for record in records:
        result = await manager.add_memory(user_id=user_id, **record)
        if result is not None:
            ok += 1
            print(f"[{ok}/{len(records)}] 已写入 {result.id} ({result.type})")
        else:
            print(f"写入失败: {record['content'][:80]}")

    print(f"导入完成：成功 {ok} / 共 {len(records)} 条")
    return 0 if ok == len(records) else 1


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="迁移 MEMORY.md 到长期记忆库")
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=backend_dir / "memory" / "MEMORY.md",
        help="MEMORY.md 路径（默认 backend/memory/MEMORY.md）",
    )
    parser.add_argument(
        "--user-id",
        default=settings.memory_default_user_id,
        help=f"写入的 user_id（默认 {settings.memory_default_user_id}）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印解析结果，不写数据库",
    )
    args = parser.parse_args()
    return asyncio.run(import_memories(args.memory_path, args.user_id, args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
