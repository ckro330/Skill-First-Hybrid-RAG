from __future__ import annotations

from pathlib import Path

from config import get_settings

# system prompt 组件按「变化频率从低到高」排列：越稳定越靠前，越易变越靠后。
# 上下文缓存按前缀精确匹配，前面只要有一处变化，它之后的 token 全部失效，
# 因此几乎不变的 workspace 文件放在最前，最易变的技能快照放到最后。
SYSTEM_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("Identity", "workspace/IDENTITY.md"),
    ("Soul", "workspace/SOUL.md"),
    ("User Profile", "workspace/USER.md"),
    ("Agents Guide", "workspace/AGENTS.md"),
)

# 易变组件：SKILLS_SNAPSHOT.md 在每次技能增删改时都会被 refresh_snapshot 重写，
# 单独放到 system 末尾渲染，保证它变化时前面的恒定内容仍然命中缓存。
VOLATILE_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("Skills Snapshot", "SKILLS_SNAPSHOT.md"),
)

RUNTIME_OVERRIDE = """<!-- Runtime Override -->
When explicit retrieval evidence is provided for the current request, prioritize that evidence.
Do not assume missing evidence exists elsewhere.
"""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _read_component(base_dir: Path, relative_path: str, limit: int) -> str:
    path = base_dir / relative_path
    if not path.exists():
        return f"[missing component: {relative_path}]"
    return _truncate(path.read_text(encoding="utf-8"), limit)


def build_system_prompt(base_dir: Path) -> str:
    settings = get_settings()
    parts: list[str] = []

    for label, relative_path in SYSTEM_COMPONENTS:
        content = _read_component(base_dir, relative_path, settings.component_char_limit)
        parts.append(f"<!-- {label} -->\n{content}")

    parts.append(
        "<!-- Long-term Memory -->\n"
        "长期记忆存储规则：\n"
        "1. 当用户明确要求记住、表达持久偏好/身份/习惯，"
        "或对话中出现值得跨任务复用的经验、方法论时，"
        "使用 add_longterm_memory 工具写入长期记忆。\n"
        "2. importance 标准（0~1）：\n"
        "   - 0.9~1.0：用户明确要求记住的信息、身份/偏好声明、长期承诺\n"
        "   - 0.75~0.9：可复用的方法论、经验教训、跨任务有价值的偏好或事实\n"
        "   - 0.5~0.75：仅对当前任务有参考价值的细节\n"
        "   - 0.2~0.5：一次性事实、临时请求\n"
        "   - 0~0.2：寒暄、无关闲聊\n"
        "3. 只有 importance > 0.7 时才值得写入；"
        "正文要精炼、自包含、可复用，避免记录琐碎或一次性内容。\n"
        "4. 长期记忆由 MemoryManager 按需检索动态注入。"
        "你应优先使用当次检索到的记忆片段，不要假设未检索到的记忆仍然有效。"
    )
    parts.append(RUNTIME_OVERRIDE)

    for label, relative_path in VOLATILE_COMPONENTS:
        content = _read_component(base_dir, relative_path, settings.component_char_limit)
        parts.append(f"<!-- {label} -->\n{content}")

    return "\n\n".join(parts)
