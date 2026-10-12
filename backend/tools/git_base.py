from __future__ import annotations

import subprocess
from pathlib import Path


GIT_TIMEOUT_SECONDS = 20
MAX_OUTPUT_CHARS = 8000


def run_git(
    root_dir: Path,
    args: list[str],
    *,
    timeout: int = GIT_TIMEOUT_SECONDS,
    max_chars: int = MAX_OUTPUT_CHARS,
) -> str:
    """在项目目录内执行 git 子命令，返回 stdout + stderr 截断后的文本。"""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return "git 未安装或不在 PATH 中。"
    except subprocess.TimeoutExpired:
        return f"git 命令执行超时（>{timeout}s）。"

    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    if not output:
        output = f"[no output; exit code={completed.returncode}]"
    if len(output) > max_chars:
        output = output[:max_chars] + "\n...[truncated]"
    return output


def ensure_git_repo(root_dir: Path) -> str | None:
    """确认目录位于 git 仓库内；不在仓库内时返回错误文本，否则返回 None。"""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return "git 未安装或不在 PATH 中。"
    except subprocess.TimeoutExpired:
        return "检查 git 仓库状态超时。"
    if completed.returncode != 0 or (completed.stdout or "").strip() != "true":
        return "当前目录不在 git 仓库内，无法执行 git 操作。"
    return None


def is_safe_ref(value: str | None) -> bool:
    """拒绝以 - 开头的引用或分支名，避免被解析成 git 选项。"""
    if value is None:
        return True
    value = value.strip()
    return bool(value) and not value.startswith("-")
