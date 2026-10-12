from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from tools.path_guard import PROTECTED_DIR_NAMES


MAX_OUTPUT_CHARS = 12000
MAX_FILE_BYTES = 2_000_000
MAX_RESULTS_HARD_LIMIT = 1000


class GrepInput(BaseModel):
    pattern: str = Field(..., min_length=1, description="搜索模式，默认按正则表达式解释")
    path: str = Field(".", description="搜索起点：项目内的相对文件或目录路径")
    glob: str | None = Field(None, description='限定文件名通配符，如 "*.py"、"SKILL.md"')
    fixed_string: bool = Field(False, description="把 pattern 当作普通字符串而不是正则")
    case_sensitive: bool = Field(False, description="是否区分大小写")
    context_lines: int = Field(0, ge=0, le=5, description="每个命中前后附带的上下文行数")
    max_results: int = Field(100, ge=1, le=MAX_RESULTS_HARD_LIMIT, description="最多返回的命中条数")


class GrepTool(BaseTool):
    name: str = "grep"
    description: str = (
        "在项目内按正则或普通字符串搜索文件内容，返回 文件:行号:内容。"
        "支持 glob 过滤文件名、context_lines 查看上下文、max_results 限制条数。"
        "自动跳过 .git/.venv/node_modules/__pycache__ 等目录与二进制文件。"
        "适合跨文件定位函数定义、报错关键字、配置项，避免整文件读取。"
    )
    args_schema: Type[BaseModel] = GrepInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _resolve(self, path: str) -> Path:
        normalized = (path or ".").replace("\\", "/").strip() or "."
        candidate = (self._root_dir / normalized).resolve()
        if self._root_dir not in candidate.parents and candidate != self._root_dir:
            raise ValueError("Path traversal detected.")
        return candidate

    def _iter_files(self, start: Path, glob_pattern: str | None) -> Iterator[Path]:
        if start.is_file():
            yield start
            return
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(
                name for name in dirnames if name.lower() not in PROTECTED_DIR_NAMES
            )
            for filename in sorted(filenames):
                if glob_pattern and not fnmatch.fnmatch(filename, glob_pattern):
                    continue
                yield Path(dirpath) / filename

    def _run(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        fixed_string: bool = False,
        case_sensitive: bool = False,
        context_lines: int = 0,
        max_results: int = 100,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            start = self._resolve(path)
        except ValueError as exc:
            return f"grep 失败：{exc}"
        if not start.exists():
            return f"grep 失败：路径不存在 {path}"

        if fixed_string:
            needle = pattern if case_sensitive else pattern.lower()

            def matches(line: str) -> bool:
                haystack = line if case_sensitive else line.lower()
                return needle in haystack
        else:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as exc:
                return f"grep 失败：正则表达式非法（{exc}）"

            def matches(line: str) -> bool:
                return regex.search(line) is not None

        blocks: list[str] = []
        hits = 0
        scanned = 0
        for file_path in self._iter_files(start, glob):
            try:
                if file_path.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "\x00" in text[:4096]:
                continue
            scanned += 1
            lines = text.splitlines()
            relative = file_path.relative_to(self._root_dir).as_posix()
            for index, line in enumerate(lines):
                if not matches(line):
                    continue
                hits += 1
                if context_lines:
                    low = max(0, index - context_lines)
                    high = min(len(lines), index + context_lines + 1)
                    for ctx in range(low, high):
                        marker = ":" if ctx == index else "-"
                        blocks.append(f"{relative}{marker}{ctx + 1}{marker} {lines[ctx]}")
                    blocks.append("--")
                else:
                    blocks.append(f"{relative}:{index + 1}: {line}")
                if hits >= max_results:
                    break
            if hits >= max_results:
                break

        header = f"扫描 {scanned} 个文件，命中 {hits} 处（上限 {max_results}，模式={'固定字符串' if fixed_string else '正则'}）。"
        if not blocks:
            return f"{header}\n未找到匹配。"
        body = "\n".join(blocks)
        if len(body) > MAX_OUTPUT_CHARS:
            body = body[:MAX_OUTPUT_CHARS] + "\n...[truncated]"
        return f"{header}\n{body}"

    async def _arun(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        fixed_string: bool = False,
        case_sensitive: bool = False,
        context_lines: int = 0,
        max_results: int = 100,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._run,
            pattern,
            path,
            glob,
            fixed_string,
            case_sensitive,
            context_lines,
            max_results,
            None,
        )
