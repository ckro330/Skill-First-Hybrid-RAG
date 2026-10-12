from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from tools.path_guard import resolve_file_path


MAX_OUTPUT_CHARS = 12000
LEVEL_ALIASES: dict[str, str] = {
    "TRACE": r"TRACE",
    "DEBUG": r"DEBUG|DBG",
    "INFO": r"INFO|NOTICE",
    "WARN": r"WARN(?:ING)?",
    "ERROR": r"ERROR|ERR",
    "FATAL": r"FATAL|CRITICAL|CRIT",
}


class LogSearchInput(BaseModel):
    path: str = Field(..., description="项目内相对日志文件路径，如 storage/logs/app.log")
    pattern: str | None = Field(None, description="额外内容过滤：正则表达式")
    level: str | None = Field(
        None,
        description="日志级别过滤：TRACE/DEBUG/INFO/WARN/ERROR/FATAL（大小写不敏感）",
    )
    case_sensitive: bool = Field(False, description="是否区分大小写")
    context_lines: int = Field(1, ge=0, le=5, description="每个命中前后附带的上下文行数")
    max_results: int = Field(100, ge=1, le=1000, description="最多返回的命中条数")
    tail_lines: int | None = Field(
        None,
        ge=1,
        le=500000,
        description="只扫描文件末尾 N 行；缺省扫描整个文件",
    )


class LogSearchTool(BaseTool):
    name: str = "log_search"
    description: str = (
        "在单个日志文件里按级别和/或正则过滤，返回带行号的命中行及上下文。"
        "适合定位报错、异常堆栈、特定请求链路；大文件可用 tail_lines 只看尾部。"
        "只读操作，不会修改文件。"
    )
    args_schema: Type[BaseModel] = LogSearchInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _run(
        self,
        path: str,
        pattern: str | None = None,
        level: str | None = None,
        case_sensitive: bool = False,
        context_lines: int = 1,
        max_results: int = 100,
        tail_lines: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            file_path = resolve_file_path(self._root_dir, path)
        except ValueError as exc:
            return f"log_search 失败：{exc}"
        if not file_path.exists():
            return f"log_search 失败：文件不存在 {path}"

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"log_search 失败：{exc}"

        flags = 0 if case_sensitive else re.IGNORECASE
        matchers = []
        if level:
            level_key = level.strip().upper()
            expression = LEVEL_ALIASES.get(level_key, re.escape(level_key))
            try:
                matchers.append(re.compile(rf"\b(?:{expression})\b", flags))
            except re.error:
                return f"log_search 失败：无法识别的日志级别 {level}"
        if pattern:
            try:
                matchers.append(re.compile(pattern, flags))
            except re.error as exc:
                return f"log_search 失败：正则表达式非法（{exc}）"

        lines = text.splitlines()
        offset = 0
        if tail_lines and len(lines) > tail_lines:
            offset = len(lines) - tail_lines
            lines = lines[offset:]

        if not matchers:
            tail = lines[-min(len(lines), 50):]
            if not tail:
                return "log_search：文件为空。"
            numbered = "\n".join(
                f"{offset + i + 1}: {line}" for i, line in enumerate(tail)
            )
            return f"未提供 level/pattern，返回末尾 {len(tail)} 行：\n{numbered}"

        blocks: list[str] = []
        hits = 0
        for index, line in enumerate(lines):
            if not all(matcher.search(line) for matcher in matchers):
                continue
            hits += 1
            if context_lines:
                low = max(0, index - context_lines)
                high = min(len(lines), index + context_lines + 1)
                for ctx in range(low, high):
                    marker = ":" if ctx == index else "-"
                    blocks.append(f"{offset + ctx + 1}{marker} {lines[ctx]}")
                blocks.append("--")
            else:
                blocks.append(f"{offset + index + 1}: {line}")
            if hits >= max_results:
                break

        header = f"文件 {path}：扫描 {len(lines)} 行，命中 {hits} 处（上限 {max_results}）。"
        if not blocks:
            return f"{header}\n未找到匹配。"
        body = "\n".join(blocks)
        if len(body) > MAX_OUTPUT_CHARS:
            body = body[:MAX_OUTPUT_CHARS] + "\n...[truncated]"
        return f"{header}\n{body}"

    async def _arun(
        self,
        path: str,
        pattern: str | None = None,
        level: str | None = None,
        case_sensitive: bool = False,
        context_lines: int = 1,
        max_results: int = 100,
        tail_lines: int | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._run,
            path,
            pattern,
            level,
            case_sensitive,
            context_lines,
            max_results,
            tail_lines,
            None,
        )
