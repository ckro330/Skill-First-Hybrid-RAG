from __future__ import annotations

import asyncio
import re
from collections import Counter
from pathlib import Path
from typing import Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from tools.path_guard import resolve_file_path


LEVEL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("FATAL", re.compile(r"\b(?:FATAL|CRITICAL|CRIT)\b", re.IGNORECASE)),
    ("ERROR", re.compile(r"\b(?:ERROR|ERR)\b", re.IGNORECASE)),
    ("WARN", re.compile(r"\bWARN(?:ING)?\b", re.IGNORECASE)),
    ("INFO", re.compile(r"\b(?:INFO|NOTICE)\b", re.IGNORECASE)),
    ("DEBUG", re.compile(r"\b(?:DEBUG|DBG)\b", re.IGNORECASE)),
    ("TRACE", re.compile(r"\bTRACE\b", re.IGNORECASE)),
)
ERROR_LEVELS = {"FATAL", "ERROR"}


class LogStatsInput(BaseModel):
    path: str = Field(..., description="项目内相对日志文件路径")
    top_errors: int = Field(10, ge=1, le=50, description="返回的错误指纹 Top N 数量")
    fingerprint_chars: int = Field(90, ge=20, le=200, description="每条错误指纹保留的字符数")


class LogStatsTool(BaseTool):
    name: str = "log_stats"
    description: str = (
        "统计日志文件的整体情况：总行数、各级别（FATAL/ERROR/WARN/INFO/DEBUG/TRACE）出现次数，"
        "以及 Top N 错误指纹（把时间戳、数字、十六进制等归一化后聚合）。"
        "适合先快速判断日志是否异常，再决定用 log_search 深入定位。只读操作。"
    )
    args_schema: Type[BaseModel] = LogStatsInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _run(
        self,
        path: str,
        top_errors: int = 10,
        fingerprint_chars: int = 90,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            file_path = resolve_file_path(self._root_dir, path)
        except ValueError as exc:
            return f"log_stats 失败：{exc}"
        if not file_path.exists():
            return f"log_stats 失败：文件不存在 {path}"
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"log_stats 失败：{exc}"

        lines = text.splitlines()
        level_counts: Counter[str] = Counter()
        error_fingerprints: Counter[str] = Counter()

        for line in lines:
            matched_level = ""
            for name, regex in LEVEL_PATTERNS:
                if regex.search(line):
                    matched_level = name
                    break
            if matched_level:
                level_counts[matched_level] += 1
            if matched_level in ERROR_LEVELS:
                error_fingerprints[self._fingerprint(line, fingerprint_chars)] += 1

        summary = [
            f"文件 {path}",
            f"总行数: {len(lines)}",
            f"非空行: {sum(1 for line in lines if line.strip())}",
        ]
        if level_counts:
            summary.append("级别分布:")
            for name, _ in LEVEL_PATTERNS:
                count = level_counts.get(name, 0)
                if count:
                    summary.append(f"  {name}: {count}")
        else:
            summary.append("级别分布: 未识别到常见级别标记")

        if error_fingerprints:
            summary.append(f"Top {top_errors} 错误指纹:")
            for fingerprint, count in error_fingerprints.most_common(top_errors):
                summary.append(f"  [{count}x] {fingerprint}")
        else:
            summary.append("未发现 FATAL/ERROR 级别日志。")
        return "\n".join(summary)

    @staticmethod
    def _fingerprint(line: str, width: int) -> str:
        text = re.sub(r"\b[0-9a-fA-F]{8,}\b", "<hex>", line)
        text = re.sub(r"\b\d+\b", "<n>", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:width]

    async def _arun(
        self,
        path: str,
        top_errors: int = 10,
        fingerprint_chars: int = 90,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, path, top_errors, fingerprint_chars, None)
