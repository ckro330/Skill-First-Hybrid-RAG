from __future__ import annotations

import asyncio
import os
import platform
import subprocess
from pathlib import Path
from typing import Type

from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from config import get_settings


BLOCKED_PATTERNS = (
    "rm -rf /",
    "shutdown",
    "reboot",
    "mkfs",
    "format ",
    ":(){:|:&};:",
)


class TerminalToolInput(BaseModel):
    command: str = Field(..., description="Shell command to execute inside the project root")


class TerminalTool(BaseTool):
    name: str = "terminal"
    description: str = (
        "Execute shell commands inside the project root. Use this for quick inspection, "
        "building, or local commands. Dangerous system-destructive commands are blocked."
    )
    args_schema: Type[BaseModel] = TerminalToolInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir

    def _run(
        self,
        command: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        lowered = command.lower()
        if any(pattern in lowered for pattern in BLOCKED_PATTERNS):
            return "Blocked: command matches the terminal blacklist."

        settings = get_settings()
        # 子进程按 UTF-8 输出，避免 Windows 默认 GBK 造成的乱码/解码失败
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        if platform.system().lower().startswith("win"):
            # PowerShell 会把子进程输出按控制台编码重新编码，这里显式切成 UTF-8
            wrapped = (
                "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
                "$OutputEncoding=[System.Text.Encoding]::UTF8;"
                f"{command}"
            )
            shell_command = ["powershell", "-NoProfile", "-Command", wrapped]
        else:
            shell_command = ["bash", "-lc", command]
        try:
            completed = subprocess.run(
                shell_command,
                cwd=self._root_dir,
                capture_output=True,
                text=True,
                # Windows 默认用系统本地编码（GBK）解码子进程输出，
                # 一旦命令输出 UTF-8 中文就会抛 UnicodeDecodeError 并丢失输出
                encoding="utf-8",
                errors="replace",
                env=environment,
                timeout=settings.terminal_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"Timed out after {settings.terminal_timeout_seconds} seconds."

        combined = (completed.stdout or "") + (completed.stderr or "")
        combined = combined.strip() or "[no output]"
        return combined[:5000]

    async def _arun(
        self,
        command: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, command, None)
