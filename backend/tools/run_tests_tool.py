from __future__ import annotations

import asyncio
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Literal, Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from tools.path_guard import PROTECTED_DIR_NAMES


MAX_OUTPUT_CHARS = 8000
TAIL_CHARS = 4000


class RunTestsInput(BaseModel):
    framework: Literal[
        "auto", "pytest", "unittest", "npm", "vitest", "jest", "go", "cargo"
    ] = Field("auto", description="测试框架；auto 时按项目文件自动探测")
    path: str | None = Field(None, description="可选：指定测试文件或目录")
    command: str | None = Field(None, description="可选：完全自定义要执行的命令（覆盖 framework）")
    extra_args: str | None = Field(None, description="可选：追加到测试命令末尾的参数")
    timeout_seconds: int = Field(300, ge=10, le=1800, description="超时时间（秒）")


class RunTestsTool(BaseTool):
    name: str = "run_tests"
    description: str = (
        "运行项目测试并返回结构化结果：自动探测 pytest/unittest/npm/vitest/jest/go/cargo，"
        "输出退出码、通过/失败/错误/跳过统计和失败输出尾部。"
        "可用 path 指定测试范围、extra_args 追加参数、command 完全自定义命令。"
    )
    args_schema: Type[BaseModel] = RunTestsInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _detect_framework(self) -> str | None:
        root = self._root_dir
        for marker in ("pytest.ini", "tox.ini"):
            if (root / marker).exists():
                return "pytest"
        pyproject = root / "pyproject.toml"
        if pyproject.exists() and "pytest" in pyproject.read_text(encoding="utf-8", errors="replace"):
            return "pytest"
        setup_cfg = root / "setup.cfg"
        if setup_cfg.exists() and "pytest" in setup_cfg.read_text(encoding="utf-8", errors="replace"):
            return "pytest"
        if (root / "go.mod").exists():
            return "go"
        if (root / "Cargo.toml").exists():
            return "cargo"
        package_json = root / "package.json"
        if package_json.exists():
            content = package_json.read_text(encoding="utf-8", errors="replace")
            if "vitest" in content:
                return "vitest"
            if "jest" in content:
                return "jest"
            return "npm"
        if self._has_python_tests():
            return "pytest"
        return None

    def _has_python_tests(self) -> bool:
        for dirpath, dirnames, filenames in os.walk(self._root_dir):
            dirnames[:] = [name for name in dirnames if name.lower() not in PROTECTED_DIR_NAMES]
            for filename in filenames:
                if filename.startswith("test_") and filename.endswith(".py"):
                    return True
                if filename.endswith("_test.py"):
                    return True
        return False

    def _build_command(self, framework: str, path: str | None, extra_args: str | None) -> str:
        base = {
            "pytest": "python -m pytest",
            "unittest": "python -m unittest discover",
            "npm": "npm test --",
            "vitest": "npx vitest run",
            "jest": "npx jest",
            "go": "go test ./...",
            "cargo": "cargo test",
        }[framework]
        parts = [base]
        if path:
            parts.append(path.replace("\\", "/").strip())
        if extra_args:
            parts.append(extra_args.strip())
        return " ".join(part for part in parts if part)

    @staticmethod
    def _summarize(output: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        patterns = (
            ("passed", r"(\d+)\s+passed"),
            ("failed", r"(\d+)\s+failed"),
            ("errors", r"(\d+)\s+errors?"),
            ("skipped", r"(\d+)\s+skipped"),
        )
        for label, pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                counts[label] = int(match.group(1))
        return counts

    def _run(
        self,
        framework: str = "auto",
        path: str | None = None,
        command: str | None = None,
        extra_args: str | None = None,
        timeout_seconds: int = 300,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        if command and command.strip():
            resolved_framework = "custom"
            shell_command = command.strip()
        else:
            resolved_framework = framework
            if resolved_framework == "auto":
                detected = self._detect_framework()
                if not detected:
                    return (
                        "run_tests 失败：未能自动探测测试框架。"
                        "请用 framework 指定，或用 command 传完整命令。"
                    )
                resolved_framework = detected
            shell_command = self._build_command(resolved_framework, path, extra_args)

        if platform.system().lower().startswith("win"):
            argv = ["powershell", "-NoProfile", "-Command", shell_command]
        else:
            argv = ["bash", "-lc", shell_command]

        try:
            completed = subprocess.run(
                argv,
                cwd=self._root_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                # 测试框架多为 Python 系，统一按 UTF-8 输出，避免中文用例名乱码
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"run_tests 超时（>{timeout_seconds}s）：{shell_command}"

        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        counts = self._summarize(output)
        header = [
            f"命令: {shell_command}",
            f"框架: {resolved_framework}",
            f"退出码: {completed.returncode}",
        ]
        if counts:
            header.append(
                "统计: "
                + ", ".join(f"{key}={value}" for key, value in counts.items())
            )
        header.append("结果: " + ("通过" if completed.returncode == 0 else "失败"))

        tail = output[-TAIL_CHARS:] if len(output) > TAIL_CHARS else output
        if not tail:
            tail = "[no output]"
        body = "\n".join(header) + f"\n\n输出（尾部）:\n{tail}"
        if len(body) > MAX_OUTPUT_CHARS:
            body = body[:MAX_OUTPUT_CHARS] + "\n...[truncated]"
        return body

    async def _arun(
        self,
        framework: str = "auto",
        path: str | None = None,
        command: str | None = None,
        extra_args: str | None = None,
        timeout_seconds: int = 300,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._run,
            framework,
            path,
            command,
            extra_args,
            timeout_seconds,
            None,
        )
