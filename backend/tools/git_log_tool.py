from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from tools.git_base import ensure_git_repo, is_safe_ref, run_git


class GitLogInput(BaseModel):
    max_count: int = Field(20, ge=1, le=200, description="返回的提交数量")
    oneline: bool = Field(True, description="使用 --oneline 紧凑输出")
    path: str | None = Field(None, description="只查看影响该路径的提交")
    ref: str | None = Field(None, description="从哪个引用开始，如 HEAD、main、某个 SHA")


class GitLogTool(BaseTool):
    name: str = "git_log"
    description: str = (
        "查看 git 提交历史：数量、是否 oneline、按路径或引用过滤。"
        "用于理解改动脉络、定位某次改动来源。只读操作。"
    )
    args_schema: Type[BaseModel] = GitLogInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _run(
        self,
        max_count: int = 20,
        oneline: bool = True,
        path: str | None = None,
        ref: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        error = ensure_git_repo(self._root_dir)
        if error:
            return error
        if not is_safe_ref(ref):
            return "git log 失败：ref 不能以 - 开头。"

        args = ["log", f"-n{max_count}"]
        args.extend(["--oneline", "--decorate"] if oneline else ["--pretty=medium"])
        if ref:
            args.append(ref.strip())
        if path:
            args.extend(["--", path.replace("\\", "/").strip()])
        return run_git(self._root_dir, args)

    async def _arun(
        self,
        max_count: int = 20,
        oneline: bool = True,
        path: str | None = None,
        ref: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, max_count, oneline, path, ref, None)
