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


class GitDiffInput(BaseModel):
    staged: bool = Field(False, description="查看已暂存改动（等价于 git diff --cached）")
    ref: str | None = Field(
        None,
        description="对比的 git 引用，如 HEAD~1、main、某个 commit SHA；缺省对比当前工作区",
    )
    path: str | None = Field(None, description="限定相对路径，如 graph/agent.py")
    stat_only: bool = Field(False, description="只输出 --stat 变更统计")
    context_lines: int = Field(3, ge=0, le=50, description="diff 上下文行数")


class GitDiffTool(BaseTool):
    name: str = "git_diff"
    description: str = (
        "查看 git 差异：默认是工作区未暂存改动，staged=true 查看已暂存改动，"
        "ref 指定对比基线（如 HEAD~1）。可用 path 限定文件、stat_only 只看统计。只读操作。"
    )
    args_schema: Type[BaseModel] = GitDiffInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _run(
        self,
        staged: bool = False,
        ref: str | None = None,
        path: str | None = None,
        stat_only: bool = False,
        context_lines: int = 3,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        error = ensure_git_repo(self._root_dir)
        if error:
            return error
        if not is_safe_ref(ref):
            return "git diff 失败：ref 不能以 - 开头。"

        args = ["diff"]
        if staged:
            args.append("--cached")
        if stat_only:
            args.append("--stat")
        args.append(f"--unified={context_lines}")
        if ref:
            args.append(ref.strip())
        if path:
            args.extend(["--", path.replace("\\", "/").strip()])
        return run_git(self._root_dir, args)

    async def _arun(
        self,
        staged: bool = False,
        ref: str | None = None,
        path: str | None = None,
        stat_only: bool = False,
        context_lines: int = 3,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._run,
            staged,
            ref,
            path,
            stat_only,
            context_lines,
            None,
        )
