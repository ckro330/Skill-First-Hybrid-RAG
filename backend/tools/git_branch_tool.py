from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Literal, Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from tools.git_base import ensure_git_repo, is_safe_ref, run_git


BRANCH_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


class GitBranchInput(BaseModel):
    action: Literal["list", "create", "switch", "create_and_switch", "delete"] = Field(
        ..., description="要执行的分支操作"
    )
    name: str | None = Field(None, description="分支名；create/switch/create_and_switch/delete 必填")
    start_point: str | None = Field(None, description="可选：创建分支的起点，如 main、某个 SHA")


class GitBranchTool(BaseTool):
    name: str = "git_branch"
    description: str = (
        "分支管理：list 列出本地/远程分支，create 新建分支，switch 切换分支，"
        "create_and_switch 新建并切换，delete 删除已合并分支（-d，安全删除）。"
        "删除属于破坏性操作，必须先确认用户意图。不执行 push。"
    )
    args_schema: Type[BaseModel] = GitBranchInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _run(
        self,
        action: str,
        name: str | None = None,
        start_point: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        error = ensure_git_repo(self._root_dir)
        if error:
            return error
        if not is_safe_ref(start_point):
            return "git branch 失败：start_point 不能以 - 开头。"

        if action == "list":
            current = run_git(self._root_dir, ["branch", "--show-current"])
            branches = run_git(self._root_dir, ["branch", "-a", "-vv"])
            return f"当前分支: {current}\n\n{branches}"

        cleaned_name = (name or "").strip()
        if not cleaned_name:
            return f"git branch 失败：action={action} 需要提供 name。"
        if not BRANCH_NAME_PATTERN.match(cleaned_name) or ".." in cleaned_name:
            return "git branch 失败：分支名不合法。"

        if action == "create":
            args = ["branch", cleaned_name]
            if start_point:
                args.append(start_point.strip())
            return run_git(self._root_dir, args)
        if action == "switch":
            return run_git(self._root_dir, ["switch", cleaned_name])
        if action == "create_and_switch":
            args = ["switch", "-c", cleaned_name]
            if start_point:
                args.append(start_point.strip())
            return run_git(self._root_dir, args)
        if action == "delete":
            return run_git(self._root_dir, ["branch", "-d", cleaned_name])
        return f"git branch 失败：未知 action={action}。"

    async def _arun(
        self,
        action: str,
        name: str | None = None,
        start_point: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, action, name, start_point, None)
