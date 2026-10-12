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

from tools.git_base import ensure_git_repo, run_git


class GitStatusInput(BaseModel):
    short: bool = Field(True, description="使用 git status --short --branch 的紧凑输出")


class GitStatusTool(BaseTool):
    name: str = "git_status"
    description: str = (
        "查看当前 git 仓库的工作区状态：当前分支、已暂存/未暂存改动、未跟踪文件。"
        "只读操作，不修改仓库。开始任何代码修改前应先看状态。"
    )
    args_schema: Type[BaseModel] = GitStatusInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _run(
        self,
        short: bool = True,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        error = ensure_git_repo(self._root_dir)
        if error:
            return error
        args = ["status", "--short", "--branch"] if short else ["status"]
        return run_git(self._root_dir, args)

    async def _arun(
        self,
        short: bool = True,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, short, None)
