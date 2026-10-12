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


class GitAddInput(BaseModel):
    paths: list[str] = Field(
        ...,
        description='要暂存的相对路径列表；暂存全部改动时传 ["."]',
    )


class GitAddTool(BaseTool):
    name: str = "git_add"
    description: str = (
        "把指定路径的改动加入暂存区（git add）。paths 为相对路径列表，"
        '可用 ["."] 暂存全部。执行后会返回最新状态。只暂存，不提交。'
    )
    args_schema: Type[BaseModel] = GitAddInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _run(
        self,
        paths: list[str],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        error = ensure_git_repo(self._root_dir)
        if error:
            return error
        cleaned = [str(item).replace("\\", "/").strip() for item in (paths or [])]
        cleaned = [item for item in cleaned if item]
        if not cleaned:
            return "git add 失败：paths 不能为空。"
        if any(item.startswith("-") for item in cleaned):
            return "git add 失败：路径不能以 - 开头。"

        added = run_git(self._root_dir, ["add", "--", *cleaned])
        status = run_git(self._root_dir, ["status", "--short", "--branch"])
        return f"{added}\n\n{status}"

    async def _arun(
        self,
        paths: list[str],
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, paths, None)
