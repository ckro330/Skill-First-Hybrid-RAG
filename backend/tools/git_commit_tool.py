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


class GitCommitInput(BaseModel):
    message: str = Field(..., min_length=2, description="提交信息（提交标题，简洁说明改动）")
    paths: list[str] | None = Field(
        None,
        description="可选：提交前先暂存这些相对路径；缺省直接提交当前已暂存内容",
    )
    allow_empty: bool = Field(False, description="允许空提交（无改动时也创建 commit）")


class GitCommitTool(BaseTool):
    name: str = "git_commit"
    description: str = (
        "创建一次 git 提交。可选先 git add 给定 paths，再提交已暂存内容。"
        "只提交，绝不 push；如需推送请提示用户手动执行。"
        "提交前应先用 git_status / git_diff 确认改动范围。"
    )
    args_schema: Type[BaseModel] = GitCommitInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _run(
        self,
        message: str,
        paths: list[str] | None = None,
        allow_empty: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        error = ensure_git_repo(self._root_dir)
        if error:
            return error
        message = (message or "").strip()
        if not message:
            return "git commit 失败：提交信息不能为空。"

        preamble = ""
        if paths:
            cleaned = [str(item).replace("\\", "/").strip() for item in paths]
            cleaned = [item for item in cleaned if item]
            if not cleaned:
                return "git commit 失败：paths 为空。"
            if any(item.startswith("-") for item in cleaned):
                return "git commit 失败：路径不能以 - 开头。"
            preamble = run_git(self._root_dir, ["add", "--", *cleaned]) + "\n"

        args = ["commit", "-m", message]
        if allow_empty:
            args.append("--allow-empty")
        committed = run_git(self._root_dir, args)
        head = run_git(self._root_dir, ["log", "-1", "--oneline"])
        return f"{preamble}{committed}\n\nHEAD: {head}"

    async def _arun(
        self,
        message: str,
        paths: list[str] | None = None,
        allow_empty: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, message, paths, allow_empty, None)
