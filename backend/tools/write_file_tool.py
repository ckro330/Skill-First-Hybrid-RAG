from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Type

from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from tools.path_guard import MAX_CONTENT_CHARS, resolve_file_path


class WriteFileInput(BaseModel):
    path: str = Field(..., description="Relative path inside the project root, for example skills/foo/SKILL.md")
    content: str = Field(..., description="Full text content to write, UTF-8 encoded")
    overwrite: bool = Field(
        False,
        description="Replace the file when it already exists; default false refuses to clobber it",
    )


class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = (
        "Create or overwrite a UTF-8 text file under the project root, creating missing parent "
        "directories. Refuses to replace an existing file unless overwrite=true. Prefer edit_file "
        "for small changes to existing files."
    )
    args_schema: Type[BaseModel] = WriteFileInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _run(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            file_path = resolve_file_path(self._root_dir, path)
        except ValueError as exc:
            return f"Write failed: {exc}"

        if len(content) > MAX_CONTENT_CHARS:
            return f"Write failed: content exceeds {MAX_CONTENT_CHARS} characters."

        existed = file_path.exists()
        if existed and not overwrite:
            return "Write failed: file already exists. Pass overwrite=true or use edit_file."

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8", newline="")
        except OSError as exc:
            return f"Write failed: {exc}"

        normalized = path.replace("\\", "/").strip("/")
        action = "Overwrote" if existed else "Created"
        return f"{action} {normalized} ({len(content)} chars)."

    async def _arun(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, path, content, overwrite, None)
