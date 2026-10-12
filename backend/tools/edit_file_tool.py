from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Type

from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from tools.path_guard import MAX_CONTENT_CHARS, resolve_file_path


class EditFileInput(BaseModel):
    path: str = Field(..., description="Relative path inside the project root, for example skills/foo/SKILL.md")
    old_text: str = Field(..., description="Exact existing text to replace; include surrounding context so it stays unique")
    new_text: str = Field("", description="Replacement text; pass an empty string to delete old_text")
    replace_all: bool = Field(False, description="Replace every occurrence instead of requiring a unique match")


class EditFileTool(BaseTool):
    name: str = "edit_file"
    description: str = (
        "Edit an existing UTF-8 text file under the project root by replacing an exact old_text block "
        "with new_text. Fails when old_text is missing or ambiguous, so copy it verbatim from read_file "
        "and keep enough surrounding context. Set replace_all=true to replace every occurrence."
    )
    args_schema: Type[BaseModel] = EditFileInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()

    def _run(
        self,
        path: str,
        old_text: str,
        new_text: str = "",
        replace_all: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            file_path = resolve_file_path(self._root_dir, path)
        except ValueError as exc:
            return f"Edit failed: {exc}"

        if not old_text:
            return "Edit failed: old_text must not be empty."
        if not file_path.exists():
            return "Edit failed: file does not exist. Use write_file to create it."

        try:
            with file_path.open("r", encoding="utf-8", newline="") as handle:
                original = handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            return f"Edit failed: {exc}"

        count = original.count(old_text)
        if count == 0:
            return "Edit failed: old_text was not found. Read the file first and copy the exact block."
        if count > 1 and not replace_all:
            return f"Edit failed: old_text matches {count} places. Add surrounding context or set replace_all=true."

        updated = original.replace(old_text, new_text) if replace_all else original.replace(old_text, new_text, 1)
        if len(updated) > MAX_CONTENT_CHARS:
            return f"Edit failed: result exceeds {MAX_CONTENT_CHARS} characters."

        try:
            file_path.write_text(updated, encoding="utf-8", newline="")
        except OSError as exc:
            return f"Edit failed: {exc}"

        normalized = path.replace("\\", "/").strip("/")
        replaced = count if replace_all else 1
        delta = len(updated) - len(original)
        return f"Edited {normalized} ({replaced} replacement(s), {delta:+d} chars)."

    async def _arun(
        self,
        path: str,
        old_text: str,
        new_text: str = "",
        replace_all: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, path, old_text, new_text, replace_all, None)
