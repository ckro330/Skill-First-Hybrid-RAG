from __future__ import annotations

from pathlib import Path


MAX_CONTENT_CHARS = 200_000
PROTECTED_DIR_NAMES = {".git", ".venv", "venv", "node_modules", "__pycache__", ".next"}
PROTECTED_FILE_NAMES = {".env"}


def resolve_file_path(root_dir: Path, path: str) -> Path:
    """把相对路径解析为项目根目录内的文件路径，越界或命中受保护路径时报错。"""
    normalized = (path or "").replace("\\", "/").strip()
    if not normalized:
        raise ValueError("Path is empty.")

    candidate = (root_dir / normalized).resolve()
    if root_dir not in candidate.parents and candidate != root_dir:
        raise ValueError("Path traversal detected.")
    if candidate == root_dir or candidate.is_dir():
        raise ValueError("Path must point to a file.")

    relative = candidate.relative_to(root_dir)
    if any(part.lower() in PROTECTED_DIR_NAMES for part in relative.parts):
        raise ValueError("Path is inside a protected directory.")
    if relative.name.lower() in PROTECTED_FILE_NAMES:
        raise ValueError("Path is a protected file.")
    return candidate
