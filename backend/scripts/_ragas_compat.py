# -*- coding: utf-8 -*-
"""Ragas 导入兼容层。

背景
----
ragas（0.3.x / 0.4.x 均如此）在 ``ragas/llms/base.py`` 顶部硬导入::

    from langchain_community.chat_models.vertexai import ChatVertexAI

而 ``langchain-community`` 0.4.x 已移除该模块（VertexAI 迁到独立的
``langchain-google-vertexai``）。因此本项目环境里 ``import ragas`` 会抛
``ModuleNotFoundError``，又被各评测脚本的 ``except ImportError`` 静默吞掉，
结果是 Ragas 指标"看起来跑了"，实际全部被跳过。

处理方式
--------
在导入 ragas 之前注册一个占位模块。``ChatVertexAI`` 在 ragas 中只出现在
``MULTIPLE_COMPLETION_SUPPORTED`` 的 ``isinstance`` 判断列表里，占位类不参与
任何实际调用，因此不影响指标结果。

用法
----
在任何 ``import ragas`` / ``from ragas... import ...`` 之前::

    import _ragas_compat  # noqa: F401  （导入即生效）

该模块可重复导入，已存在真实实现时不做任何替换。
"""
from __future__ import annotations

import sys
import types

MODULE_NAME = "langchain_community.chat_models.vertexai"


class _ChatVertexAIPlaceholder:
    """占位类：仅用于满足 ragas 的 isinstance 判断。"""


def install() -> bool:
    """注册占位模块；返回 True 表示本次真正打桩，False 表示原本就可用。"""

    if MODULE_NAME in sys.modules:
        return False
    try:  # 真实实现存在时（langchain-community < 0.4）直接使用
        __import__(MODULE_NAME)
        return False
    except Exception:
        pass

    module = types.ModuleType(MODULE_NAME)
    module.ChatVertexAI = _ChatVertexAIPlaceholder  # type: ignore[attr-defined]
    module.__doc__ = "Shim module registered by _ragas_compat for ragas compatibility."
    sys.modules[MODULE_NAME] = module

    try:  # 让 `from langchain_community.chat_models import vertexai` 也能工作
        import langchain_community.chat_models as _chat_models

        setattr(_chat_models, "vertexai", module)
    except Exception:
        pass
    return True


PATCHED = install()
