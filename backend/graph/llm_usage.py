"""LLM 调用的 token / 上下文缓存统计。

DeepSeek 与 OpenAI 都会在 usage 里返回缓存命中信息，LangChain 统一映射到
`usage_metadata.input_token_details.cache_read`，所以这里不需要区分厂商。

采集方式是给模型挂一个 callback（见 AgentManager._build_chat_model），
每次 LLM 调用结束后把数据记进"当前轮次"的上下文变量里。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


@dataclass(frozen=True)
class LlmCallUsage:
    prompt_tokens: int
    output_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int


_turn_bucket: ContextVar[list[LlmCallUsage] | None] = ContextVar(
    "llm_usage_turn", default=None
)


def extract_usage(message: Any) -> LlmCallUsage | None:
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict) or not usage:
        return None
    details = usage.get("input_token_details") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    hit = int(details.get("cache_read") or 0)
    return LlmCallUsage(
        prompt_tokens=prompt_tokens,
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_hit_tokens=hit,
        cache_miss_tokens=max(0, prompt_tokens - hit),
    )


class LlmUsageCollector(BaseCallbackHandler):
    """记录每次 LLM 调用的用量；没有开启本轮采集时静默跳过。"""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        bucket = _turn_bucket.get()
        if bucket is None:
            return
        for generation_list in getattr(response, "generations", []):
            for generation in generation_list:
                record = extract_usage(getattr(generation, "message", None))
                if record is not None:
                    bucket.append(record)


usage_collector = LlmUsageCollector()


def begin_turn() -> None:
    """开始采集一轮对话（在 SSE 生成器内部调用）。"""
    _turn_bucket.set([])


def end_turn() -> dict[str, Any]:
    """结束采集并汇总本轮所有 LLM 调用。"""
    bucket = _turn_bucket.get() or []
    _turn_bucket.set(None)
    prompt_tokens = sum(record.prompt_tokens for record in bucket)
    hit = sum(record.cache_hit_tokens for record in bucket)
    return {
        "llm_calls": len(bucket),
        "prompt_tokens": prompt_tokens,
        "output_tokens": sum(record.output_tokens for record in bucket),
        "cache_hit_tokens": hit,
        "cache_miss_tokens": prompt_tokens - hit,
        "cache_hit_rate": round(hit / prompt_tokens, 4) if prompt_tokens else 0.0,
    }
