"""测量 Planner 两段调用（领域判定 + 任务拆解）的上下文缓存命中情况。

用途：改完提示词结构后跑一次，验证"可复用前缀"是否真的变长。
每次测量都使用唯一的请求文本，避免跨轮缓存污染，只看同一次任务内部的复用。

用法（在 backend 目录下）:
    .venv\\Scripts\\python -m scripts.measure_planner_cache 3
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.callbacks import BaseCallbackHandler

from agents import planner as planner_module
from graph.agent import agent_manager

BASE = Path(__file__).resolve().parent.parent

CODING_REQUEST = (
    "后端启动时 storage/logs/app.log 里有报错，帮我定位根因、修掉它，然后跑一遍测试验证。"
)

STAGE_NAMES = ("领域判定 + 任务拆解（单次调用）",)


class _UsageProbe(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def on_llm_end(self, response, **kwargs) -> None:
        for generation_list in getattr(response, "generations", []):
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                details = usage.get("input_token_details") or {}
                prompt_tokens = int(usage.get("input_tokens") or 0)
                hit = int(details.get("cache_read") or 0)
                self.calls.append(
                    {
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": int(usage.get("output_tokens") or 0),
                        "cache_hit_tokens": hit,
                        "cache_miss_tokens": prompt_tokens - hit,
                    }
                )


def _request_for_round(index: int) -> str:
    return f"{CODING_REQUEST}（独立请求编号 {index + 1}，请照常处理）"


async def _run_once(probe: _UsageProbe, message: str) -> dict:
    probe.calls.clear()
    planner = planner_module.Planner()
    planner.configure(BASE, lambda: agent_manager._build_chat_model(callbacks=[probe]))
    decision = await planner.decide(message, [])

    calls = list(probe.calls)
    prompt_tokens = sum(call["prompt_tokens"] for call in calls)
    hit = sum(call["cache_hit_tokens"] for call in calls)
    return {
        "domain": decision.domain,
        "task_count": len(decision.tasks),
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": prompt_tokens - hit,
        "cache_hit_rate": round(hit / prompt_tokens, 4) if prompt_tokens else 0.0,
    }


async def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    probe = _UsageProbe()
    results = []

    for index in range(rounds):
        result = await _run_once(probe, _request_for_round(index))
        result["round"] = index + 1
        results.append(result)
        print(
            f"round {index + 1}: domain={result['domain']} tasks={result['task_count']} "
            f"prompt={result['prompt_tokens']} hit={result['cache_hit_tokens']} "
            f"miss={result['cache_miss_tokens']} hit_rate={result['cache_hit_rate']:.1%}"
        )

    steady = results[-1]
    print("\n稳定态明细:")
    for index, call in enumerate(steady["calls"]):
        stage = STAGE_NAMES[index] if index < len(STAGE_NAMES) else f"第 {index + 1} 次调用"
        print(
            f"  {stage}: prompt={call['prompt_tokens']} "
            f"hit={call['cache_hit_tokens']} miss={call['cache_miss_tokens']}"
        )
    print(
        f"  合计: prompt={steady['prompt_tokens']} "
        f"hit={steady['cache_hit_tokens']} miss={steady['cache_miss_tokens']} "
        f"hit_rate={steady['cache_hit_rate']:.1%}"
    )
    print("\n原始数据:")
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
