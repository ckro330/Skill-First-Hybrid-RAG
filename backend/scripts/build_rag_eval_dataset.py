# -*- coding: utf-8 -*-
"""从知识库切片生成 Ragas 评测数据集（question + reference + reference_contexts）。

标注由 LLM 基于「原文片段」生成，因此 reference 答案严格来自该片段，
reference_context_ids 使用与检索一致的 `source_path::locator`，可直接用于
id_based_context_precision / id_based_context_recall 等带参考的指标。

用法（在 backend/ 目录下）
--------------------------
  python scripts/build_rag_eval_dataset.py --count 12
  python scripts/build_rag_eval_dataset.py --count 20 --min-chars 400 --seed 7 --output scripts/datasets/rag_eval_v2.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DEFAULT_OUTPUT = Path("scripts") / "datasets" / "rag_eval_sample.jsonl"
CONTEXT_ID_SEP = "::"
JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

PROMPT_TEMPLATE = (
    "你是评测数据集标注员。下面是从知识库中截取的一段原文。\n\n"
    "要求：\n"
    "1. 生成一个「仅凭这段原文就能完整回答」的问题，问题要具体、贴近真实用户提问，"
    "不要出现「这段文字」「上文」之类的指代。\n"
    "2. 给出基于原文的标准答案，2-4 句，不得引入原文之外的信息。\n"
    "3. 只输出严格 JSON：{\"question\": \"...\", \"reference\": \"...\"}，不要输出任何其他内容。\n\n"
    "【原文】\n{text}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从知识库切片生成 Ragas 评测数据集")
    parser.add_argument("--count", type=int, default=12, help="生成多少条")
    parser.add_argument("--seed", type=int, default=20261002, help="采样随机种子")
    parser.add_argument("--min-chars", type=int, default=300, help="片段最小字符数")
    parser.add_argument("--max-chars", type=int, default=1200, help="片段最大字符数")
    parser.add_argument("--path-filter", action="append", default=[], help="限定知识库路径前缀，可重复")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSONL 路径")
    parser.add_argument("--append", action="store_true", help="追加到已有文件而不是覆盖")
    return parser.parse_args()


def extract_json(text: str) -> dict[str, Any] | None:
    match = JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def candidate_chunks(min_chars: int, max_chars: int, path_filters: list[str]) -> list[dict[str, Any]]:
    from knowledge_retrieval import knowledge_indexer

    knowledge_indexer.configure(BACKEND_DIR)
    knowledge_indexer._ensure_loaded()

    candidates: list[dict[str, Any]] = []
    for item in knowledge_indexer._documents:
        text = str(item.get("text") or "").strip()
        if not (min_chars <= len(text) <= max_chars):
            continue
        source_path = str(item.get("source_path") or "")
        if path_filters and not any(source_path.startswith(p) for p in path_filters):
            continue
        if not text or text.startswith("#"):
            continue
        candidates.append(item)
    return candidates


async def build(args: argparse.Namespace) -> int:
    from graph.agent import agent_manager

    agent_manager.initialize(BACKEND_DIR)
    model = agent_manager._build_chat_model()

    candidates = candidate_chunks(args.min_chars, args.max_chars, list(args.path_filter))
    if not candidates:
        raise SystemExit("没有满足条件的知识库片段，请放宽 --min-chars / --max-chars")
    print(f"候选片段：{len(candidates)}")

    rng = random.Random(args.seed)
    picked = rng.sample(candidates, min(args.count, len(candidates)))

    output_path = args.output if args.output.is_absolute() else (BACKEND_DIR / args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    written = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for index, item in enumerate(picked, start=1):
            text = str(item["text"]).strip()
            source_path = str(item["source_path"])
            locator = str(item["locator"])
            print(f"[{index}/{len(picked)}] {source_path} / {locator}")
            # 模板里含 JSON 花括号，不能用 str.format，改用占位替换
            prompt = PROMPT_TEMPLATE.replace("{text}", text[:4000])
            response = await model.ainvoke(prompt)
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = "".join(
                    str(block.get("text", "")) for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            payload = extract_json(str(content))
            if not payload or not payload.get("question") or not payload.get("reference"):
                print("    ! 标注解析失败，跳过")
                continue
            record = {
                "id": f"{output_path.stem}-{written + 1:03d}",
                "question": str(payload["question"]).strip(),
                "reference": str(payload["reference"]).strip(),
                "reference_contexts": [text],
                "reference_context_ids": [f"{source_path}{CONTEXT_ID_SEP}{locator}"],
                "source": {"source_path": source_path, "locator": locator, "source_type": item.get("source_type")},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n已写入 {written} 条 -> {output_path}")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(build(args))


if __name__ == "__main__":
    raise SystemExit(main())
