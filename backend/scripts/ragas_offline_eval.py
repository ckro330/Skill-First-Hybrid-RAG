# -*- coding: utf-8 -*-
"""基于 Ragas 的「检索 + 回答」离线评测，产出可复现报告。

链路
----
数据集(JSONL) -> 检索(hybrid / vector / bm25 / orchestrator) -> LLM 作答 -> Ragas 指标 -> JSON + Markdown 报告

覆盖指标
--------
- 上下文精确率：id_based_context_precision / non_llm_context_precision_with_reference /
  llm_context_precision_without_reference
- 上下文召回率：id_based_context_recall / non_llm_context_recall / context_recall
- 忠实度：faithfulness
- 答案相关性：answer_relevancy

可复现性
--------
报告固定记录：数据集 sha256、检索参数、索引状态与 manifest 哈希、模型与温度、
ragas/langchain 等依赖版本、git commit 与工作区是否脏、随机种子、复现命令。

用法（在 backend/ 目录下）
--------------------------
  python scripts/ragas_offline_eval.py --dataset scripts/datasets/rag_eval_sample.jsonl
  python scripts/ragas_offline_eval.py --metrics off            # 只跑检索，快速迭代检索策略
  python scripts/ragas_offline_eval.py --retriever vector --top-k 8 --run-name vector_top8
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (str(BACKEND_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import _ragas_compat  # noqa: F401  —— 必须在任何 ragas 导入之前

DEFAULT_DATASET = Path("scripts") / "datasets" / "rag_eval_sample.jsonl"
DEFAULT_OUTPUT_DIR = Path("storage") / "eval_outputs"
CONTEXT_ID_SEP = "::"
MAX_CONTEXT_CHARS = 6000

ANSWER_SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。必须遵守：\n"
    "1. 只依据【检索证据】作答，不使用证据之外的知识，不编造。\n"
    "2. 若证据不足以回答，直接说明证据不足并指出还缺什么，不要强行作答。\n"
    "3. 使用与提问相同的语言，先给结论再给依据。\n"
    "4. 引用证据时标注编号，例如 [1]。\n"
)

FAMILY_PRIORITY: dict[str, list[str]] = {
    "context_precision": [
        "id_based_context_precision",
        "non_llm_context_precision_with_reference",
        "llm_context_precision_without_reference",
    ],
    "context_recall": [
        "id_based_context_recall",
        "non_llm_context_recall",
        "context_recall",
    ],
    "faithfulness": ["faithfulness"],
    "answer_relevancy": ["answer_relevancy"],
}

FAMILY_LABEL = {
    "context_precision": "上下文精确率",
    "context_recall": "上下文召回率",
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
}


# --------------------------------------------------------------------------- #
# 数据结构与基础工具
# --------------------------------------------------------------------------- #
@dataclass
class EvalSample:
    sample_id: str
    question: str
    reference: str = ""
    reference_contexts: list[str] = field(default_factory=list)
    reference_context_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    context_id: str
    text: str
    source_path: str
    locator: str
    channel: str
    score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "source_path": self.source_path,
            "locator": self.locator,
            "channel": self.channel,
            "score": self.score,
            "text": self.text,
        }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, 6)


def stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def context_id_of(source_path: str, locator: str) -> str:
    return f"{source_path}{CONTEXT_ID_SEP}{locator}"


def package_versions() -> dict[str, str]:
    from importlib import metadata

    versions: dict[str, str] = {}
    for name in (
        "ragas", "langchain", "langchain-core", "langchain-community",
        "langchain-openai", "langchain-deepseek", "sentence-transformers",
        "llama-index-core", "datasets", "pandas", "numpy",
    ):
        try:
            versions[name] = metadata.version(name)
        except Exception:
            continue
    return versions


def git_info(base_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": "", "dirty": None, "branch": ""}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=base_dir,
                                capture_output=True, text=True, timeout=10)
        if commit.returncode == 0:
            info["commit"] = commit.stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=base_dir,
                                capture_output=True, text=True, timeout=10)
        if branch.returncode == 0:
            info["branch"] = branch.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=base_dir,
                                capture_output=True, text=True, timeout=10)
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except Exception:
        pass
    return info


def load_dataset(path: Path, limit: int = 0) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} 应为 JSON 对象")
            question = str(item.get("question") or item.get("user_input") or "").strip()
            if not question:
                raise ValueError(f"{path}:{line_number} 缺少 question 字段")
            samples.append(
                EvalSample(
                    sample_id=str(item.get("id") or f"q{line_number}"),
                    question=question,
                    reference=str(item.get("reference") or item.get("ground_truth") or "").strip(),
                    reference_contexts=[str(x) for x in (item.get("reference_contexts") or [])],
                    reference_context_ids=[str(x) for x in (item.get("reference_context_ids") or [])],
                    metadata={
                        k: v for k, v in item.items()
                        if k not in {
                            "id", "question", "user_input", "reference",
                            "ground_truth", "reference_contexts", "reference_context_ids",
                        }
                    },
                )
            )
    if limit and limit > 0:
        samples = samples[:limit]
    return samples


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
def _to_chunk(evidence: Any) -> RetrievedChunk:
    source_path = str(getattr(evidence, "source_path", "") or "")
    locator = str(getattr(evidence, "locator", "") or "")
    return RetrievedChunk(
        context_id=context_id_of(source_path, locator),
        text=str(getattr(evidence, "snippet", "") or "").strip(),
        source_path=source_path,
        locator=locator,
        channel=str(getattr(evidence, "channel", "") or ""),
        score=sanitize_float(getattr(evidence, "score", None)),
    )


async def retrieve(question: str, config: dict[str, Any], model_builder: Any) -> list[RetrievedChunk]:
    from knowledge_retrieval import knowledge_indexer, knowledge_orchestrator

    mode = config["mode"]
    path_filters = config["path_filters"] or None
    query_hints = config["query_hints"] or None

    if mode == "orchestrator":
        knowledge_orchestrator.configure(BACKEND_DIR, model_builder)
        result = None
        async for event in knowledge_orchestrator.astream(question):
            if event.get("type") == "orchestrated_result":
                result = event["result"]
        evidences = list(result.evidences) if result is not None else []
    elif mode == "vector":
        evidences = knowledge_indexer.retrieve_vector(
            question, top_k=config["top_k"], path_filters=path_filters
        )
    elif mode == "bm25":
        evidences = knowledge_indexer.retrieve_bm25(
            question, top_k=config["top_k"], path_filters=path_filters, query_hints=query_hints
        )
    else:  # hybrid：向量 + BM25 双路召回，再 RRF 融合
        from knowledge_retrieval.fusion import reciprocal_rank_fusion

        vector_hits = knowledge_indexer.retrieve_vector(
            question, top_k=config["top_k"], path_filters=path_filters
        )
        bm25_hits = knowledge_indexer.retrieve_bm25(
            question, top_k=config["top_k"], path_filters=path_filters, query_hints=query_hints
        )
        evidences = reciprocal_rank_fusion(
            [vector_hits, bm25_hits],
            top_k=config["fusion_top_k"],
            rank_constant=config["rank_constant"],
        )
    return [_to_chunk(item) for item in evidences]


async def generate_answer(question: str, chunks: list[RetrievedChunk], model: Any) -> str:
    if chunks:
        blocks = [
            f"[{i}] 来源: {c.source_path} ({c.locator})\n{c.text}"
            for i, c in enumerate(chunks, start=1)
        ]
        context = "\n\n".join(blocks)[:MAX_CONTEXT_CHARS]
    else:
        context = "（未检索到任何证据）"
    response = await model.ainvoke(
        [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"【检索证据】\n{context}\n\n【问题】\n{question}"},
        ]
    )
    return stringify_content(getattr(response, "content", response)).strip()


# --------------------------------------------------------------------------- #
# Ragas 指标
# --------------------------------------------------------------------------- #
def build_metrics(
    tier: str,
    *,
    has_reference_context_ids: bool,
    has_reference_contexts: bool,
    has_reference: bool,
    llm: Any,
    embeddings: Any,
) -> tuple[list[tuple[str, Any]], list[dict[str, str]]]:
    from ragas.metrics import (
        Faithfulness,
        IDBasedContextPrecision,
        IDBasedContextRecall,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        NonLLMContextPrecisionWithReference,
        NonLLMContextRecall,
        ResponseRelevancy,
    )

    specs: list[tuple[str, Any]] = []
    skipped: list[dict[str, str]] = []

    if tier in {"deterministic", "all"}:
        if has_reference_context_ids:
            specs.append(("context_precision", IDBasedContextPrecision()))
            specs.append(("context_recall", IDBasedContextRecall()))
        else:
            for name in ("id_based_context_precision", "id_based_context_recall"):
                skipped.append({"metric": name, "reason": "数据集缺少 reference_context_ids"})
        if has_reference_contexts:
            specs.append(("context_precision", NonLLMContextPrecisionWithReference()))
            specs.append(("context_recall", NonLLMContextRecall()))
        else:
            for name in ("non_llm_context_precision_with_reference", "non_llm_context_recall"):
                skipped.append({"metric": name, "reason": "数据集缺少 reference_contexts"})

    if tier in {"llm", "all"}:
        if llm is None:
            for name in (
                "faithfulness", "llm_context_precision_without_reference",
                "context_recall", "answer_relevancy",
            ):
                skipped.append({"metric": name, "reason": "缺少 evaluator LLM"})
        else:
            specs.append(("faithfulness", Faithfulness()))
            specs.append(("context_precision", LLMContextPrecisionWithoutReference()))
            if has_reference:
                specs.append(("context_recall", LLMContextRecall()))
            else:
                skipped.append({"metric": "context_recall", "reason": "数据集缺少 reference 标准答案"})
            if embeddings is not None:
                # strictness 默认 3，会向 LLM 请求 n=3；DeepSeek 等兼容端点只支持 n=1，
                # 不显式降为 1 会让 answer_relevancy 全样本报 400 而拿到 None。
                specs.append(("answer_relevancy", ResponseRelevancy(strictness=1)))
            else:
                skipped.append({"metric": "answer_relevancy", "reason": "缺少 evaluator embedding"})

    return specs, skipped


def _extract_per_sample(evaluation: Any, expected: int) -> list[dict[str, Any]]:
    """从 Ragas EvaluationResult 里取出逐样本分数（兼容不同版本的容器）。"""

    try:
        records = evaluation.to_pandas().to_dict(orient="records")
        if records:
            return [{str(k): v for k, v in row.items()} for row in records]
    except Exception:
        pass
    raw_scores = getattr(evaluation, "scores", None)
    if isinstance(raw_scores, list) and raw_scores:
        return [{str(k): v for k, v in row.items()} for row in raw_scores]
    return [{} for _ in range(expected)]


def _build_evaluator() -> tuple[Any, Any]:
    """Ragas 的裁判 LLM 与 embedding，均复用项目自身配置。"""

    from graph.agent import agent_manager
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(agent_manager._build_chat_model())
    embeddings = None
    try:
        from config import get_settings
        from knowledge_retrieval.embeddings import LocalLangchainEmbeddings

        settings = get_settings()
        if settings.embedding_provider == "local":
            embeddings = LangchainEmbeddingsWrapper(LocalLangchainEmbeddings())
        elif settings.embedding_api_key:
            from langchain_openai import OpenAIEmbeddings

            embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    model=settings.embedding_model,
                    api_key=settings.embedding_api_key,
                    base_url=settings.embedding_base_url,
                )
            )
    except Exception:
        embeddings = None
    return llm, embeddings


def run_ragas(
    tier: str,
    rows: list[dict[str, Any]],
    *,
    seed: int,
    max_workers: int,
    timeout: int,
    build_evaluator: bool,
) -> dict[str, Any]:
    from ragas import EvaluationDataset, evaluate as ragas_evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.run_config import RunConfig

    has_reference_context_ids = any(row["reference_context_ids"] for row in rows)
    has_reference_contexts = any(row["reference_contexts"] for row in rows)
    has_reference = any(row["reference"] for row in rows)

    llm = None
    embeddings = None
    if tier in {"llm", "all"} and build_evaluator:
        llm, embeddings = _build_evaluator()

    specs, skipped = build_metrics(
        tier,
        has_reference_context_ids=has_reference_context_ids,
        has_reference_contexts=has_reference_contexts,
        has_reference=has_reference,
        llm=llm,
        embeddings=embeddings,
    )
    if not specs:
        return {"executed": [], "skipped": skipped, "summary": {}, "headline": {}, "per_sample": []}

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["response"],
            reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
            reference_contexts=row["reference_contexts"],
            retrieved_context_ids=row["retrieved_context_ids"],
            reference_context_ids=row["reference_context_ids"],
        )
        for row in rows
    ]

    evaluation = ragas_evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[metric for _, metric in specs],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(seed=seed, max_workers=max_workers, timeout=timeout),
        raise_exceptions=False,
        show_progress=True,
    )

    records = _extract_per_sample(evaluation, len(rows))
    executed = [
        {"family": family, "name": str(getattr(metric, "name", type(metric).__name__))}
        for family, metric in specs
    ]
    executed_names = [item["name"] for item in executed]

    summary: dict[str, float | None] = {}
    for name in executed_names:
        numbers = [
            value for value in (sanitize_float(row.get(name)) for row in records)
            if value is not None
        ]
        summary[name] = round(sum(numbers) / len(numbers), 6) if numbers else None

    headline: dict[str, Any] = {}
    for family, priority in FAMILY_PRIORITY.items():
        chosen = next((name for name in priority if name in summary), None)
        if chosen is not None:
            headline[family] = {
                "label": FAMILY_LABEL[family],
                "metric": chosen,
                "score": summary[chosen],
            }

    per_sample = [
        {
            "index": index + 1,
            "scores": {name: sanitize_float(row.get(name)) for name in executed_names},
        }
        for index, row in enumerate(records)
    ]

    return {
        "executed": executed,
        "skipped": skipped,
        "summary": summary,
        "headline": headline,
        "per_sample": per_sample,
    }


# --------------------------------------------------------------------------- #
# 可复现报告
# --------------------------------------------------------------------------- #
def build_report_markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    dataset = report["dataset"]
    config = report["retrieval_config"]
    lines: list[str] = []
    lines.append(f"# Ragas 离线评测报告 · {run['run_name']}")
    lines.append("")
    lines.append(f"- 运行 ID：`{run['run_id']}`")
    lines.append(f"- 生成时间：{run['created_at']}（耗时 {run['duration_sec']:.1f}s）")
    lines.append(
        f"- 数据集：`{dataset['path']}`，使用 {dataset['used']} 条，"
        f"sha256 `{dataset['sha256'][:12]}`"
    )
    lines.append("")
    lines.append("## 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append(run["command"])
    lines.append("```")
    lines.append("")

    lines.append("## 指标结果")
    lines.append("")
    headline = report["ragas"]["headline"]
    if headline:
        lines.append("| 指标 | 分数 | 采用 Ragas 指标 |")
        lines.append("| --- | --- | --- |")
        for _, item in headline.items():
            score = item["score"]
            lines.append(f"| {item['label']} | {score if score is not None else 'n/a'} | `{item['metric']}` |")
    else:
        lines.append("未产出指标（`--metrics off`，或全部指标被跳过）。")
    lines.append("")

    summary = report["ragas"]["summary"]
    if summary:
        lines.append("### 全部已执行指标")
        lines.append("")
        lines.append("| Ragas 指标 | 平均分 |")
        lines.append("| --- | --- |")
        for name, value in summary.items():
            lines.append(f"| `{name}` | {value if value is not None else 'n/a'} |")
        lines.append("")

    skipped = report["ragas"]["skipped"]
    if skipped:
        lines.append("### 未执行指标与原因")
        lines.append("")
        lines.append("| 指标 | 原因 |")
        lines.append("| --- | --- |")
        for item in skipped:
            lines.append(f"| `{item['metric']}` | {item['reason']} |")
        lines.append("")

    stats = report["retrieval_stats"]
    lines.append("## 检索统计")
    lines.append("")
    lines.append(f"- 检索模式：`{config['mode']}`，top_k={config['top_k']}，fusion_top_k={config['fusion_top_k']}")
    lines.append(f"- 平均召回片段数：{stats['mean_contexts']}")
    lines.append(f"- 空召回样本数：{stats['samples_without_context']}")
    if stats["hit_rate"] is not None:
        lines.append(f"- 参考片段命中率：{stats['hit_rate']}（{stats['hit_denominator']} 条有标注的样本）")
    if stats.get("precision_ceiling") is not None:
        lines.append(
            f"- `id_based_context_precision` 在该配置下的理论上限：{stats['precision_ceiling']}"
            "（每题参考片段数固定，检索条数越多上限越低）"
        )
    lines.append("")

    lines.append("## 运行环境与可复现信息")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 回答模型 | `{report['answer_config']['provider']}/{report['answer_config']['model']}`（temperature=0） |")
    lines.append(f"| 嵌入模型 | `{report['embedding_config']['model']}`（{report['embedding_config']['provider']}） |")
    lines.append(f"| 随机种子 | {run['seed']} |")
    lines.append(f"| Python | {run['python']} |")
    lines.append(f"| git | `{run['git']['commit'][:12] or 'n/a'}`（branch={run['git']['branch'] or 'n/a'}，dirty={run['git']['dirty']}） |")
    lines.append("| 依赖版本 | " + "、".join(f"{k}=={v}" for k, v in run["packages"].items()) + " |")
    lines.append("")

    index = report["index"]
    lines.append("## 索引状态")
    lines.append("")
    lines.append(f"- ready={index['ready']}，vector_ready={index['vector_ready']}，bm25_ready={index['bm25_ready']}")
    lines.append(f"- 已索引文件数：{index['indexed_files']}")
    lines.append(f"- manifest sha256：`{(index['manifest_sha256'] or 'n/a')[:12]}`")
    lines.append("")

    failures = report.get("low_score_samples") or []
    if failures:
        lines.append("## 低分样本（优先排查）")
        lines.append("")
        for item in failures:
            lines.append(f"### {item['id']}（{item['metric']} = {item['score']}）")
            lines.append("")
            lines.append(f"- 问题：{item['question']}")
            lines.append(f"- 回答：{(item['response'] or '（空）')[:400]}")
            lines.append(f"- 参考：{(item['reference'] or '（无）')[:400]}")
            lines.append(f"- 命中参考片段：{item['reference_hit']}")
            lines.append("")
    return "\n".join(lines)


def pick_low_score_samples(report: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    """按「忠实度优先」挑出最差的若干样本，便于定位检索/生成问题。"""

    headline = report["ragas"]["headline"]
    priority = [
        name for name in ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
        if name in headline
    ]
    if not priority or limit <= 0:
        return []
    metric_key = headline[priority[0]]["metric"]
    ranked = sorted(report["samples"], key=lambda item: item.get("sort_key", 1.0))
    picked: list[dict[str, Any]] = []
    for sample in ranked[:limit]:
        picked.append({
            "id": sample["id"],
            "metric": metric_key,
            "score": sample["scores"].get(metric_key),
            "question": sample["question"],
            "response": sample["response"],
            "reference": sample["reference"],
            "reference_hit": sample["reference_hit"],
        })
    return picked


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ragas 检索 + 回答离线评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="评测数据集 JSONL，相对 backend/ 或绝对路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="报告输出目录，相对 backend/")
    parser.add_argument("--run-name", default="rag_offline", help="本次运行名，用于报告文件名")
    parser.add_argument("--retriever", choices=("hybrid", "vector", "bm25", "orchestrator"),
                        default="hybrid", help="检索策略")
    parser.add_argument("--top-k", type=int, default=6, help="单路召回条数")
    parser.add_argument("--fusion-top-k", type=int, default=6, help="RRF 融合后保留条数")
    parser.add_argument("--rank-constant", type=int, default=60, help="RRF rank constant")
    parser.add_argument("--path-filter", action="append", default=[], help="限定检索路径前缀，可重复")
    parser.add_argument("--query-hint", action="append", default=[], help="附加查询词，可重复")
    parser.add_argument("--metrics", choices=("off", "deterministic", "llm", "all"), default="all",
                        help="off=只测检索；deterministic=无需 LLM 的指标；llm=LLM 指标；all=全部")
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条，0 表示全部")
    parser.add_argument("--seed", type=int, default=20261002, help="随机种子（记录并传给 Ragas）")
    parser.add_argument("--max-workers", type=int, default=4, help="Ragas 并发度")
    parser.add_argument("--timeout", type=int, default=180, help="单条指标超时秒数")
    parser.add_argument("--rebuild-index", action="store_true", help="评测前重建知识索引")
    parser.add_argument("--no-answer", action="store_true", help="不调用 LLM 作答（只测检索）")
    parser.add_argument("--low-score-limit", type=int, default=5, help="报告中低分样本条数")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    random.seed(args.seed)

    dataset_path = args.dataset if args.dataset.is_absolute() else (BACKEND_DIR / args.dataset)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (BACKEND_DIR / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path, limit=args.limit)
    if not samples:
        raise SystemExit(f"数据集为空：{dataset_path}")

    from config import get_settings
    from graph.agent import agent_manager
    from knowledge_retrieval import knowledge_indexer

    settings = get_settings()
    agent_manager.initialize(BACKEND_DIR)
    knowledge_indexer.configure(BACKEND_DIR)
    if args.rebuild_index:
        knowledge_indexer.rebuild_index()
    else:
        knowledge_indexer._ensure_loaded()

    model_builder = agent_manager._build_chat_model
    model = None if args.no_answer else model_builder()

    retrieval_config = {
        "mode": args.retriever,
        "top_k": args.top_k,
        "fusion_top_k": args.fusion_top_k,
        "rank_constant": args.rank_constant,
        "path_filters": list(args.path_filter),
        "query_hints": list(args.query_hint),
    }

    rows: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    hit_total = 0
    hit_denominator = 0
    context_total = 0
    empty_contexts = 0
    ceiling_total = 0.0
    ceiling_count = 0

    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample.question[:60]}")
        error = ""
        chunks: list[RetrievedChunk] = []
        response = ""
        try:
            chunks = await retrieve(sample.question, retrieval_config, model_builder)
            if model is not None:
                response = await generate_answer(sample.question, chunks, model)
        except Exception as exc:  # noqa: BLE001 - 单条失败不应中断整轮评测
            error = f"{type(exc).__name__}: {exc}"
            print(f"    ! {error}")

        retrieved_ids = [c.context_id for c in chunks]
        reference_hit: bool | None = None
        if sample.reference_context_ids:
            hit_denominator += 1
            reference_hit = any(cid in set(retrieved_ids) for cid in sample.reference_context_ids)
            hit_total += 1 if reference_hit else 0
        context_total += len(chunks)
        if not chunks:
            empty_contexts += 1
        if sample.reference_context_ids and retrieved_ids:
            # 参考片段数固定时，检索条数越多 id_based_context_precision 上限越低
            ceiling_count += 1
            ceiling_total += min(1.0, len(sample.reference_context_ids) / len(retrieved_ids))

        rows.append({
            "id": sample.sample_id,
            "question": sample.question,
            "response": response,
            "reference": sample.reference,
            "retrieved_contexts": [c.text for c in chunks],
            "reference_contexts": sample.reference_contexts,
            "retrieved_context_ids": retrieved_ids,
            "reference_context_ids": sample.reference_context_ids,
        })
        sample_records.append({
            "index": index,
            "id": sample.sample_id,
            "question": sample.question,
            "response": response,
            "reference": sample.reference,
            "retrieved_context_ids": retrieved_ids,
            "reference_context_ids": sample.reference_context_ids,
            "retrieved_contexts": [c.text for c in chunks],
            "reference_contexts": sample.reference_contexts,
            "chunks": [c.to_dict() for c in chunks],
            "reference_hit": reference_hit,
            "error": error,
        })

    if args.metrics == "off":
        ragas_result: dict[str, Any] = {
            "executed": [], "skipped": [], "summary": {}, "headline": {}, "per_sample": [],
        }
    else:
        ragas_result = run_ragas(
            args.metrics, rows,
            seed=args.seed,
            max_workers=args.max_workers,
            timeout=args.timeout,
            build_evaluator=not args.no_answer,
        )

    for record, scored in zip(sample_records, ragas_result.get("per_sample", [])):
        record["scores"] = scored.get("scores", {})
        values = [v for v in record["scores"].values() if v is not None]
        record["sort_key"] = min(values) if values else 1.0

    index_status = knowledge_indexer.status()
    manifest_path = BACKEND_DIR / "storage" / "knowledge" / "manifest.json"

    report: dict[str, Any] = {
        "run": {
            "run_id": f"{args.run_name}-{time.strftime('%Y%m%d-%H%M%S')}",
            "run_name": args.run_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": time.time() - started,
            "seed": args.seed,
            "python": sys.version.split()[0],
            "command": " ".join([sys.executable, *sys.argv]),
            "packages": package_versions(),
            "git": git_info(BACKEND_DIR),
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "used": len(samples),
            "limit": args.limit,
            "has_reference_contexts": any(s.reference_contexts for s in samples),
            "has_reference_context_ids": any(s.reference_context_ids for s in samples),
            "has_reference": any(s.reference for s in samples),
        },
        "retrieval_config": retrieval_config,
        "answer_config": {
            "enabled": model is not None,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "temperature": 0,
            "max_context_chars": MAX_CONTEXT_CHARS,
        },
        "embedding_config": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
        },
        "index": {
            **index_status.to_dict(),
            "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        },
        "ragas": {
            "metrics_tier": args.metrics,
            "executed": ragas_result.get("executed", []),
            "skipped": ragas_result.get("skipped", []),
            "summary": ragas_result.get("summary", {}),
            "headline": ragas_result.get("headline", {}),
        },
        "retrieval_stats": {
            "mean_contexts": round(context_total / len(samples), 4),
            "samples_without_context": empty_contexts,
            "hit_rate": round(hit_total / hit_denominator, 6) if hit_denominator else None,
            "hit_denominator": hit_denominator,
            "precision_ceiling": round(ceiling_total / ceiling_count, 6) if ceiling_count else None,
        },
        "samples": sample_records,
    }
    report["low_score_samples"] = pick_low_score_samples(report, args.low_score_limit)

    json_path = output_dir / f"{args.run_name}.report.json"
    md_path = output_dir / f"{args.run_name}.report.md"
    dataset_export = output_dir / f"{args.run_name}.ragas_dataset.jsonl"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_report_markdown(report), encoding="utf-8")
    with dataset_export.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report["report_paths"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "ragas_dataset": str(dataset_export),
    }
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"运行：{report['run']['run_id']}（{report['run']['duration_sec']:.1f}s）")
    config = report["retrieval_config"]
    print(f"检索：{config['mode']} top_k={config['top_k']} fusion_top_k={config['fusion_top_k']}")
    stats = report["retrieval_stats"]
    print(f"检索统计：平均片段 {stats['mean_contexts']}，空召回 {stats['samples_without_context']}，"
          f"命中率 {stats['hit_rate']}")
    headline = report["ragas"]["headline"]
    if headline:
        print("-" * 72)
        for _, item in headline.items():
            print(f"  {item['label']:<10} {item['score']}  ({item['metric']})")
    for item in report["ragas"]["skipped"]:
        print(f"  [skip] {item['metric']}: {item['reason']}")
    print("-" * 72)
    for key, path in report["report_paths"].items():
        print(f"  {key:<15} {path}")
    print("=" * 72)


def main() -> int:
    args = parse_args()
    report = asyncio.run(run(args))
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
