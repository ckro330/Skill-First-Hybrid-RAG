from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings
from tools.skills_scanner import scan_skills


# ---------- L1：指令占位 ----------

COMMANDS: dict[str, str] = {
    "/help": "显示可用指令列表",
    "/skills": "列出当前可用的 Skill",
    "/exit": "结束当前会话（占位实现）",
}

UNKNOWN_COMMAND_MESSAGE = "未知指令，输入 /help 查看可用指令。"


# ---------- L2：混合检索参数 ----------

SKILL_HEAD_LINES = 8  # SKILL.md 正文开头行数，作为检索源的一部分
RRF_RANK_CONSTANT = 60
WEIGHT_VECTOR = 0.5
WEIGHT_BM25 = 0.5


# ---------- L3：路由 LLM 提示词 ----------

ROUTER_SYSTEM_PROMPT_ALL = """你是一个 Skill 路由器，负责在 Agent 执行任务之前，从可用的 Skill 列表中选择最合适的一个。

可用的 Skill 列表：
{skill_menu}

路由规则：
1. 根据用户的问题，选择最合适的 Skill。
2. 只返回 Skill 名称（即列表第一列的 ID），不要解释，不要输出任何其他内容。
3. 如果没有合适的 Skill，只返回 NONE。
4. 当用户问题同时适用于多个 Skill 时，选择匹配度最高的一个。"""


ROUTER_SYSTEM_PROMPT_CANDIDATES = """你是一个 Skill 路由器。语义检索已经筛出候选 Skill，请从候选列表中选择与用户问题最匹配的一个。

候选 Skill 列表：
{skill_menu}

路由规则：
1. 只能从候选 Skill 列表中选择一个。
2. 只返回 Skill 名称（即列表第一列的 ID），不要解释，不要输出任何其他内容。
3. 如果候选 Skill 都不合适，只返回 NONE。"""


ROUTER_HUMAN_PROMPT = "用户问题：{query}\n\n请只返回最合适的 Skill ID，或 NONE。"


_NONE_MARKERS = {"NONE", "NULL", "NO", "N/A", "NA", "无", "没有", "不适用", "无合适技能"}


@dataclass
class RouteDecision:
    """三层路由的最终决策。kind: command / skill / none。"""

    kind: str
    skill_id: str | None = None
    command: str | None = None
    stage: str = ""  # 决定层：L1 / L2 / L3 / fallback
    candidates: list[str] = field(default_factory=list)
    reason: str = ""


class SkillRouter:
    """三层技能路由：L1 指令判断 → L2 向量/BM25 混合检索 → L3 轻量 LLM 决策。"""

    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self._model_builder: Callable[[], Any] | None = None
        self._embedding_model: Any | None = None

    def configure(self, base_dir: Path, model_builder: Callable[[], Any]) -> None:
        self.base_dir = base_dir
        self._model_builder = model_builder

    # ---------- L1：指令判断层（占位） ----------

    def _route_command(self, query: str) -> RouteDecision | None:
        stripped = query.strip()
        if not stripped.startswith("/"):
            return None
        command = stripped.split()[0].lower()
        if command in COMMANDS:
            return RouteDecision(
                kind="command",
                command=command,
                stage="L1",
                reason=f"用户输入以 / 开头，命中已注册指令 {command}。",
            )
        return RouteDecision(
            kind="command",
            command=command,
            stage="L1",
            reason=f"用户输入以 / 开头，但指令 {command} 未注册。",
        )

    def handle_command(self, command: str) -> str:
        """占位指令处理：不调用 LLM，直接返回确定性文本。"""
        if command == "/help":
            lines = ["可用指令："]
            lines.extend(f"- {name}: {description}" for name, description in COMMANDS.items())
            lines.append("其他以 / 开头的输入会提示未知指令。")
            return "\n".join(lines)
        if command == "/skills":
            menu = self._build_menu()
            return f"当前可用 Skill：\n{menu}"
        if command == "/exit":
            return "会话结束指令为占位实现，当前不会真的销毁会话。"
        return f"{UNKNOWN_COMMAND_MESSAGE}（{command}）"

    # ---------- L2：向量 + BM25 + RRF 混合检索 ----------

    def _skill_source_text(self, skill: Any) -> str:
        """检索源 = name + description + SKILL.md 正文开头几行。"""
        if self.base_dir is None:
            raise RuntimeError("SkillRouter is not configured")
        text = f"{skill.name}\n{skill.description}"
        skill_path = self.base_dir / skill.path
        try:
            body = skill_path.read_text(encoding="utf-8")
            frontmatter = re.match(r"^---\n.*?\n---\n?", body, re.DOTALL)
            if frontmatter:
                body = body[frontmatter.end() :]
            head = "\n".join(body.splitlines()[:SKILL_HEAD_LINES])
            text += "\n" + head
        except OSError:
            pass
        return text

    def _tokenize(self, text: str) -> list[str]:
        """英文按词、中文按单字 + 相邻双字切分，供 BM25 使用。"""
        lowered = text.lower()
        tokens: list[str] = re.findall(r"[a-z0-9_]+", lowered)
        for block in re.findall(r"[\u4e00-\u9fff]+", lowered):
            tokens.extend(block)
            tokens.extend(block[i : i + 2] for i in range(len(block) - 1))
        return tokens

    def _vector_scores(self, query: str, corpus: list[str]) -> list[float] | None:
        """sentence-transformers 向量相似度，映射到 [0, 1]；不可用时返回 None。"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        try:
            if self._embedding_model is None:
                self._embedding_model = SentenceTransformer(get_settings().skill_embedding_model)
            embeddings = self._embedding_model.encode(
                [query, *corpus],
                normalize_embeddings=True,
            )
            query_vec = embeddings[0]
            return [
                (float(query_vec @ embeddings[i]) + 1.0) / 2.0
                for i in range(1, len(embeddings))
            ]
        except Exception:
            return None

    def _bm25_scores(self, query: str, corpus: list[str]) -> list[float] | None:
        """rank_bm25 词法打分，按最大值归一化到 [0, 1]；不可用时返回 None。"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return None
        try:
            bm25 = BM25Okapi([self._tokenize(text) for text in corpus])
            raw_scores = bm25.get_scores(self._tokenize(query))
            scores = [max(0.0, float(score)) for score in raw_scores]
            max_score = max(scores) if scores else 0.0
            if max_score <= 0:
                return [0.0] * len(scores)
            return [score / max_score for score in scores]
        except Exception:
            return None

    def _rrf_order(self, vector: list[float] | None, bm25: list[float] | None) -> list[int]:
        """对两个打分列表做 RRF 排序，返回技能索引（从高到低）。"""
        size = len(vector) if vector is not None else len(bm25 or [])
        rrf_scores: dict[int, float] = {i: 0.0 for i in range(size)}
        for score_list in (vector, bm25):
            if score_list is None:
                continue
            ranked = sorted(range(size), key=lambda i: score_list[i], reverse=True)
            for rank, index in enumerate(ranked, start=1):
                rrf_scores[index] += 1.0 / (RRF_RANK_CONSTANT + rank)
        return [
            index
            for index, _ in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        ]

    def _retrieve_candidates(self, query: str) -> list[tuple[Any, float]]:
        """L2 混合检索：返回按 RRF 排序、且组合分超过阈值的 Top 3 候选。"""
        if self.base_dir is None:
            raise RuntimeError("SkillRouter is not configured")
        records = scan_skills(self.base_dir)
        if not records:
            return []

        corpus = [self._skill_source_text(record) for record in records]
        vector_scores = self._vector_scores(query, corpus)
        bm25_scores = self._bm25_scores(query, corpus)
        if vector_scores is None and bm25_scores is None:
            return []

        combined: list[float] = []
        for index in range(len(records)):
            vector = vector_scores[index] if vector_scores is not None else 0.0
            bm25 = bm25_scores[index] if bm25_scores is not None else 0.0
            if vector_scores is not None and bm25_scores is not None:
                combined.append(WEIGHT_VECTOR * vector + WEIGHT_BM25 * bm25)
            else:
                combined.append(vector if vector_scores is not None else bm25)

        threshold = get_settings().skill_route_threshold
        ordered = self._rrf_order(vector_scores, bm25_scores)
        candidates = [
            (records[index], combined[index])
            for index in ordered
            if combined[index] >= threshold
        ]
        return candidates[:3]

    # ---------- L3：轻量 LLM 路由 ----------

    def _build_menu(
        self,
        records: list[Any] | None = None,
        *,
        include_negative: bool = True,
    ) -> str:
        if self.base_dir is None:
            raise RuntimeError("SkillRouter is not configured")
        lines: list[str] = []
        for skill in records or scan_skills(self.base_dir):
            line = f"- {skill.id}（{skill.name}）: {skill.description}"
            if include_negative and skill.negative_triggers:
                line += f"；负向触发：{skill.negative_triggers}"
            lines.append(line)
        return "\n".join(lines)

    async def _route_llm(
        self,
        query: str,
        records: list[Any],
        *,
        only_name_description: bool,
    ) -> str:
        if self._model_builder is None:
            raise RuntimeError("SkillRouter is not configured")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    ROUTER_SYSTEM_PROMPT_ALL
                    if only_name_description
                    else ROUTER_SYSTEM_PROMPT_CANDIDATES,
                ),
                ("human", ROUTER_HUMAN_PROMPT),
            ]
        )
        chain = prompt | self._model_builder() | StrOutputParser()
        raw = await chain.ainvoke(
            {
                "skill_menu": self._build_menu(
                    records,
                    include_negative=not only_name_description,
                ),
                "query": query,
            }
        )
        return self._normalize(raw, records)

    def _normalize(self, raw: str, records: list[Any]) -> str:
        """把模型输出规整成候选内的技能 ID 或 NONE。"""
        text = raw.strip()
        text = re.sub(r"```[a-zA-Z]*", "", text).strip("` \t\"'“”‘’。.，,；;")
        if not text:
            return "NONE"
        if text.upper() in _NONE_MARKERS:
            return "NONE"

        lowered = text.lower()
        for skill in records:
            if lowered == skill.id.lower() or text == skill.name:
                return skill.id

        tokens = re.split(r"[\s,，、;；:：()（）\[\]【】]+", text)
        for token in tokens:
            token = token.strip("`\"'“”‘’")
            token_lower = token.lower()
            if not token_lower:
                continue
            for skill in records:
                if token_lower == skill.id.lower() or token == skill.name:
                    return skill.id

        for skill in records:
            if skill.id.lower() in lowered:
                return skill.id

        return "NONE"

    # ---------- 总入口 ----------

    async def route(self, query: str) -> RouteDecision:
        """三层路由：L1 → L2 → L3，返回最终决策。"""
        if self.base_dir is None or self._model_builder is None:
            raise RuntimeError("SkillRouter is not configured")

        # L1：指令判断层（占位）
        command_decision = self._route_command(query)
        if command_decision is not None:
            return command_decision

        # L2：向量 + BM25 + RRF 语义检索
        candidates = self._retrieve_candidates(query)
        candidate_ids = [record.id for record, _ in candidates]

        if not candidates:
            # L2 未命中 → L3 从全部技能中选择，只传 name + description
            all_records = scan_skills(self.base_dir)
            skill_id = await self._route_llm(
                query,
                all_records,
                only_name_description=True,
            )
            return self._build_skill_decision(
                skill_id,
                stage="L3",
                candidates=[],
                reason="L2 未筛选出候选技能，L3 从全部技能中选择。",
            )

        if len(candidates) == 1:
            # 唯一命中：直接采用，不调用路由 LLM
            record, score = candidates[0]
            return RouteDecision(
                kind="skill",
                skill_id=record.id,
                stage="L2",
                candidates=candidate_ids,
                reason=f"L2 检索唯一命中「{record.id}」（score={score:.3f}），直接采用。",
            )

        # 2 或 3 个候选 → L3 在候选内二选一/三选一
        skill_id = await self._route_llm(
            query,
            [record for record, _ in candidates],
            only_name_description=False,
        )
        return self._build_skill_decision(
            skill_id,
            stage="L3",
            candidates=candidate_ids,
            reason=f"L2 筛选出 {len(candidates)} 个候选，L3 在候选内决策。",
        )

    def _build_skill_decision(
        self,
        skill_id: str,
        *,
        stage: str,
        candidates: list[str],
        reason: str,
    ) -> RouteDecision:
        if skill_id == "NONE":
            return RouteDecision(
                kind="none",
                stage=stage,
                candidates=candidates,
                reason=f"{reason} L3 判定候选均不合适。",
            )
        return RouteDecision(
            kind="skill",
            skill_id=skill_id,
            stage=stage,
            candidates=candidates,
            reason=reason,
        )


skill_router = SkillRouter()
