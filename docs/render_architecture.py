# -*- coding: utf-8 -*-
"""渲染「代码助手多 Agent 架构」图（Planner 领域分流 + LangGraph 父子编排）。

Skill / Tool 数据直接从 backend/agents/registry.py 读取，保证图与代码一致。
用法：用 backend 的 venv 运行：python docs/render_architecture.py
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from agents.registry import DOWNSTREAM_AGENTS  # noqa: E402

_cfg = (BACKEND / "config.py").read_text(encoding="utf-8")
MIN_SCORE = "0.55"
_m = re.search(r"memory_retrieval_min_score:\s*float\s*=\s*([0-9.]+)", _cfg)
if _m:
    MIN_SCORE = _m.group(1)

W, H = 1560, 1780

CREG = "C:/Windows/Fonts/msyh.ttc"
CBOLD = "C:/Windows/Fonts/msyhbd.ttc"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(CBOLD if bold else CREG, size)


F_TITLE = font(31, True)
F_SUB = font(15)
F_BOX = font(19, True)
F_BODY = font(14)
F_SMALL = font(12)
F_TINY = font(11.5)
F_ID = font(17, True)
F_NAME = font(15)
F_PILL = font(13)

INK = "#0F172A"
GRAY = "#64748B"
LINE = "#475569"
PANEL_BG = "#F8FAFC"
PANEL_EDGE = "#CBD5E1"

img = Image.new("RGB", (W, H), "#FFFFFF")
d = ImageDraw.Draw(img)


def rrect(box, fill, outline, width=2, radius=13):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def ctext(cx, cy, s, f, fill):
    d.text((cx, cy), s, font=f, fill=fill, anchor="mm")


def ltext(x, y, s, f, fill):
    d.text((x, y), s, font=f, fill=fill, anchor="lm")


def arrow(p1, p2, color=LINE, width=2, head=10):
    d.line([p1, p2], fill=color, width=width)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    for sign in (-1, 1):
        a = ang + sign * math.radians(152)
        d.line([p2, (p2[0] + head * math.cos(a), p2[1] + head * math.sin(a))],
               fill=color, width=width)


def poly_arrow(pts, color=LINE, width=2, head=10):
    for i in range(len(pts) - 2):
        d.line([pts[i], pts[i + 1]], fill=color, width=width)
    arrow(pts[-2], pts[-1], color, width, head)


def darrow(p1, p2, color=LINE, width=2, head=8):
    d.line([p1, p2], fill=color, width=width)
    for p, q in ((p1, p2), (p2, p1)):
        ang = math.atan2(q[1] - p[1], q[0] - p[0])
        for sign in (-1, 1):
            a = ang + sign * math.radians(152)
            d.line([p, (p[0] + head * math.cos(a), p[1] + head * math.sin(a))],
                   fill=color, width=width)


def wrap(s, f, max_w):
    parts = s.split(", ")
    lines, cur = [], ""
    for p in parts:
        cand = p if not cur else cur + ", " + p
        if d.textlength(cand, font=f) <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = p
    if cur:
        lines.append(cur)
    return lines


# ---------------- 标题 ----------------
ctext(W // 2, 40, "代码助手 · 多 Agent 架构", F_TITLE, INK)
ctext(W // 2, 78, "Planner 领域分流 ＋ LangGraph 父子编排（父 agent supervisor 与 4 个下游 Agent 协作，A2A 通信）",
      F_SUB, GRAY)

# ---------------- 外层链路 ----------------
CX = 780
rrect((550, 120, 1010, 180), "#F1F5F9", "#94A3B8")
ctext(CX, 150, "用户请求（前端 Chat）", F_BOX, INK)
arrow((CX, 180), (CX, 218))

rrect((470, 218, 1090, 290), "#F1F5F9", "#94A3B8")
ctext(CX, 243, "AgentManager.astream()  ·  SSE 流式入口", F_BOX, INK)
ctext(CX, 268, "会话历史 · 压缩上下文 · 事件透传", F_SMALL, GRAY)
arrow((CX, 290), (CX, 328))

rrect((400, 328, 1160, 394), "#F5F3FF", "#8B5CF6")
ctext(CX, 351, "长期记忆检索  memory_manager", F_BOX, "#5B21B6")
ctext(CX, 374, f"min_score={MIN_SCORE} · top_k=3 · 命中则注入上下文（三条链路共用）", F_SMALL, "#7C3AED")
arrow((CX, 394), (CX, 432))

# ---------------- Planner ----------------
rrect((230, 432, 1330, 548), "#EEF2FF", "#6366F1")
ctext(CX, 462, "Planner（外层）· 领域判定 ＋ 任务拆解", F_BOX, "#3730A3")
ctext(CX, 492, "LLM 分类 ＋ heuristic 兜底  →  PlanDecision{domain, multi_agent, tasks[]}", F_BODY, "#4338CA")
ctext(CX, 522, "multi_agent=true 拆多任务并标注 depends_on ｜ 单一任务走 single-agent", F_SMALL, GRAY)

# 分流
d.line([(CX, 548), (CX, 572)], fill=LINE, width=2)
d.line([(250, 572), (1310, 572)], fill=LINE, width=2)
for bx in (250, 780, 1310):
    arrow((bx, 572), (bx, 596))

BY0, BY1 = 596, 716
rrect((60, BY0, 440, BY1), "#ECFEFF", "#0891B2")
ctext(250, BY0 + 26, "RAG 领域（知识库）", F_BOX, "#155E75")
ctext(250, BY0 + 56, "knowledge_orchestrator + rag-skill", F_SMALL, "#0E7490")
ctext(250, BY0 + 84, "检索证据 → 受限作答，不编造", F_SMALL, GRAY)

rrect((560, BY0, 1000, BY1), "#E0E7FF", "#4F46E5", width=3)
ctext(780, BY0 + 28, "Coding 领域", F_BOX, "#3730A3")
ctext(780, BY0 + 60, "LangGraph 父子多 Agent 编排", F_BODY, "#4338CA")
ctext(780, BY0 + 90, "见下方展开  ▼", F_SMALL, "#6366F1")

rrect((1120, BY0, 1500, BY1), "#FDF2F8", "#DB2777")
ctext(1310, BY0 + 26, "通用闲聊", F_BOX, "#9D174D")
ctext(1310, BY0 + 56, "直连 LLM 作答", F_SMALL, "#BE185D")
ctext(1310, BY0 + 84, "不进入工具链路", F_SMALL, GRAY)

arrow((780, BY1), (780, 756))

# ---------------- LangGraph 面板 ----------------
PX0, PY0, PX1, PY1 = 40, 756, 1520, 1612
rrect((PX0, PY0, PX1, PY1), PANEL_BG, PANEL_EDGE, width=2, radius=18)
ctext(780, 786, "LangGraph StateGraph(TeamState)  ·  父 agent 与子 agent 构成父子图", F_BODY, "#334155")

SUP_Y0, SUP_Y1 = 816, 888
rrect((520, SUP_Y0, 1040, SUP_Y1), "#DBEAFE", "#2563EB", width=3)
ctext(780, 840, "supervisor · 父 agent（planner）", F_BOX, "#1D4ED8")
ctext(780, 866, "order_tasks 拓扑排序 · 依赖满足即派发 · 未就绪不派发", F_SMALL, "#1E40AF")

# 条件边
d.line([(780, SUP_Y1), (780, 908)], fill=LINE, width=2)
d.line([(279, 908), (1281, 908)], fill=LINE, width=2)
ltext(800, 894, "route_after_supervisor：每轮选 1 个就绪子任务，完成后回到 supervisor", F_SMALL, GRAY)

CH_Y0, CH_Y1 = 948, 1258
CH_W = 310
CH_XS = [124, 458, 792, 1126]
CENTERS = [x + CH_W // 2 for x in CH_XS]
for cx in CENTERS:
    arrow((cx, 908), (cx, CH_Y0))

for x0, agent in zip(CH_XS, DOWNSTREAM_AGENTS):
    x1 = x0 + CH_W
    rrect((x0, CH_Y0, x1, CH_Y1), "#FFFFFF", "#94A3B8")
    d.rounded_rectangle((x0, CH_Y0, x1, CH_Y0 + 34), radius=13, fill="#E2E8F0")
    d.rectangle((x0, CH_Y0 + 22, x1, CH_Y0 + 34), fill="#E2E8F0")
    ltext(x0 + 14, CH_Y0 + 18, agent.agent_id, F_ID, "#1E3A8A")
    ltext(x0 + 14, CH_Y0 + 50, agent.name, F_NAME, GRAY)
    d.line([(x0 + 14, CH_Y0 + 68), (x1 - 14, CH_Y0 + 68)], fill="#E2E8F0", width=1)

    ltext(x0 + 14, CH_Y0 + 84, "Skill", F_SMALL, GRAY)
    y = CH_Y0 + 104
    for line in wrap(", ".join(agent.skills), F_BODY, CH_W - 28):
        ltext(x0 + 14, y, line, F_BODY, "#0F172A")
        y += 20

    ltext(x0 + 14, CH_Y0 + 136, f"Tools（{len(agent.tools)}）", F_SMALL, GRAY)
    y = CH_Y0 + 156
    for line in wrap(", ".join(agent.tools), F_TINY, CH_W - 28):
        ltext(x0 + 14, y, line, F_TINY, "#334155")
        y += 18

    d.line([(x0 + 14, CH_Y0 + 240), (x1 - 14, CH_Y0 + 240)], fill="#E2E8F0", width=1)
    ltext(x0 + 14, CH_Y0 + 258, "内部：ReAct 子图", F_SMALL, "#334155")
    ltext(x0 + 14, CH_Y0 + 278, "model ←→ tools（max_steps 限制）", F_TINY, GRAY)

# A2A 总线
rrect((124, 1298, 1436, 1352), "#FEF3C7", "#F59E0B")
ctext(780, 1317, "A2A 消息总线", F_BODY, "#92400E")
ctext(780, 1338, "planner ←→ 子 agent（task 派发 / result 回传）· 每次请求独立新建，不跨会话复用", F_SMALL, "#B45309")
for cx in CENTERS:
    darrow((cx, CH_Y1), (cx, 1298), "#F59E0B", width=2, head=7)

# 左侧回环：supervisor -> aggregate
poly_arrow([(520, 852), (82, 852), (82, 1430), (520, 1430)])

rrect((520, 1388, 1040, 1472), "#DCFCE7", "#16A34A", width=3)
ctext(780, 1412, "aggregate · 父 agent 收尾", F_BOX, "#15803D")
ctext(780, 1438, "单任务直返 ｜ 多任务 LLM 汇总 ｜ 失败项如实保留", F_SMALL, "#166534")
arrow((780, 1472), (780, 1512))

rrect((430, 1512, 1130, 1580), "#EDE9FE", "#7C3AED")
ctext(780, 1536, "最终答复", F_BOX, "#5B21B6")
ctext(780, 1560, "a2a_transcript + done → SSE 回传前端", F_SMALL, "#6D28D9")

# ---------------- SSE 事件序列 ----------------
EVENTS = ["planner", "plan", "agent_start", "a2a", "token", "tool_start/end",
          "agent_result", "new_response", "a2a_transcript", "done"]
label = "SSE 事件序列"
pad_x, gap = 12, 8
widths = [int(d.textlength(e, font=F_PILL)) + pad_x * 2 for e in EVENTS]
label_w = int(d.textlength(label, font=F_SMALL)) + 14
total = label_w + sum(widths) + gap * (len(EVENTS) - 1)
x = (W - total) // 2
y0, y1 = 1640, 1676
ltext(x, (y0 + y1) // 2, label, F_SMALL, GRAY)
x += label_w
for e, w in zip(EVENTS, widths):
    d.rounded_rectangle((x, y0, x + w, y1), radius=18, fill="#F1F5F9", outline="#CBD5E1", width=1)
    ctext(x + w // 2, (y0 + y1) // 2, e, F_PILL, "#334155")
    x += w + gap

ctext(W // 2, 1722, "图例：──→ LangGraph 编排流转（supervisor 派发 / aggregate 收尾）　　←→ A2A 任务与结果消息　　子 agent 各自驱动自己的 ReAct 子图",
      F_SMALL, GRAY)

out = ROOT / "docs" / "architecture-multi-agent.png"
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out)
print("SAVED:", out, img.size)
