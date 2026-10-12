# -*- coding: utf-8 -*-
"""渲染 team_graph.py 的 LangGraph 拓扑图（节点 / 边 / 状态 / 事件）。

节点名与边定义直接对应 backend/agents/team_graph.py 的 build_team_graph()：
  add_node(supervisor, aggregate, 4 个 *_agent)，set_entry_point(supervisor)，
  add_conditional_edges(supervisor, route_after_supervisor, {aggregate, 4 个 agent 节点})，
  每个 agent 节点 add_edge(node, supervisor)，add_edge(aggregate, END)。
用法：用 backend 的 venv 运行：python docs/render_langgraph_graph.py
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1700, 1120
W2 = 1690

CREG = "C:/Windows/Fonts/msyh.ttc"
CBOLD = "C:/Windows/Fonts/msyhbd.ttc"
CMONO = "C:/Windows/Fonts/consola.ttf"


def font(size, bold=False):
    return ImageFont.truetype(CBOLD if bold else CREG, size)


def mono(size, bold=False):
    try:
        return ImageFont.truetype(CMONO, size)
    except Exception:
        return font(size, bold)


F_TITLE = font(31, True)
F_SUB = font(15)
F_NODE = font(19, True)
F_BODY = font(14)
F_SMALL = font(12)
F_TINY = font(11.5)
F_MONO = mono(14, True)
F_MONO_S = mono(11.5)
F_BOX = font(16, True)
F_LBL = font(12, True)

INK = "#0F172A"
GRAY = "#64748B"
LINE = "#475569"
BLUE = "#2563EB"
GREEN = "#16A34A"

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


def tokens(s):
    out, buf = [], ""
    for ch in s:
        if ord(ch) > 0x2E80:
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def wrap_cjk(s, f, max_w):
    lines, cur = [], ""
    for tk in tokens(s):
        cand = cur + tk
        if cur and d.textlength(cand, font=f) > max_w:
            lines.append(cur.rstrip())
            cur = tk.lstrip()
        else:
            cur = cand
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def paragraph(x, y, s, f, fill, max_w, line_h=20):
    for line in wrap_cjk(s, f, max_w):
        ltext(x, y, line, f, fill)
        y += line_h
    return y


def bullets(x, y, items, f, fill, max_w, line_h=20, gap=8):
    for it in items:
        ltext(x, y, "·", f, "#94A3B8")
        y = paragraph(x + 14, y, it, f, fill, max_w - 14, line_h) + gap
    return y


# ---------------- 标题 ----------------
ctext(W // 2, 38, "LangGraph 父子编排图 · backend/agents/team_graph.py", F_TITLE, INK)
ctext(W // 2, 76, "StateGraph(TeamState)：supervisor 为父节点，4 个下游 agent 为子节点；子节点完成后回到父节点，待办清空交给 aggregate",
      F_SUB, GRAY)

# ---------------- 主图 ----------------
MX0, MX1 = 40, 1160
rrect((MX0, 104, MX1, 1010), "#F8FAFC", "#CBD5E1", width=2, radius=18)
ltext(MX0 + 22, 128, "编译后的图（graph.compile()）", F_LBL, "#334155")

CX = 600
rrect((520, 118, 680, 162), "#F1F5F9", "#94A3B8")
ctext(600, 140, "START", mono(14, True), "#475569")
arrow((600, 162), (600, 217))

rrect((440, 217, 760, 303), "#DBEAFE", "#2563EB", width=3)
ctext(600, 244, "supervisor", F_NODE, "#1D4ED8")
ctext(600, 272, "父 agent（planner）· 选下一个就绪子任务", F_SMALL, "#1E40AF")

arrow((600, 303), (600, 368))
rrect((450, 368, 750, 436), "#FEF3C7", "#F59E0B", width=2)
ctext(600, 391, "条件边 route_after_supervisor", F_BODY, "#92400E")
ctext(600, 415, "current_task_id 空 → aggregate，否则 → 对应子节点", F_TINY, "#B45309")

d.line([(600, 436), (600, 470)], fill=BLUE, width=2)
d.line([(240, 470), (960, 470)], fill=BLUE, width=2)
ltext(975, 458, "pending 中依赖已满足者", F_TINY, BLUE)

CH_Y0, CH_Y1, CH_W = 560, 680, 210
CH_XS = [135, 375, 615, 855]
CENTERS = [x + CH_W // 2 for x in CH_XS]
CHILDREN = [
    ("log_analysis_agent", "log-analysis-agent", "log-analysis"),
    ("code_modify_agent", "code-modify-agent", "code-modification"),
    ("test_agent", "test-agent", "test-execution, test-authoring"),
    ("git_agent", "git-agent", "git-management"),
]
for cx in CENTERS:
    arrow((cx, 470), (cx, CH_Y0), BLUE)

for x0, (node, agent_id, skill) in zip(CH_XS, CHILDREN):
    x1 = x0 + CH_W
    rrect((x0, CH_Y0, x1, CH_Y1), "#E0E7FF", "#4F46E5")
    ctext((x0 + x1) // 2, CH_Y0 + 26, node, F_MONO, "#1E3A8A")
    ctext((x0 + x1) // 2, CH_Y0 + 56, agent_id, F_MONO_S, "#4338CA")
    ltext(x0 + 12, CH_Y0 + 88, "Skill", F_TINY, GRAY)
    y = CH_Y0 + 108
    for line in wrap_cjk(skill, F_TINY, CH_W - 24):
        ltext(x0 + 12, y, line, F_TINY, "#334155")
        y += 16

# 回环：子节点 -> supervisor
RET_Y = 730
for cx in CENTERS:
    d.line([(cx, CH_Y1), (cx, RET_Y)], fill=GREEN, width=2)
d.line([(240, RET_Y), (960, RET_Y)], fill=GREEN, width=2)
d.line([(240, RET_Y), (87, RET_Y)], fill=GREEN, width=2)
d.line([(87, RET_Y), (87, 260)], fill=GREEN, width=2)
arrow((87, 260), (440, 260), GREEN, 2, 10)
ltext(105, RET_Y + 20, "每个子节点 add_edge(node, supervisor)：完成后回到父节点取下一个", F_TINY, GREEN)

# 条件边：supervisor -> aggregate
poly_arrow([(760, 260), (1112, 260), (1112, 863), (760, 863)], BLUE)
ltext(700, 840, "pending 空 → aggregate", F_TINY, BLUE)

rrect((440, 820, 760, 906), "#DCFCE7", "#16A34A", width=3)
ctext(600, 847, "aggregate", F_NODE, "#15803D")
ctext(600, 875, "父 agent 收尾：汇总子任务结果", F_SMALL, "#166534")
arrow((600, 906), (600, 957))

rrect((520, 957, 680, 1003), "#F1F5F9", "#94A3B8")
ctext(600, 980, "END", mono(14, True), "#475569")

ctext((MX0 + MX1) // 2, 1035,
      "──→ 条件边 add_conditional_edges（route_after_supervisor）　──→ 普通边 add_edge　□ 节点",
      F_TINY, GRAY)

# ---------------- 右栏 ----------------
RX0, RX1 = 1180, 1660
RW = RX1 - RX0

# State
rrect((RX0, 118, RX1, 452), "#FFFFFF", "#CBD5E1")
ltext(RX0 + 18, 142, "TeamState（TypedDict，total=False）", F_BOX, "#0F172A")
STATE = [
    ("request", "用户原始请求"),
    ("tasks", "planner 拆解出的 TaskSpec 列表"),
    ("pending", "待派发的 task_id 队列（单调递减）"),
    ("current_task_id", "本轮选中的子任务"),
    ("upstream_text", "依赖任务产出，注入子 prompt 的 Upstream Findings"),
    ("results", "task_id → 子 agent 输出（供下游与汇总使用）"),
    ("conversation_context", "历史 / 记忆上下文"),
    ("dispatched", "已派发轮数（第 2 轮起发 new_response 分段）"),
    ("final_answer", "aggregate 产出的最终答复"),
]
y = 172
for name, desc in STATE:
    ltext(RX0 + 18, y, name, mono(12, True), "#1D4ED8")
    paragraph(RX0 + 168, y, desc, F_TINY, "#334155", RW - 186, line_h=15)
    y += 31

# 节点写入 / 事件
rrect((RX0, 472, RX1, 706), "#FFFFFF", "#CBD5E1")
ltext(RX0 + 18, 496, "节点职责与事件（get_stream_writer）", F_BOX, "#0F172A")
NODES = [
    ("supervisor", "读 tasks / results / pending；写 current_task_id、upstream_text、dispatched；第 2 轮起发 new_response 分段"),
    ("*_agent ×4", "驱动 AgentExecutor 子图；写 results[task_id] 并清空 current_task_id；透传 agent_start / token / tool_start / tool_end / a2a / agent_result"),
    ("aggregate", "写 final_answer；发 new_response 与内部事件 _team_final"),
    ("异常路径", "子 agent 抛异常被捕获：发 error 事件、补一条 failed=true 的 A2A result，不拖垮父图"),
]
y = 526
for name, desc in NODES:
    ltext(RX0 + 18, y, name, mono(12, True), "#1D4ED8")
    y = paragraph(RX0 + 132, y, desc, F_TINY, "#334155", RW - 175, line_h=16) + 10

# 终止性
rrect((RX0, 726, RX1, 906), "#FFFFFF", "#CBD5E1")
ltext(RX0 + 18, 750, "终止性与健壮性", F_BOX, "#0F172A")
bullets(RX0 + 18, 782, [
    "order_tasks 稳定拓扑排序；检测到环则退回原始顺序",
    "每轮只派发 1 个就绪任务，pending 单调递减",
    "无可执行任务（缺依赖 / 未知 agent）→ 标记失败并清空 pending，图自然收尾",
], F_TINY, "#334155", RW - 36, line_h=16, gap=6)

# 驱动
rrect((RX0, 926, RX1, 1050), "#FFFFFF", "#CBD5E1")
ltext(RX0 + 18, 950, "驱动方 CodingOrchestrator.astream()", F_BOX, "#0F172A")
bullets(RX0 + 18, 982, [
    "order_tasks → new A2ABus() → graph.astream(initial, stream_mode=[\"custom\"])",
    "转发除 _team_final 外的全部事件；末尾补 a2a_transcript + done",
], F_TINY, "#334155", RW - 36, line_h=16, gap=6)

out = ROOT / "docs" / "langgraph-team-graph.png"
img.save(out)
print("SAVED:", out, img.size)
