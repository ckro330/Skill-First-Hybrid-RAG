"use client";

import { ArrowRight, Workflow } from "lucide-react";

import { agentLabel, DOMAIN_LABELS, type AgentRun, type PlanInfo } from "@/lib/api";

const STATUS_META: Record<string, { label: string; badge: string }> = {
  pending: {
    label: "待执行",
    badge: "bg-[rgba(13,37,48,0.08)] text-[var(--color-ink-soft)]"
  },
  running: {
    label: "执行中",
    badge: "bg-[rgba(212,106,74,0.16)] text-[var(--color-ember)]"
  },
  done: {
    label: "已完成",
    badge: "bg-[rgba(15,139,141,0.14)] text-[var(--color-ocean)]"
  }
};

export function PlanCard({ plan, agents }: { plan: PlanInfo; agents: AgentRun[] }) {
  const tasks = plan.tasks ?? [];
  const statusByTask = new Map(agents.map((run) => [run.task_id, run.status]));
  const doneCount = tasks.filter((task) => statusByTask.get(task.task_id) === "done").length;

  return (
    <div className="mb-4 rounded-3xl border border-[rgba(13,37,48,0.12)] bg-[rgba(13,37,48,0.05)] p-4">
      <div className="flex flex-wrap items-center gap-2 text-sm font-medium text-[var(--color-ink)]">
        <Workflow className="shrink-0" size={16} />
        <span>Planner 任务拆解</span>
        <span className="rounded-full bg-white/70 px-2 py-0.5 text-[11px] font-normal text-[var(--color-ink-soft)]">
          {DOMAIN_LABELS[plan.domain] ?? plan.domain}
        </span>
        <span className="rounded-full bg-white/70 px-2 py-0.5 text-[11px] font-normal text-[var(--color-ink-soft)]">
          {plan.multi_agent
            ? `multi agent · ${tasks.length} 个子任务`
            : tasks.length
              ? "single agent"
              : "无需拆解"}
        </span>
        {tasks.length ? (
          <span className="mono text-[11px] font-normal text-[var(--color-ink-soft)]">
            {doneCount}/{tasks.length} 完成
          </span>
        ) : null}
        {plan.stage ? (
          <span className="mono text-[11px] font-normal text-[var(--color-ink-soft)]">
            {plan.stage}
          </span>
        ) : null}
      </div>

      {plan.reason ? (
        <p className="mt-2 text-xs leading-5 text-[var(--color-ink-soft)]">{plan.reason}</p>
      ) : null}

      {tasks.length ? (
        <ol className="mt-3 space-y-2">
          {tasks.map((task, index) => {
            const status = statusByTask.get(task.task_id) ?? "pending";
            const meta = STATUS_META[status] ?? STATUS_META.pending;

            return (
              <li className="rounded-2xl bg-white/70 p-3" key={task.task_id || index}>
                <div className="flex flex-wrap items-center gap-2 text-sm font-medium text-[var(--color-ink)]">
                  <span className="mono flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[rgba(13,37,48,0.08)] text-[11px]">
                    {index + 1}
                  </span>
                  <span className="truncate">{task.title || "未命名子任务"}</span>
                  <ArrowRight className="shrink-0 text-[var(--color-ink-soft)]" size={12} />
                  <span className="shrink-0 rounded-full bg-[rgba(212,106,74,0.12)] px-2 py-0.5 text-[11px] text-[var(--color-ember)]">
                    {agentLabel(task.agent_id)}
                  </span>
                  <span
                    className={`ml-auto shrink-0 rounded-full px-2 py-0.5 text-[11px] font-normal ${meta.badge}`}
                  >
                    {meta.label}
                  </span>
                </div>

                {task.instruction ? (
                  <p className="mt-2 text-xs leading-5 text-[var(--color-ink-soft)]">
                    {task.instruction}
                  </p>
                ) : null}

                <div className="mt-1 flex flex-wrap items-center gap-2">
                  <span className="mono text-[11px] text-[var(--color-ink-soft)]">
                    {task.agent_id}
                  </span>
                  {task.depends_on?.length ? (
                    <span className="mono text-[11px] text-[var(--color-ink-soft)]">
                      · 依赖 {task.depends_on.join("、")}
                    </span>
                  ) : null}
                </div>
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="mt-3 rounded-2xl bg-white/70 p-3 text-xs leading-5 text-[var(--color-ink-soft)]">
          {plan.domain === "coding"
            ? "无需拆解，按单个子任务处理。"
            : "未拆解子任务，直接走对应链路回答。"}
        </p>
      )}
    </div>
  );
}
