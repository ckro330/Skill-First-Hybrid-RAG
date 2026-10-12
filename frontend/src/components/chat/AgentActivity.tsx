"use client";

import { Bot, Loader2, Network } from "lucide-react";

import { A2A_KIND_LABELS, agentLabel, type A2AMessage, type AgentRun } from "@/lib/api";

function a2aSummary(message: A2AMessage) {
  const content = message.content.trim().replace(/\s+/g, " ");
  return content.length > 120 ? `${content.slice(0, 120)}...` : content;
}

function A2AList({ messages }: { messages: A2AMessage[] }) {
  return (
    <div className="mt-2 space-y-2">
      {messages.map((message, index) => (
        <div
          className="rounded-2xl bg-white/70 p-2.5"
          key={message.message_id || `${message.task_id}-${index}`}
        >
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-[var(--color-ink-soft)]">
            <span className="font-medium text-[var(--color-ink)]">
              {agentLabel(message.from)}
            </span>
            <span>→</span>
            <span className="font-medium text-[var(--color-ink)]">{agentLabel(message.to)}</span>
            <span className="rounded-full bg-[rgba(13,37,48,0.08)] px-2 py-0.5">
              {A2A_KIND_LABELS[message.kind] ?? message.kind}
            </span>
            {message.task_id ? <span className="mono">{message.task_id}</span> : null}
          </div>
          {a2aSummary(message) ? (
            <p className="mt-1 text-xs leading-5 text-[var(--color-ink)]">
              {a2aSummary(message)}
            </p>
          ) : null}
        </div>
      ))}
    </div>
  );
}

export function AgentActivity({
  agents,
  a2a
}: {
  agents: AgentRun[];
  a2a: A2AMessage[];
}) {
  if (!agents.length && !a2a.length) {
    return null;
  }

  return (
    <div className="mb-4">
      {agents.map((run, index) => {
        const isRunning = run.status === "running";

        return (
          <div
            className="mb-2 flex flex-wrap items-center gap-2 rounded-2xl border border-[rgba(212,106,74,0.18)] bg-[rgba(212,106,74,0.08)] px-3 py-2 text-sm"
            key={run.task_id || `${run.agent_id}-${index}`}
          >
            {isRunning ? (
              <Loader2 className="shrink-0 animate-spin text-[var(--color-ember)]" size={16} />
            ) : (
              <Bot className="shrink-0 text-[var(--color-ember)]" size={16} />
            )}
            <span className="font-medium text-[var(--color-ink)]">
              {run.agent_name || agentLabel(run.agent_id)}
            </span>
            {run.title ? (
              <span className="truncate text-[var(--color-ink-soft)]">· {run.title}</span>
            ) : null}
            <span
              className={`ml-auto shrink-0 rounded-full px-2 py-0.5 text-[11px] ${
                isRunning
                  ? "bg-[rgba(212,106,74,0.16)] text-[var(--color-ember)]"
                  : "bg-[rgba(15,139,141,0.12)] text-[var(--color-ocean)]"
              }`}
            >
              {isRunning ? "运行中" : "已完成"}
            </span>
          </div>
        );
      })}

      {a2a.length ? (
        <details className="rounded-2xl border border-[rgba(13,37,48,0.12)] bg-[rgba(13,37,48,0.04)] p-3">
          <summary className="flex cursor-pointer list-none items-center gap-2 text-xs font-medium text-[var(--color-ink-soft)]">
            <Network size={14} />
            A2A 消息 {a2a.length} 条
          </summary>
          <A2AList messages={a2a} />
        </details>
      ) : null}
    </div>
  );
}

export function A2ATrace({ messages }: { messages: A2AMessage[] }) {
  if (!messages.length) {
    return null;
  }

  return (
    <details className="mb-4 rounded-3xl border border-[rgba(13,37,48,0.12)] bg-[rgba(13,37,48,0.04)] p-4">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-sm font-medium text-[var(--color-ink)]">
        <Network size={16} />
        A2A 通信记录 {messages.length} 条
      </summary>
      <A2AList messages={messages} />
    </details>
  );
}
