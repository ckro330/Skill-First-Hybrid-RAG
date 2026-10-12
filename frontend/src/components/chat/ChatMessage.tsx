"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { A2ATrace, AgentActivity } from "@/components/chat/AgentActivity";
import { RetrievalCard } from "@/components/chat/RetrievalCard";
import { ThoughtChain } from "@/components/chat/ThoughtChain";
import type { A2AMessage, AgentRun, RetrievalStep, ToolCall } from "@/lib/api";

export function ChatMessage({
  role,
  content,
  toolCalls,
  retrievalSteps,
  agents,
  a2a,
  a2aTranscript
}: {
  role: "user" | "assistant";
  content: string;
  toolCalls: ToolCall[];
  retrievalSteps: RetrievalStep[];
  agents: AgentRun[];
  a2a: A2AMessage[];
  a2aTranscript: A2AMessage[];
}) {
  const isUser = role === "user";

  return (
    <article
      className={`max-w-[90%] rounded-[28px] px-5 py-4 ${
        isUser
          ? "ml-auto bg-[rgba(13,37,48,0.92)] text-white"
          : "panel mr-auto text-[var(--color-ink)]"
      }`}
    >
      {!isUser && <AgentActivity agents={agents} a2a={a2a} />}
      {!isUser && <A2ATrace messages={a2aTranscript} />}
      {!isUser && <RetrievalCard steps={retrievalSteps} />}
      {!isUser && <ThoughtChain toolCalls={toolCalls} />}
      <div className={isUser ? "whitespace-pre-wrap leading-7" : "markdown"}>
        {isUser ? (
          content
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {content || "正在思考..."}
          </ReactMarkdown>
        )}
      </div>
    </article>
  );
}
