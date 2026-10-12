export type ToolCall = {
  tool: string;
  input: string;
  output: string;
};

export type Evidence = {
  source_path: string;
  source_type: string;
  locator: string;
  snippet: string;
  channel: "memory" | "skill" | "vector" | "bm25" | "fused";
  score: number | null;
  parent_id: string | null;
};

export type RetrievalStep = {
  kind: "memory" | "knowledge";
  stage: string;
  title: string;
  message: string;
  results: Evidence[];
};

export type PlanTask = {
  task_id: string;
  title: string;
  agent_id: string;
  instruction: string;
  depends_on: string[];
};

export type PlanInfo = {
  domain: string;
  reason: string;
  stage: string;
  multi_agent: boolean;
  tasks: PlanTask[];
};

export type AgentRun = {
  task_id: string;
  agent_id: string;
  agent_name: string;
  title: string;
  status: "running" | "done";
};

export type A2AMessage = {
  message_id: string;
  task_id: string;
  from: string;
  to: string;
  kind: "task" | "result" | "info";
  content: string;
};

export const AGENT_LABELS: Record<string, string> = {
  planner: "Planner",
  "log-analysis-agent": "日志分析 Agent",
  "code-modify-agent": "代码修改 Agent",
  "test-agent": "测试 Agent",
  "git-agent": "Git 管理 Agent"
};

export const DOMAIN_LABELS: Record<string, string> = {
  coding: "代码任务",
  rag: "知识库",
  general: "通用问答"
};

export const A2A_KIND_LABELS: Record<string, string> = {
  task: "派发",
  result: "回传",
  info: "通知"
};

export function agentLabel(agentId: string) {
  return AGENT_LABELS[agentId] ?? (agentId || "未知 Agent");
}

export type KnowledgeIndexStatus = {
  ready: boolean;
  building: boolean;
  last_built_at: number | null;
  indexed_files: number;
  vector_ready: boolean;
  bm25_ready: boolean;
};

/** 一轮对话的 LLM 用量（含上下文缓存命中），由后端 usage 事件推送。 */
export type TurnUsage = {
  llm_calls: number;
  prompt_tokens: number;
  output_tokens: number;
  cache_hit_tokens: number;
  cache_miss_tokens: number;
  cache_hit_rate: number;
};

export type SessionSummary = {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
};

export type SessionHistory = {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  compressed_context?: string;
  messages: Array<{
    role: "user" | "assistant";
    content: string;
    tool_calls?: ToolCall[];
    retrieval_steps?: RetrievalStep[];
    plan?: PlanInfo;
    agents?: AgentRun[];
    a2a?: A2AMessage[];
    a2a_transcript?: A2AMessage[];
  }>;
};

export type StreamHandlers = {
  onEvent: (event: string, data: Record<string, unknown>) => void;
};

const DEFAULT_API_PORT = "8004";

function normalizeApiBase(base: string) {
  return base.endsWith("/") ? base.slice(0, -1) : base;
}

function getApiBase() {
  const configuredBase = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (configuredBase) {
    return normalizeApiBase(configuredBase);
  }

  if (typeof window === "undefined") {
    return `http://127.0.0.1:${DEFAULT_API_PORT}/api`;
  }

  return `${window.location.protocol}//${window.location.hostname}:${DEFAULT_API_PORT}/api`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBase()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function listSessions() {
  return request<SessionSummary[]>("/sessions");
}

export async function createSession(title = "新会话") {
  return request<SessionSummary>("/sessions", {
    method: "POST",
    body: JSON.stringify({ title })
  });
}

export async function renameSession(sessionId: string, title: string) {
  return request<SessionSummary>(`/sessions/${sessionId}`, {
    method: "PUT",
    body: JSON.stringify({ title })
  });
}

export async function deleteSession(sessionId: string) {
  return request<{ ok: boolean }>(`/sessions/${sessionId}`, {
    method: "DELETE"
  });
}

export async function getSessionHistory(sessionId: string) {
  return request<SessionHistory>(`/sessions/${sessionId}/history`);
}

export async function getSessionTokens(sessionId: string) {
  return request<{
    system_tokens: number;
    message_tokens: number;
    total_tokens: number;
  }>(`/tokens/session/${sessionId}`);
}

export async function listSkills() {
  return request<Array<{ name: string; description: string; path: string }>>("/skills");
}

export async function loadFile(path: string) {
  return request<{ path: string; content: string }>(
    `/files?path=${encodeURIComponent(path)}`
  );
}

export async function saveFile(path: string, content: string) {
  return request<{ ok: boolean; path: string }>("/files", {
    method: "POST",
    body: JSON.stringify({ path, content })
  });
}

export async function compressSession(sessionId: string) {
  return request<{ archived_count: number; remaining_count: number }>(
    `/sessions/${sessionId}/compress`,
    { method: "POST" }
  );
}

export async function getKnowledgeIndexStatus() {
  return request<KnowledgeIndexStatus>("/knowledge/index/status");
}

export async function rebuildKnowledgeIndex() {
  return request<{ accepted: boolean }>("/knowledge/index/rebuild", {
    method: "POST"
  });
}

export async function streamChat(
  payload: {
    message: string;
    session_id: string;
  },
  handlers: StreamHandlers
) {
  const response = await fetch(`${getApiBase()}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      ...payload,
      stream: true
    })
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const flushBlock = (block: string) => {
    const lines = block.split("\n");
    let event = "message";
    const dataLines: string[] = [];

    for (const line of lines) {
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      }
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }

    if (!dataLines.length) {
      return;
    }

    const data = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
    handlers.onEvent(event, data);
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      flushBlock(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      if (buffer.trim()) {
        flushBlock(buffer);
      }
      break;
    }
  }
}
