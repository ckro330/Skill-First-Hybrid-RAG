"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import {
  compressSession,
  createSession,
  deleteSession,
  getKnowledgeIndexStatus,
  getSessionHistory,
  getSessionTokens,
  listSessions,
  listSkills,
  loadFile,
  renameSession,
  rebuildKnowledgeIndex as rebuildKnowledgeIndexRequest,
  saveFile,
  streamChat,
  type A2AMessage,
  type AgentRun,
  type Evidence,
  type KnowledgeIndexStatus,
  type PlanInfo,
  type PlanTask,
  type RetrievalStep,
  type SessionSummary,
  type ToolCall,
  type TurnUsage
} from "@/lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls: ToolCall[];
  retrievalSteps: RetrievalStep[];
  plan: PlanInfo | null;
  agents: AgentRun[];
  a2a: A2AMessage[];
  a2aTranscript: A2AMessage[];
};

type TokenStats = {
  system_tokens: number;
  message_tokens: number;
  total_tokens: number;
};

type AppStore = {
  sessions: SessionSummary[];
  currentSessionId: string | null;
  messages: Message[];
  isStreaming: boolean;
  skills: Array<{ name: string; description: string; path: string }>;
  editableFiles: string[];
  inspectorPath: string;
  inspectorContent: string;
  inspectorDirty: boolean;
  sidebarWidth: number;
  inspectorWidth: number;
  tokenStats: TokenStats | null;
  turnUsage: TurnUsage | null;
  knowledgeIndexStatus: KnowledgeIndexStatus | null;
  createNewSession: () => Promise<void>;
  selectSession: (sessionId: string) => Promise<void>;
  sendMessage: (value: string) => Promise<void>;
  renameCurrentSession: (title: string) => Promise<void>;
  removeSession: (sessionId: string) => Promise<void>;
  exportSessionLog: () => Promise<void>;
  loadInspectorFile: (path: string) => Promise<void>;
  updateInspectorContent: (value: string) => void;
  saveInspector: () => Promise<void>;
  compressCurrentSession: () => Promise<void>;
  rebuildKnowledgeIndex: () => Promise<void>;
  setSidebarWidth: (width: number) => void;
  setInspectorWidth: (width: number) => void;
};

const FIXED_FILES = [
  "workspace/SOUL.md",
  "workspace/IDENTITY.md",
  "workspace/USER.md",
  "workspace/AGENTS.md",
  "SKILLS_SNAPSHOT.md"
];

const StoreContext = createContext<AppStore | null>(null);

function makeId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function normalizeEvidence(value: unknown): Evidence | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const item = value as Record<string, unknown>;
  const scoreValue = item.score;
  const score =
    typeof scoreValue === "number"
      ? scoreValue
      : typeof scoreValue === "string" && scoreValue.trim()
        ? Number(scoreValue)
        : null;

  return {
    source_path: String(item.source_path ?? ""),
    source_type: String(item.source_type ?? ""),
    locator: String(item.locator ?? ""),
    snippet: String(item.snippet ?? ""),
    channel: (item.channel as Evidence["channel"]) ?? "skill",
    score: Number.isFinite(score) ? score : null,
    parent_id: item.parent_id ? String(item.parent_id) : null
  };
}

function normalizeRetrievalStep(value: unknown): RetrievalStep | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const item = value as Record<string, unknown>;
  const rawResults = Array.isArray(item.results) ? item.results : [];
  const results = rawResults
    .map((entry) => normalizeEvidence(entry))
    .filter((entry): entry is Evidence => entry !== null);

  return {
    kind: item.kind === "memory" ? "memory" : "knowledge",
    stage: String(item.stage ?? "unknown"),
    title: String(item.title ?? "检索结果"),
    message: String(item.message ?? ""),
    results
  };
}

function normalizePlanTask(value: unknown): PlanTask | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const item = value as Record<string, unknown>;
  const dependsOn = Array.isArray(item.depends_on)
    ? item.depends_on.map((entry) => String(entry))
    : [];

  return {
    task_id: String(item.task_id ?? ""),
    title: String(item.title ?? ""),
    agent_id: String(item.agent_id ?? ""),
    instruction: String(item.instruction ?? ""),
    depends_on: dependsOn
  };
}

function normalizePlan(value: unknown): PlanInfo | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const item = value as Record<string, unknown>;
  const rawTasks = Array.isArray(item.tasks) ? item.tasks : [];

  return {
    domain: String(item.domain ?? ""),
    reason: String(item.reason ?? ""),
    stage: String(item.stage ?? ""),
    multi_agent: Boolean(item.multi_agent),
    tasks: rawTasks
      .map((task) => normalizePlanTask(task))
      .filter((task): task is PlanTask => task !== null)
  };
}

/** planner 事件先到、orchestrator 的 plan 事件随后，后者带排序后的任务，前者带判定理由。 */
function mergePlan(previous: PlanInfo | null, next: PlanInfo): PlanInfo {
  if (!previous) {
    return next;
  }

  return {
    domain: next.domain || previous.domain,
    reason: next.reason || previous.reason,
    stage: next.stage || previous.stage,
    multi_agent: next.multi_agent || previous.multi_agent,
    tasks: next.tasks.length ? next.tasks : previous.tasks
  };
}

function normalizeAgentRun(value: unknown): AgentRun | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const item = value as Record<string, unknown>;

  return {
    task_id: String(item.task_id ?? ""),
    agent_id: String(item.agent_id ?? ""),
    agent_name: String(item.agent_name ?? ""),
    title: String(item.title ?? ""),
    status: item.status === "running" ? "running" : "done"
  };
}

function normalizeA2AMessage(value: unknown): A2AMessage | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const item = value as Record<string, unknown>;
  const kind = item.kind === "task" || item.kind === "info" ? item.kind : "result";

  return {
    message_id: String(item.message_id ?? ""),
    task_id: String(item.task_id ?? ""),
    from: String(item.from ?? ""),
    to: String(item.to ?? ""),
    kind,
    content: String(item.content ?? "")
  };
}

function toUiMessages(history: Awaited<ReturnType<typeof getSessionHistory>>["messages"]) {
  return history.map((message) => ({
    id: makeId(),
    role: message.role,
    content: message.content ?? "",
    toolCalls: message.tool_calls ?? [],
    retrievalSteps: (message.retrieval_steps ?? [])
      .map((step) => normalizeRetrievalStep(step))
      .filter((step): step is RetrievalStep => step !== null),
    plan: normalizePlan(message.plan),
    agents: (message.agents ?? [])
      .map((run) => normalizeAgentRun(run))
      .filter((run): run is AgentRun => run !== null),
    a2a: (message.a2a ?? [])
      .map((entry) => normalizeA2AMessage(entry))
      .filter((entry): entry is A2AMessage => entry !== null),
    a2aTranscript: (message.a2a_transcript ?? [])
      .map((entry) => normalizeA2AMessage(entry))
      .filter((entry): entry is A2AMessage => entry !== null)
  }));
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [skills, setSkills] = useState<Array<{ name: string; description: string; path: string }>>([]);
  const [inspectorPath, setInspectorPath] = useState("workspace/USER.md");
  const [inspectorContent, setInspectorContent] = useState("");
  const [inspectorDirty, setInspectorDirty] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(308);
  const [inspectorWidth, setInspectorWidth] = useState(360);
  const [tokenStats, setTokenStats] = useState<TokenStats | null>(null);
  const [turnUsage, setTurnUsage] = useState<TurnUsage | null>(null);
  const [knowledgeIndexStatus, setKnowledgeIndexStatus] = useState<KnowledgeIndexStatus | null>(
    null
  );

  const editableFiles = useMemo(
    () => [...FIXED_FILES, ...skills.map((skill) => skill.path)],
    [skills]
  );

  async function refreshSessions() {
    setSessions(await listSessions());
  }

  async function refreshSkills() {
    setSkills(await listSkills());
  }

  async function refreshKnowledgeIndexStatus() {
    setKnowledgeIndexStatus(await getKnowledgeIndexStatus());
  }

  async function refreshSessionDetails(sessionId: string) {
    const [history, tokens] = await Promise.all([
      getSessionHistory(sessionId),
      getSessionTokens(sessionId)
    ]);
    setMessages(toUiMessages(history.messages));
    setTokenStats(tokens);
  }

  async function createNewSession() {
    const created = await createSession();
    await refreshSessions();
    setCurrentSessionId(created.id);
    setMessages([]);
    setTokenStats(null);
  }

  async function selectSession(sessionId: string) {
    setCurrentSessionId(sessionId);
    await refreshSessionDetails(sessionId);
  }

  async function ensureSession() {
    if (currentSessionId) {
      return currentSessionId;
    }

    const created = await createSession();
    setCurrentSessionId(created.id);
    await refreshSessions();
    return created.id;
  }

  async function sendMessage(value: string) {
    if (!value.trim() || isStreaming) {
      return;
    }

    const sessionId = await ensureSession();
    const userMessage: Message = {
      id: makeId(),
      role: "user",
      content: value.trim(),
      toolCalls: [],
      retrievalSteps: [],
      plan: null,
      agents: [],
      a2a: [],
      a2aTranscript: []
    };
    const assistantMessage: Message = {
      id: makeId(),
      role: "assistant",
      content: "",
      toolCalls: [],
      retrievalSteps: [],
      plan: null,
      agents: [],
      a2a: [],
      a2aTranscript: []
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    let activeAssistantId = assistantMessage.id;

    const patchAssistant = (updater: (message: Message) => Message) => {
      setMessages((prev) =>
        prev.map((message) => (message.id === activeAssistantId ? updater(message) : message))
      );
    };

    try {
      await streamChat(
        { message: value.trim(), session_id: sessionId },
        {
          onEvent(event, data) {
            if (event === "retrieval") {
              const step = normalizeRetrievalStep(data);
              if (!step) {
                return;
              }
              patchAssistant((message) => ({
                ...message,
                retrievalSteps: [...message.retrievalSteps, step]
              }));
              return;
            }

            if (event === "token") {
              patchAssistant((message) => ({
                ...message,
                content: `${message.content}${String(data.content ?? "")}`
              }));
              return;
            }

            if (event === "tool_start") {
              patchAssistant((message) => ({
                ...message,
                toolCalls: [
                  ...message.toolCalls,
                  {
                    tool: String(data.tool ?? "tool"),
                    input: String(data.input ?? ""),
                    output: ""
                  }
                ]
              }));
              return;
            }

            if (event === "tool_end") {
              patchAssistant((message) => ({
                ...message,
                toolCalls: message.toolCalls.map((toolCall, index, list) =>
                  index === list.length - 1
                    ? { ...toolCall, output: String(data.output ?? "") }
                    : toolCall
                )
              }));
              return;
            }

            if (event === "planner" || event === "plan") {
              const plan = normalizePlan(data);
              if (!plan) {
                return;
              }
              patchAssistant((message) => ({
                ...message,
                plan: mergePlan(message.plan, plan)
              }));
              return;
            }

            if (event === "agent_start") {
              const run = normalizeAgentRun({
                task_id: data.task_id,
                agent_id: data.agent_id,
                agent_name: data.agent_name,
                title: data.title,
                status: "running"
              });
              if (!run) {
                return;
              }
              patchAssistant((message) => ({
                ...message,
                agents: [...message.agents, run]
              }));
              return;
            }

            if (event === "a2a") {
              const a2aMessage = normalizeA2AMessage(data.message);
              if (!a2aMessage) {
                return;
              }
              patchAssistant((message) => ({
                ...message,
                a2a: [...message.a2a, a2aMessage]
              }));
              return;
            }

            if (event === "agent_result") {
              patchAssistant((message) => ({
                ...message,
                content: message.content || String(data.content ?? ""),
                agents: message.agents.map((run) =>
                  run.task_id === String(data.task_id ?? "") ? { ...run, status: "done" } : run
                )
              }));
              return;
            }

            if (event === "a2a_transcript") {
              const transcript = (Array.isArray(data.messages) ? data.messages : [])
                .map((entry) => normalizeA2AMessage(entry))
                .filter((entry): entry is A2AMessage => entry !== null);
              patchAssistant((message) => ({ ...message, a2aTranscript: transcript }));
              return;
            }

            if (event === "new_response") {
              const nextAssistant: Message = {
                id: makeId(),
                role: "assistant",
                content: "",
                toolCalls: [],
                retrievalSteps: [],
                plan: null,
                agents: [],
                a2a: [],
                a2aTranscript: []
              };
              activeAssistantId = nextAssistant.id;
              setMessages((prev) => [...prev, nextAssistant]);
              return;
            }

            if (event === "done") {
              const finalContent = String(data.content ?? "");
              patchAssistant((message) =>
                message.content
                  ? message
                  : {
                      ...message,
                      content: finalContent
                    }
              );
              return;
            }

            if (event === "usage") {
              const promptTokens = Number(data.prompt_tokens ?? 0);
              if (!promptTokens) {
                return;
              }
              setTurnUsage({
                llm_calls: Number(data.llm_calls ?? 0),
                prompt_tokens: promptTokens,
                output_tokens: Number(data.output_tokens ?? 0),
                cache_hit_tokens: Number(data.cache_hit_tokens ?? 0),
                cache_miss_tokens: Number(data.cache_miss_tokens ?? 0),
                cache_hit_rate: Number(data.cache_hit_rate ?? 0)
              });
              return;
            }

            if (event === "title") {
              void refreshSessions();
              return;
            }

            if (event === "error") {
              patchAssistant((message) => ({
                ...message,
                content:
                  message.content || `请求失败: ${String(data.error ?? "unknown error")}`
              }));
            }
          }
        }
      );
    } finally {
      setIsStreaming(false);
      await refreshSessions();
      await refreshSessionDetails(sessionId);
    }
  }

  async function renameCurrentSession(title: string) {
    if (!currentSessionId || !title.trim()) {
      return;
    }
    await renameSession(currentSessionId, title.trim());
    await refreshSessions();
  }

  async function removeSession(sessionId: string) {
    await deleteSession(sessionId);
    await refreshSessions();
    if (currentSessionId === sessionId) {
      const nextSessions = await listSessions();
      setSessions(nextSessions);
      if (nextSessions.length) {
        setCurrentSessionId(nextSessions[0].id);
        await refreshSessionDetails(nextSessions[0].id);
      } else {
        setCurrentSessionId(null);
        setMessages([]);
        setTokenStats(null);
      }
    }
  }

  async function exportSessionLog() {
    if (!currentSessionId) {
      return;
    }
    const history = await getSessionHistory(currentSessionId);
    const payload = {
      exported_at: new Date().toISOString(),
      session_id: currentSessionId,
      session: history,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `session-${currentSessionId}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  async function loadInspectorFile(path: string) {
    setInspectorPath(path);
    const file = await loadFile(path);
    setInspectorContent(file.content);
    setInspectorDirty(false);
  }

  function updateInspectorContent(value: string) {
    setInspectorContent(value);
    setInspectorDirty(true);
  }

  async function saveInspector() {
    await saveFile(inspectorPath, inspectorContent);
    setInspectorDirty(false);
    await refreshSkills();
  }

  async function compressCurrentSession() {
    if (!currentSessionId) {
      return;
    }
    await compressSession(currentSessionId);
    await refreshSessionDetails(currentSessionId);
    await refreshSessions();
  }

  async function rebuildKnowledgeIndex() {
    await rebuildKnowledgeIndexRequest();
    await refreshKnowledgeIndexStatus();
  }

  useEffect(() => {
    void (async () => {
      const [initialSessions, initialSkills, initialKnowledgeIndexStatus] = await Promise.all([
        listSessions(),
        listSkills(),
        getKnowledgeIndexStatus()
      ]);

      setSessions(initialSessions);
      setSkills(initialSkills);
      setKnowledgeIndexStatus(initialKnowledgeIndexStatus);

      if (initialSessions.length) {
        setCurrentSessionId(initialSessions[0].id);
        await refreshSessionDetails(initialSessions[0].id);
      } else {
        const created = await createSession();
        setCurrentSessionId(created.id);
        setSessions([created]);
      }

      const file = await loadFile("workspace/USER.md");
      setInspectorPath(file.path);
      setInspectorContent(file.content);
    })();
  }, []);

  useEffect(() => {
    if (!knowledgeIndexStatus?.building) {
      return;
    }

    const timer = window.setInterval(() => {
      void getKnowledgeIndexStatus().then((status) => setKnowledgeIndexStatus(status));
    }, 3000);

    return () => window.clearInterval(timer);
  }, [knowledgeIndexStatus?.building]);

  const value: AppStore = {
    sessions,
    currentSessionId,
    messages,
    isStreaming,
    skills,
    editableFiles,
    inspectorPath,
    inspectorContent,
    inspectorDirty,
    sidebarWidth,
    inspectorWidth,
    tokenStats,
    turnUsage,
    knowledgeIndexStatus,
    createNewSession,
    selectSession,
    sendMessage,
    renameCurrentSession,
    removeSession,
    exportSessionLog,
    loadInspectorFile,
    updateInspectorContent,
    saveInspector,
    compressCurrentSession,
    rebuildKnowledgeIndex,
    setSidebarWidth,
    setInspectorWidth
  };

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useAppStore() {
  const value = useContext(StoreContext);
  if (!value) {
    throw new Error("useAppStore must be used inside AppProvider");
  }
  return value;
}
