"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { BarChart3, RefreshCw, Send, Gauge } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { benchmark, chat, resetSession } from "@/lib/api";
import { BenchmarkResponse, ChatResponse } from "@/lib/types";
import { MetricCard } from "@/components/metric-card";
import { VerdictPill } from "@/components/verdict-pill";
import { LayerBar } from "@/components/layer-bar";
import { BenchmarkPanel } from "@/components/benchmark-panel";

function createSessionId() {
  return `sess-${Math.random().toString(36).slice(2, 10)}`;
}

type ChatTurn = {
  turn: number;
  prompt: string;
  response: string;
  decision: "ALLOW" | "WARN" | "BLOCK";
  riskLevel: string;
  composite: number;
  topThreat: string;
  modelUsed: string;
  reason: string;
};

const LAYER_LABELS: Record<string, string> = {
  smoothllm: "SmoothLLM Layer",
  neural: "Neural Layer",
  scallop: "Scallop Layer",
  z3: "Z3 Layer",
  knowledgegraph: "Knowledge Graph Layer",
};

const THREAT_LABELS: Record<string, string> = {
  none: "None",
  prompt_injection: "Prompt Injection",
  jailbreak: "Jailbreak",
  pii_extraction: "PII Extraction",
  system_prompt_leak: "System Prompt Leak",
  harmful_content: "Harmful Content",
  code_exploit: "Code Exploit",
  social_engineering: "Social Engineering",
  data_exfiltration: "Data Exfiltration",
  resource_abuse: "Resource Abuse",
  misinformation: "Misinformation",
};

function toTitleCase(text: string): string {
  return text
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

function formatLayerName(rawName: string): string {
  const key = rawName.toLowerCase().replace(/[\s_-]/g, "");
  return LAYER_LABELS[key] ?? `${toTitleCase(rawName.replace(/[_-]+/g, " "))} Layer`;
}

function formatThreatName(rawThreat: string): string {
  const key = rawThreat.toLowerCase().trim();
  if (THREAT_LABELS[key]) {
    return THREAT_LABELS[key];
  }
  return toTitleCase(rawThreat.replace(/[_-]+/g, " "));
}

function formatLayerLabel(rawLabel: string): string {
  if (!rawLabel) return rawLabel;
  const normalized = rawLabel.toLowerCase().trim();
  if (THREAT_LABELS[normalized] || normalized.includes("_") || normalized.includes("-")) {
    return formatThreatName(rawLabel);
  }
  return rawLabel;
}

export default function Page() {
  const [sessionId, setSessionId] = useState("sess-pending");
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState<ChatResponse | null>(null);
  const [chatTurns, setChatTurns] = useState<ChatTurn[]>([]);
  const [err, setErr] = useState("");
  const [bench, setBench] = useState<BenchmarkResponse | null>(null);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState("");
  const [benchmarkDurationMs, setBenchmarkDurationMs] = useState<number | null>(null);
  const [securityEnabled, setSecurityEnabled] = useState(true);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setSessionId(createSessionId());
  }, []);

  const hasSecurityMetrics = !!last?.protection_enabled;
  const decision = hasSecurityMetrics ? (last?.analysis.verdict.decision ?? "ALLOW") : "ALLOW";

  const riskTone = useMemo(() => {
    if (decision === "BLOCK") return "danger" as const;
    if (decision === "WARN") return "warn" as const;
    return "safe" as const;
  }, [decision]);

  useEffect(() => {
    if (!chatScrollRef.current) return;
    chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
  }, [chatTurns]);

  const securityEvents = useMemo(
    () => chatTurns.filter((item) => item.decision !== "ALLOW").slice().reverse(),
    [chatTurns],
  );

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!prompt.trim() || sessionId === "sess-pending") return;
    setBusy(true);
    setErr("");
    try {
      const res = await chat(prompt, sessionId, securityEnabled);
      setLast(res);
      const refusalText = "Sorry, I can't help you with that.";
      const assistantResponse = res.blocked
        ? (res.llm_text || refusalText)
        : (res.llm_text || res.llm_error || "No response");
      setChatTurns((prev) => [
        ...prev,
        {
          turn: res.analysis.turn_number,
          prompt: res.analysis.prompt,
          response: assistantResponse,
          decision: res.analysis.verdict.decision as "ALLOW" | "WARN" | "BLOCK",
          riskLevel: res.analysis.verdict.decision === "ALLOW" ? "SAFE" : res.analysis.verdict.risk_level,
          composite: res.analysis.verdict.composite_score,
          topThreat: res.analysis.verdict.decision === "ALLOW" ? "none" : res.analysis.top_threat,
          modelUsed: res.llm_model || res.llm_backend || "unknown",
          reason: res.analysis.verdict.primary_reason,
        },
      ]);
      setPrompt("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  function onPromptKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    if (busy || !prompt.trim()) return;
    void onSubmit(event as unknown as FormEvent);
  }

  async function onRunBenchmark() {
    const startedAt = Date.now();
    setBenchmarkRunning(true);
    setBenchmarkError("");
    setBusy(true);
    setErr("");
    try {
      const res = await benchmark();
      setBench(res);
      setBenchmarkDurationMs(Date.now() - startedAt);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Benchmark failed";
      setBenchmarkError(msg);
      setErr(msg);
      setBenchmarkDurationMs(null);
    } finally {
      setBenchmarkRunning(false);
      setBusy(false);
    }
  }

  async function onReset() {
    if (sessionId === "sess-pending") return;
    setBusy(true);
    setErr("");
    try {
      await resetSession(sessionId);
      setLast(null);
      setChatTurns([]);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Reset failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto min-h-screen max-w-7xl px-6 py-8">
      <header className="mb-8 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="rounded-xl border border-border bg-panel overflow-hidden" style={{ width: 72, height: 72 }}>
            <img
              src="/thinksym_logo.png"
              alt="ThinkSym logo"
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
            />
          </div>
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">ThinkSym</h1>
            <p className="mt-1 text-sm text-slate-400">NeuroSymbolic LLM Security</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="inline-flex items-center gap-3 rounded-lg border border-border bg-panel px-4 py-2 text-sm text-slate-200">
            <span className="font-medium">ThinkSym Security</span>
            <button
              type="button"
              role="switch"
              aria-checked={securityEnabled}
              aria-label="Toggle ThinkSym Security"
              onClick={() => setSecurityEnabled((prev) => !prev)}
              className={`relative inline-flex h-6 w-12 items-center rounded-full transition-colors ${securityEnabled ? "bg-emerald-500" : "bg-red-500"}`}
            >
              <span
                className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${securityEnabled ? "translate-x-6" : "translate-x-1"}`}
              />
            </button>
            <span className={`text-xs font-semibold ${securityEnabled ? "text-emerald-300" : "text-red-300"}`}>
              {securityEnabled ? "ON" : "OFF"}
            </span>
          </div>
          <button onClick={onRunBenchmark} disabled={busy || benchmarkRunning} className="inline-flex items-center gap-2 rounded-lg border border-border bg-panel px-4 py-2 text-sm hover:bg-slate-800 disabled:opacity-50">
            <BarChart3 size={16} />
            {benchmarkRunning ? "Running Benchmark..." : "Benchmark"}
          </button>
          <button onClick={onReset} disabled={busy} className="inline-flex items-center gap-2 rounded-lg border border-border bg-panel px-4 py-2 text-sm hover:bg-slate-800 disabled:opacity-50">
            <RefreshCw size={16} />
            Reset Session
          </button>
        </div>
      </header>

      <section className="grid gap-4 md:grid-cols-4">
        <MetricCard label="Decision" value={hasSecurityMetrics ? (last?.analysis.verdict.decision ?? "N/A") : "N/A"} tone={hasSecurityMetrics ? riskTone : "safe"} />
        <MetricCard label="Composite" value={hasSecurityMetrics && last ? last.analysis.verdict.composite_score.toFixed(3) : "N/A"} tone={hasSecurityMetrics ? riskTone : "safe"} />
        <MetricCard label="Top Threat" value={hasSecurityMetrics ? formatThreatName(last?.analysis.top_threat ?? "none") : "N/A"} />
        <MetricCard label="Latency" value={last ? `${Math.round(last.llm_latency_ms || last.analysis.total_time_ms)} ms` : "0 ms"} />
      </section>

      <section className="mt-6 grid gap-6 lg:grid-cols-2">
        <div className="panel p-5">
          <div className="mb-4">
            <h2 className="flex items-center gap-2 text-lg font-semibold"><Send size={18} /> Secure Chat</h2>
          </div>
          <div ref={chatScrollRef} className="mb-4 max-h-[26rem] space-y-3 overflow-auto rounded-lg border border-border bg-slate-950 p-4">
            {chatTurns.length > 0 && (
              chatTurns.map((item) => (
                <div key={item.turn} className="space-y-2">
                  <div className="flex justify-end">
                    <div className="max-w-[85%] rounded-xl border border-border bg-panel px-4 py-3">
                      <p className="mb-1 text-[11px] uppercase tracking-wide text-slate-400">Turn {item.turn}</p>
                      <p className="whitespace-pre-wrap text-sm text-slate-100">{item.prompt}</p>
                    </div>
                  </div>
                  <div className="flex justify-start">
                    <div className="max-w-[85%] rounded-xl border border-border bg-slate-900 px-4 py-3">
                      <p className="mb-1 text-[11px] uppercase tracking-wide text-slate-400">Assistant</p>
                      <div className="text-sm text-slate-200">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            p: ({ children }) => <p className="mb-3 last:mb-0 leading-6">{children}</p>,
                            ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
                            ul: ({ children }) => <ul className="mb-3 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
                            li: ({ children }) => <li className="leading-6">{children}</li>,
                            strong: ({ children }) => <strong className="font-semibold text-slate-50">{children}</strong>,
                            em: ({ children }) => <em className="italic">{children}</em>,
                            code: ({ children }) => <code className="rounded bg-slate-800 px-1.5 py-0.5 text-[0.9em] text-slate-100">{children}</code>,
                            pre: ({ children }) => <pre className="mb-3 overflow-x-auto rounded-lg bg-slate-950 p-3 text-xs last:mb-0">{children}</pre>,
                            a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer" className="text-[#9ab6ff] underline underline-offset-2">{children}</a>,
                          }}
                        >
                          {item.response}
                        </ReactMarkdown>
                      </div>
                      <p className="mt-2 text-xs text-slate-400">{item.riskLevel} · {item.composite.toFixed(3)} · {formatThreatName(item.topThreat)} · {item.modelUsed}</p>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
          <form onSubmit={onSubmit} className="space-y-3">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={onPromptKeyDown}
              placeholder="Type your prompt here"
              className="h-36 w-full rounded-lg border border-border bg-slate-950 px-3 py-2 text-sm outline-none focus:border-brand"
            />
            <button
              type="submit"
              disabled={busy || !prompt.trim()}
              className="inline-flex items-center gap-2 rounded-lg bg-brand px-4 py-2 text-sm font-medium text-slate-950 disabled:opacity-50"
            >
              <Send size={16} />
              {busy ? "Processing..." : securityEnabled ? "Secure Message" : "Send Message"}
            </button>
          </form>
          {err && <p className="mt-3 text-sm text-danger">{err}</p>}
        </div>

        <div className="panel p-5">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold"><Gauge size={18} /> Layer Observability</h2>
          {hasSecurityMetrics ? (
            <div className="space-y-4">
              {(last?.analysis.layers ?? []).map((layer) => (
                <LayerBar
                  key={layer.name}
                  name={formatLayerName(layer.name)}
                  score={layer.score}
                  label={formatLayerLabel(layer.label)}
                />
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-400">Layer observability is available only when ThinkSym Security is enabled.</p>
          )}

          <div className="mt-6 border-t border-border pt-5">
            <h3 className="mb-4 text-base font-semibold text-slate-100">Security Timeline</h3>
            {securityEvents.length === 0 ? (
              <p className="text-sm text-slate-400">No flagged security events yet.</p>
            ) : (
              <div className="max-h-72 space-y-3 overflow-auto pr-1">
                {securityEvents.map((item) => (
                  <div key={`security-${item.turn}`} className="rounded-lg border border-border bg-slate-950 p-3">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <p className="text-xs uppercase tracking-wide text-slate-400">Turn {item.turn}</p>
                      <VerdictPill decision={item.decision} />
                    </div>
                    <p className="line-clamp-3 text-sm text-slate-200">{item.prompt}</p>
                    <p className="mt-2 text-xs text-slate-400">{item.riskLevel} · {item.composite.toFixed(3)} · {formatThreatName(item.topThreat)}</p>
                    <p className="mt-2 text-xs leading-5 text-slate-300">{item.reason}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="mt-6">
        <div className="panel p-5">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold"><BarChart3 size={18} /> Benchmark Snapshot</h2>
          <BenchmarkPanel
            bench={bench}
            running={benchmarkRunning}
            error={benchmarkError}
            durationMs={benchmarkDurationMs}
          />
        </div>
      </section>
    </main>
  );
}
