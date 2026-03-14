"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { Shield, Activity, BarChart3, RefreshCw, Send, Gauge } from "lucide-react";
import { benchmark, chat, history, resetSession, stats } from "@/lib/api";
import { BenchmarkResponse, ChatResponse } from "@/lib/types";
import { MetricCard } from "@/components/metric-card";
import { VerdictPill } from "@/components/verdict-pill";
import { LayerBar } from "@/components/layer-bar";

function createSessionId() {
  return `sess-${Math.random().toString(36).slice(2, 10)}`;
}

export default function Page() {
  const [sessionId, setSessionId] = useState("sess-pending");
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState<ChatResponse | null>(null);
  const [err, setErr] = useState("");
  const [sessionStats, setSessionStats] = useState<Record<string, unknown>>({});
  const [sessionHistory, setSessionHistory] = useState<Array<{ turn_number: number; prompt: string; decision: string; risk_level: string; composite_score: number; top_threat: string }>>([]);
  const [bench, setBench] = useState<BenchmarkResponse | null>(null);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState("");
  const [benchmarkDurationMs, setBenchmarkDurationMs] = useState<number | null>(null);
  const [securityEnabled, setSecurityEnabled] = useState(true);

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

  async function refreshSessionState() {
    if (sessionId === "sess-pending") return;
    const [s, h] = await Promise.all([stats(sessionId), history(sessionId)]);
    setSessionStats(s.stats);
    setSessionHistory(h.history);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!prompt.trim() || sessionId === "sess-pending") return;
    setBusy(true);
    setErr("");
    try {
      const res = await chat(prompt, sessionId, securityEnabled);
      setLast(res);
      setPrompt("");
      await refreshSessionState();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
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
      setSessionHistory([]);
      setSessionStats({});
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Reset failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto min-h-screen max-w-7xl px-6 py-8">
      <header className="mb-8 flex items-center justify-between">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-border bg-panel px-3 py-1 text-xs text-slate-400">
            <Shield size={14} />
            Enterprise Console
          </div>
          <h1 className="mt-3 text-3xl font-semibold tracking-tight">ThinkSym Neurosymbolic Security Platform</h1>
          <p className="mt-2 text-sm text-slate-400">Session {sessionId}</p>
        </div>
        <div className="flex items-center gap-3">
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
        <MetricCard label="Top Threat" value={hasSecurityMetrics ? (last?.analysis.top_threat ?? "none") : "N/A"} />
        <MetricCard label="Latency" value={last ? `${Math.round(last.llm_latency_ms || last.analysis.total_time_ms)} ms` : "0 ms"} />
      </section>

      <section className="mt-6 grid gap-6 lg:grid-cols-2">
        <div className="panel p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-lg font-semibold"><Send size={18} /> Secure Chat</h2>
            {last && hasSecurityMetrics && <VerdictPill decision={last.analysis.verdict.decision} />}
          </div>
          <form onSubmit={onSubmit} className="space-y-3">
            <label className="flex items-center justify-between rounded-lg border border-border bg-slate-950 px-4 py-3 text-sm text-slate-200">
              <span className="font-medium">Enable ThinkSym Security</span>
              <input
                type="checkbox"
                checked={securityEnabled}
                onChange={(e) => setSecurityEnabled(e.target.checked)}
                className="h-4 w-4 accent-[#7c9cff]"
              />
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter a safe question or attack prompt"
              className="h-36 w-full rounded-lg border border-border bg-slate-950 px-3 py-2 text-sm outline-none focus:border-brand"
            />
            <button
              type="submit"
              disabled={busy || !prompt.trim()}
              className="inline-flex items-center gap-2 rounded-lg bg-brand px-4 py-2 text-sm font-medium text-slate-950 disabled:opacity-50"
            >
              <Send size={16} />
              {busy ? "Processing..." : securityEnabled ? "Run with Security" : "Run without Security"}
            </button>
          </form>
          {err && <p className="mt-3 text-sm text-danger">{err}</p>}

          {last && (
            <div className="mt-5 space-y-3 rounded-lg border border-border bg-slate-950 p-4">
              {hasSecurityMetrics ? (
                <>
                  <p className="text-xs uppercase tracking-wider text-slate-400">Verdict Reason</p>
                  <p className="text-sm text-slate-200">{last.analysis.verdict.primary_reason}</p>
                </>
              ) : (
                <>
                  <p className="text-xs uppercase tracking-wider text-slate-400">Security Status</p>
                  <p className="text-sm text-slate-200">ThinkSym Security is disabled for this run, so policy scoring and layer analytics are hidden.</p>
                </>
              )}
              {last.blocked ? (
                <p className="text-sm text-danger">Blocked before LLM execution.</p>
              ) : (
                <>
                  <p className="text-xs uppercase tracking-wider text-slate-400">Mode</p>
                  <p className="text-sm text-slate-200">{last.protection_enabled ? "ThinkSym Security Enabled" : "ThinkSym Security Disabled"}</p>
                  <p className="text-xs uppercase tracking-wider text-slate-400">Model</p>
                  <p className="text-sm text-slate-200">{last.llm_model || "unknown"}</p>
                  <p className="text-xs uppercase tracking-wider text-slate-400">Backend</p>
                  <p className="text-sm text-slate-200">{last.llm_backend || "unknown"} · {Math.round(last.llm_latency_ms || 0)} ms</p>
                  <p className="text-xs uppercase tracking-wider text-slate-400">LLM Response</p>
                  <p className="whitespace-pre-wrap text-sm text-slate-200">{last.llm_text || last.llm_error || "No response"}</p>
                </>
              )}
            </div>
          )}
        </div>

        <div className="panel p-5">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold"><Gauge size={18} /> Layer Observability</h2>
          {hasSecurityMetrics ? (
            <div className="space-y-4">
              {(last?.analysis.layers ?? []).map((layer) => (
                <LayerBar key={layer.name} name={layer.name} score={layer.score} label={layer.label} />
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-400">Layer observability is available only when ThinkSym Security is enabled.</p>
          )}
        </div>
      </section>

      <section className="mt-6 grid gap-6 lg:grid-cols-2">
        <div className="panel p-5">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold"><Activity size={18} /> Session Timeline</h2>
          <div className="max-h-80 space-y-2 overflow-auto pr-1">
            {sessionHistory.length === 0 ? (
              <p className="text-sm text-slate-400">No session activity yet.</p>
            ) : (
              sessionHistory
                .slice()
                .reverse()
                .map((item) => (
                  <div key={item.turn_number} className="rounded-lg border border-border bg-slate-950 p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <p className="text-xs text-slate-400">Turn {item.turn_number}</p>
                      <VerdictPill decision={item.decision as "ALLOW" | "WARN" | "BLOCK"} />
                    </div>
                    <p className="line-clamp-2 text-sm text-slate-200">{item.prompt}</p>
                    <p className="mt-2 text-xs text-slate-400">{item.risk_level} · {item.composite_score.toFixed(3)} · {item.top_threat}</p>
                  </div>
                ))
            )}
          </div>
        </div>

        <div className="panel p-5">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold"><BarChart3 size={18} /> Benchmark Snapshot</h2>
          {benchmarkRunning ? (
            <p className="text-sm text-slate-300">Benchmark is running. This can take 30-120 seconds depending on model warmup and backend load.</p>
          ) : benchmarkError ? (
            <p className="text-sm text-danger">Benchmark failed: {benchmarkError}</p>
          ) : !bench ? (
            <p className="text-sm text-slate-400">Run benchmark to generate adversarial/benign quality metrics.</p>
          ) : (
            <div className="space-y-3 text-sm">
              <p className="text-slate-200">Adversarial block rate: {(bench.adversarial_block_rate * 100).toFixed(1)}%</p>
              <p className="text-slate-200">Benign false positive rate: {(bench.benign_false_positive_rate * 100).toFixed(1)}%</p>
              <p className="text-slate-400">Adversarial: B {bench.blocked_adversarial} · W {bench.warned_adversarial} · A {bench.allowed_adversarial}</p>
              <p className="text-slate-400">Benign: B {bench.blocked_benign} · W {bench.warned_benign} · A {bench.allowed_benign}</p>
              {benchmarkDurationMs !== null && <p className="text-slate-400">Last run time: {(benchmarkDurationMs / 1000).toFixed(1)}s</p>}
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
