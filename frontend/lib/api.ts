import { AnalyzeResponse, BenchmarkResponse, ChatResponse, DemoCompareResponse } from "@/lib/types";

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8002";

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }

  return response.json() as Promise<T>;
}

export function health() {
  return api<{ status: string; service: string }>("/health");
}

export function analyze(prompt: string, sessionId: string) {
  return api<AnalyzeResponse>("/api/v1/analyze", {
    method: "POST",
    body: JSON.stringify({ prompt, session_id: sessionId, update_session: true, device: "cuda" }),
  });
}

export function chat(prompt: string, sessionId: string, securityEnabled = true) {
  return api<ChatResponse>("/api/v1/chat", {
    method: "POST",
    body: JSON.stringify({
      prompt,
      session_id: sessionId,
      update_session: true,
      security_enabled: securityEnabled,
      device: "cuda",
      max_new_tokens: 160,
      ollama_model: "llama3.2:3b",
    }),
  });
}

export function demoCompare(prompt: string, sessionId: string) {
  return api<DemoCompareResponse>("/api/v1/demo/compare", {
    method: "POST",
    body: JSON.stringify({ prompt, session_id: sessionId, update_session: false, security_enabled: true, device: "cuda" }),
  });
}

export function stats(sessionId: string) {
  return api<{ session_id: string; stats: Record<string, unknown> }>(`/api/v1/sessions/${sessionId}/stats`);
}

export function history(sessionId: string) {
  return api<{ session_id: string; history: Array<{ turn_number: number; prompt: string; decision: string; risk_level: string; composite_score: number; top_threat: string }> }>(`/api/v1/sessions/${sessionId}/history`);
}

export function benchmark() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 240000);

  return api<BenchmarkResponse>("/api/v1/benchmark", {
    signal: controller.signal,
  }).catch((error) => {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("Benchmark timed out after 240s. Try again or reduce benchmark dataset size.");
    }
    throw error;
  }).finally(() => {
    clearTimeout(timeout);
  });
}

export function resetSession(sessionId: string) {
  return api<{ session_id: string; reset: boolean }>(`/api/v1/sessions/${sessionId}/reset`, { method: "POST" });
}
