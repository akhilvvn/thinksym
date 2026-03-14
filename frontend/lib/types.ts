export type Verdict = {
  decision: "ALLOW" | "WARN" | "BLOCK";
  risk_level: string;
  composite_score: number;
  confidence: number;
  primary_reason: string;
  contributing_factors: string[];
  flags: Record<string, string | boolean>;
};

export type AnalyzeResponse = {
  session_id: string;
  turn_number: number;
  prompt: string;
  verdict: Verdict;
  top_threat: string;
  timings_ms: Record<string, number>;
  total_time_ms: number;
  layers: Array<{ name: string; score: number; label: string }>;
  errors: string[];
};

export type ChatResponse = {
  blocked: boolean;
  protection_enabled: boolean;
  analysis: AnalyzeResponse;
  llm_text: string;
  llm_model: string;
  llm_backend: string;
  llm_latency_ms: number;
  llm_error: string;
};

export type DemoCompareResponse = {
  prompt: string;
  baseline: ChatResponse;
  protected: ChatResponse;
};

export type BenchmarkResponse = {
  total_adversarial: number;
  blocked_adversarial: number;
  warned_adversarial: number;
  allowed_adversarial: number;
  total_benign: number;
  blocked_benign: number;
  warned_benign: number;
  allowed_benign: number;
  adversarial_block_rate: number;
  benign_false_positive_rate: number;
};
