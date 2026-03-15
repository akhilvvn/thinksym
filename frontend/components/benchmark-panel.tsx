"use client";

import { BenchmarkResponse } from "@/lib/types";

function RingGauge({
  value,
  label,
  description,
  higherIsBetter = true,
}: {
  value: number;
  label: string;
  description: string;
  higherIsBetter?: boolean;
}) {
  const r = 48;
  const circumference = 2 * Math.PI * r;
  const clamped = Math.min(1, Math.max(0, value));
  const offset = circumference * (1 - clamped);

  let color: string;
  if (higherIsBetter) {
    color = clamped >= 0.9 ? "#22c55e" : clamped >= 0.7 ? "#f59e0b" : "#ef4444";
  } else {
    color = clamped <= 0.05 ? "#22c55e" : clamped <= 0.15 ? "#f59e0b" : "#ef4444";
  }

  const pct = (value * 100).toFixed(1);

  return (
    <div className="flex flex-col items-center gap-3">
      <svg viewBox="0 0 120 120" width={140} height={140}>
        {/* Track ring */}
        <circle cx={60} cy={60} r={r} fill="none" stroke="#1e293b" strokeWidth={10} />
        {/* Progress ring */}
        <circle
          cx={60}
          cy={60}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={10}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 60 60)"
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
        {/* Percentage text */}
        <text x={60} y={54} textAnchor="middle" fill="white" fontSize={18} fontWeight="700" fontFamily="inherit">
          {pct}%
        </text>
        {/* Description text */}
        <text x={60} y={70} textAnchor="middle" fill="#64748b" fontSize={8} fontFamily="inherit">
          {description}
        </text>
      </svg>
      <p className="text-sm font-semibold text-slate-200 text-center leading-5 max-w-[120px]">{label}</p>
    </div>
  );
}

function VerdictBar({
  label,
  blocked,
  warned,
  allowed,
  total,
  isAdversarial,
}: {
  label: string;
  blocked: number;
  warned: number;
  allowed: number;
  total: number;
  isAdversarial: boolean;
}) {
  const bPct = total > 0 ? (blocked / total) * 100 : 0;
  const wPct = total > 0 ? (warned / total) * 100 : 0;
  const aPct = total > 0 ? (allowed / total) * 100 : 0;

  // For adversarial: BLOCK/WARN = good (green intent), ALLOW = bad
  // For benign: ALLOW = good, BLOCK/WARN = bad (false positives)
  const blockColor  = isAdversarial ? "#22c55e" : "#ef4444";
  const warnColor   = isAdversarial ? "#f59e0b" : "#f97316";
  const allowColor  = isAdversarial ? "#ef4444" : "#22c55e";

  const blockLabel  = isAdversarial ? "BLOCK (correct)" : "BLOCK (FP)";
  const warnLabel   = isAdversarial ? "WARN (correct)" : "WARN (FP)";
  const allowLabel  = isAdversarial ? "ALLOW (missed)" : "ALLOW (correct)";

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-200">{label}</span>
        <span className="text-xs text-slate-400">{total} prompts</span>
      </div>

      {/* Stacked bar */}
      <div className="flex h-7 w-full overflow-hidden rounded-lg border border-border">
        {blocked > 0 && (
          <div
            style={{ width: `${bPct}%`, backgroundColor: blockColor }}
            className="flex items-center justify-center transition-all duration-500"
            title={`${blockLabel}: ${blocked}`}
          >
            <span className="text-[11px] font-bold text-white drop-shadow">{blocked}</span>
          </div>
        )}
        {warned > 0 && (
          <div
            style={{ width: `${wPct}%`, backgroundColor: warnColor }}
            className="flex items-center justify-center transition-all duration-500"
            title={`${warnLabel}: ${warned}`}
          >
            <span className="text-[11px] font-bold text-white drop-shadow">{warned}</span>
          </div>
        )}
        {allowed > 0 && (
          <div
            style={{ width: `${aPct}%`, backgroundColor: allowColor }}
            className="flex items-center justify-center transition-all duration-500"
            title={`${allowLabel}: ${allowed}`}
          >
            <span className="text-[11px] font-bold text-white drop-shadow">{allowed}</span>
          </div>
        )}
        {total === 0 && (
          <div className="w-full bg-slate-800 flex items-center justify-center">
            <span className="text-xs text-slate-500">No data</span>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full flex-shrink-0" style={{ backgroundColor: blockColor }} />
          {blockLabel} — {blocked}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full flex-shrink-0" style={{ backgroundColor: warnColor }} />
          {warnLabel} — {warned}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full flex-shrink-0" style={{ backgroundColor: allowColor }} />
          {allowLabel} — {allowed}
        </span>
      </div>
    </div>
  );
}

function StatTile({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-xl border border-border bg-slate-950 px-5 py-4 text-center min-w-[90px]">
      <p className="text-2xl font-bold text-slate-100">{value}</p>
      <p className="mt-1 text-xs text-slate-400 leading-4">{label}</p>
    </div>
  );
}

export function BenchmarkPanel({
  bench,
  running,
  error,
  durationMs,
}: {
  bench: BenchmarkResponse | null;
  running: boolean;
  error: string;
  durationMs: number | null;
}) {
  if (running) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-12 text-center">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-slate-700 border-t-[#7c9cff]" />
        <p className="text-sm text-slate-300">
          Benchmark running — evaluating adversarial and benign prompt sets.
          <br />
          <span className="text-slate-500">This can take 30–120 seconds.</span>
        </p>
      </div>
    );
  }

  if (error) {
    return <p className="py-2 text-sm text-[#ef4444]">Benchmark failed: {error}</p>;
  }

  if (!bench) {
    return (
      <p className="py-2 text-sm text-slate-400">
        Click <span className="font-medium text-slate-300">Benchmark</span> in the header to evaluate pipeline accuracy across adversarial and benign prompt sets.
      </p>
    );
  }

  return (
    <div className="space-y-8">
      {/* Row 1: gauges + stat tiles */}
      <div className="flex flex-wrap items-center justify-between gap-6">
        {/* Ring gauges */}
        <div className="flex gap-10">
          <RingGauge
            value={bench.adversarial_block_rate}
            label="Adversarial Block Rate"
            description="higher is better"
            higherIsBetter
          />
          <RingGauge
            value={bench.benign_false_positive_rate}
            label="False Positive Rate"
            description="lower is better"
            higherIsBetter={false}
          />
        </div>

        {/* Stat tiles */}
        <div className="flex flex-wrap gap-3">
          <StatTile value={String(bench.total_adversarial)} label="Adversarial Prompts" />
          <StatTile value={String(bench.total_benign)} label="Benign Prompts" />
          <StatTile
            value={`${bench.blocked_adversarial}/${bench.total_adversarial}`}
            label="Correctly Blocked"
          />
          <StatTile
            value={`${bench.blocked_benign + bench.warned_benign}/${bench.total_benign}`}
            label="False Positives"
          />
          {durationMs !== null && (
            <StatTile value={`${(durationMs / 1000).toFixed(1)}s`} label="Run Time" />
          )}
        </div>
      </div>

      {/* Row 2: verdict bars */}
      <div className="space-y-6">
        <VerdictBar
          label="Adversarial Set"
          blocked={bench.blocked_adversarial}
          warned={bench.warned_adversarial}
          allowed={bench.allowed_adversarial}
          total={bench.total_adversarial}
          isAdversarial
        />
        <VerdictBar
          label="Benign Set"
          blocked={bench.blocked_benign}
          warned={bench.warned_benign}
          allowed={bench.allowed_benign}
          total={bench.total_benign}
          isAdversarial={false}
        />
      </div>
    </div>
  );
}
