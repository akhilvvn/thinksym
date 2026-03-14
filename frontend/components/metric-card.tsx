type Props = {
  label: string;
  value: string | number;
  tone?: "default" | "safe" | "warn" | "danger";
};

const toneClass: Record<NonNullable<Props["tone"]>, string> = {
  default: "text-slate-100",
  safe: "text-safe",
  warn: "text-warn",
  danger: "text-danger",
};

export function MetricCard({ label, value, tone = "default" }: Props) {
  return (
    <div className="panel p-4">
      <p className="text-xs uppercase tracking-wider text-slate-400">{label}</p>
      <p className={`mt-2 text-2xl font-semibold ${toneClass[tone]}`}>{value}</p>
    </div>
  );
}
