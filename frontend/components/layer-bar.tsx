type Props = {
  name: string;
  score: number;
  label: string;
};

export function LayerBar({ name, score, label }: Props) {
  const width = `${Math.max(2, Math.round(score * 100))}%`;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-slate-200">{name}</span>
        <span className="text-slate-400">{label} · {score.toFixed(3)}</span>
      </div>
      <div className="h-2 w-full rounded bg-slate-800">
        <div className="h-2 rounded bg-brand" style={{ width }} />
      </div>
    </div>
  );
}
