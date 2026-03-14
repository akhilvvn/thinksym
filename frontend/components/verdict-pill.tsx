type Props = {
  decision: "ALLOW" | "WARN" | "BLOCK";
};

export function VerdictPill({ decision }: Props) {
  const className =
    decision === "BLOCK"
      ? "badge bg-danger/20 text-danger"
      : decision === "WARN"
      ? "badge bg-warn/20 text-warn"
      : "badge bg-safe/20 text-safe";

  return <span className={className}>{decision}</span>;
}
