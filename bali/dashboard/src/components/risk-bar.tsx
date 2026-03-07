export function riskColor(score: number): string {
  if (score < 25) return "#00d4aa";
  if (score < 40) return "#4ecdc4";
  if (score < 55) return "#ffe66d";
  if (score < 70) return "#ff8844";
  return "#ff4466";
}

export function RiskBar({
  label,
  score,
  showValue = true,
}: {
  label: string;
  score: number;
  showValue?: boolean;
}) {
  const color = riskColor(score);
  return (
    <div style={{ margin: "6px 0" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 12,
          marginBottom: 2,
        }}
      >
        <span style={{ color: "#888" }}>{label}</span>
        {showValue && (
          <span style={{ color, fontWeight: 600 }}>{score.toFixed(1)}</span>
        )}
      </div>
      <div className="risk-bar">
        <div
          className="risk-bar-fill"
          style={{ width: `${score}%`, background: color }}
        />
      </div>
    </div>
  );
}

export function Grade({ grade }: { grade: string }) {
  return <span className={`grade grade-${grade}`}>{grade}</span>;
}
