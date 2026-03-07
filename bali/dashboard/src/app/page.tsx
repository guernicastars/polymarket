import Link from "next/link";
import { getAllDistricts, getOverviewStats } from "@/lib/data";
import { Grade, RiskBar, riskColor } from "@/components/risk-bar";

export default function OverviewPage() {
  const stats = getOverviewStats();
  const districts = getAllDistricts();
  const topSafe = districts.slice(0, 10);
  const topRisky = [...districts].reverse().slice(0, 10);

  return (
    <div>
      <h2>Bali Real Estate Risk Overview</h2>
      <p style={{ color: "#888", marginBottom: 24 }}>
        Risk assessment across {stats.totalDistricts} districts (kecamatan) in 9
        regencies
      </p>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">Districts Analyzed</div>
          <div className="value">{stats.totalDistricts}</div>
        </div>
        <div className="stat-card">
          <div className="label">Avg Risk Score</div>
          <div className="value" style={{ color: riskColor(stats.avgRisk) }}>
            {stats.avgRisk}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">Avg Land Price</div>
          <div className="value">${stats.avgPrice.toLocaleString()}/m²</div>
        </div>
        <div className="stat-card">
          <div className="label">Population</div>
          <div className="value">
            {(stats.totalPopulation / 1_000_000).toFixed(1)}M
          </div>
        </div>
      </div>

      <div className="stats-grid">
        {Object.entries(stats.grades)
          .sort()
          .map(([grade, count]) => (
            <div className="stat-card" key={grade}>
              <div className="label">Grade {grade}</div>
              <div className="value">
                <Grade grade={grade} /> {count} districts
              </div>
            </div>
          ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <div className="card">
          <h3>Safest for Investment</h3>
          <table>
            <thead>
              <tr>
                <th>District</th>
                <th>Grade</th>
                <th>Risk</th>
                <th>$/m²</th>
              </tr>
            </thead>
            <tbody>
              {topSafe.map((d) => (
                <tr key={d.id}>
                  <td>
                    <Link href={`/district/${d.id}`}>{d.name}</Link>
                    <div style={{ fontSize: 11, color: "#666" }}>
                      {d.regency}
                    </div>
                  </td>
                  <td>
                    <Grade grade={d.investment_grade} />
                  </td>
                  <td style={{ color: riskColor(d.composite_risk) }}>
                    {d.composite_risk.toFixed(1)}
                  </td>
                  <td>${d.avg_land_price_usd_m2.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Highest Risk Districts</h3>
          <table>
            <thead>
              <tr>
                <th>District</th>
                <th>Grade</th>
                <th>Risk</th>
                <th>Top Risk Factor</th>
              </tr>
            </thead>
            <tbody>
              {topRisky.map((d) => {
                const risks = [
                  { name: "Env", score: d.env_risk },
                  { name: "Seismic", score: d.seismic_risk },
                  { name: "Legal", score: d.legal_risk },
                  { name: "Admin", score: d.admin_risk },
                ];
                const topFactor = risks.sort((a, b) => b.score - a.score)[0];
                return (
                  <tr key={d.id}>
                    <td>
                      <Link href={`/district/${d.id}`}>{d.name}</Link>
                      <div style={{ fontSize: 11, color: "#666" }}>
                        {d.regency}
                      </div>
                    </td>
                    <td>
                      <Grade grade={d.investment_grade} />
                    </td>
                    <td style={{ color: riskColor(d.composite_risk) }}>
                      {d.composite_risk.toFixed(1)}
                    </td>
                    <td style={{ color: riskColor(topFactor.score), fontSize: 13 }}>
                      {topFactor.name} ({topFactor.score.toFixed(0)})
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: 24 }}>
        <h3>Risk Distribution by Category</h3>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          <div>
            <h3 style={{ fontSize: 14 }}>Environmental Risk</h3>
            {districts.slice(0, 5).map((d) => (
              <RiskBar key={d.id} label={d.name} score={d.env_risk} />
            ))}
            <div style={{ marginTop: 8, fontSize: 12, color: "#666" }}>
              Showing top 5 safest. <Link href="/districts">See all →</Link>
            </div>
          </div>
          <div>
            <h3 style={{ fontSize: 14 }}>Legal Risk</h3>
            {[...districts]
              .sort((a, b) => b.legal_risk - a.legal_risk)
              .slice(0, 5)
              .map((d) => (
                <RiskBar key={d.id} label={d.name} score={d.legal_risk} />
              ))}
            <div style={{ marginTop: 8, fontSize: 12, color: "#666" }}>
              Showing top 5 riskiest. <Link href="/legal">Legal guide →</Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
