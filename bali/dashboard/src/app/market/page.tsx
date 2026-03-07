import Link from "next/link";
import { getAllDistricts, getRegencies } from "@/lib/data";
import { Grade, riskColor } from "@/components/risk-bar";

export default function MarketPage() {
  const districts = getAllDistricts();
  const byPrice = [...districts].sort(
    (a, b) => b.avg_land_price_usd_m2 - a.avg_land_price_usd_m2
  );
  const regencies = getRegencies();

  // Regency-level aggregation
  const regencyStats = regencies.map((r) => {
    const ds = districts.filter((d) => d.regency === r);
    const avgPrice =
      ds.reduce((s, d) => s + d.avg_land_price_usd_m2, 0) / ds.length;
    const avgRisk =
      ds.reduce((s, d) => s + d.composite_risk, 0) / ds.length;
    const maxPrice = Math.max(...ds.map((d) => d.avg_land_price_usd_m2));
    const minPrice = Math.min(...ds.map((d) => d.avg_land_price_usd_m2));
    return {
      regency: r,
      districts: ds.length,
      avgPrice,
      avgRisk,
      maxPrice,
      minPrice,
      totalPop: ds.reduce((s, d) => s + d.population, 0),
    };
  }).sort((a, b) => b.avgPrice - a.avgPrice);

  // Investment value score: low risk + reasonable price
  const valueRanked = districts
    .map((d) => ({
      ...d,
      valueScore: d.composite_risk * 0.6 + Math.min(100, d.avg_land_price_usd_m2 / 70) * 0.4,
    }))
    .sort((a, b) => a.valueScore - b.valueScore);

  return (
    <div>
      <h2>Property Market Data</h2>
      <p style={{ color: "#888", marginBottom: 24 }}>
        Land prices and investment value analysis across Bali
      </p>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">Most Expensive</div>
          <div className="value">${byPrice[0].avg_land_price_usd_m2.toLocaleString()}/m²</div>
          <div style={{ fontSize: 12, color: "#888" }}>{byPrice[0].name}</div>
        </div>
        <div className="stat-card">
          <div className="label">Most Affordable</div>
          <div className="value">
            ${byPrice[byPrice.length - 1].avg_land_price_usd_m2.toLocaleString()}/m²
          </div>
          <div style={{ fontSize: 12, color: "#888" }}>
            {byPrice[byPrice.length - 1].name}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">Price Range</div>
          <div className="value">97x</div>
          <div style={{ fontSize: 12, color: "#888" }}>
            ${byPrice[byPrice.length - 1].avg_land_price_usd_m2} – ${byPrice[0].avg_land_price_usd_m2.toLocaleString()}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">Best Value District</div>
          <div className="value" style={{ fontSize: 20 }}>{valueRanked[0].name}</div>
          <div style={{ fontSize: 12, color: "#888" }}>
            Score: {valueRanked[0].valueScore.toFixed(1)}
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <div className="card">
          <h3>Regency Price Comparison</h3>
          <table>
            <thead>
              <tr><th>Regency</th><th>Avg $/m²</th><th>Range</th><th>Avg Risk</th><th>Districts</th></tr>
            </thead>
            <tbody>
              {regencyStats.map((r) => (
                <tr key={r.regency}>
                  <td style={{ fontWeight: 600, textTransform: "capitalize" }}>{r.regency}</td>
                  <td>${Math.round(r.avgPrice).toLocaleString()}</td>
                  <td style={{ fontSize: 12, color: "#888" }}>
                    ${r.minPrice.toLocaleString()} – ${r.maxPrice.toLocaleString()}
                  </td>
                  <td style={{ color: riskColor(r.avgRisk) }}>{r.avgRisk.toFixed(1)}</td>
                  <td>{r.districts}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Top Investment Value (Risk-Adjusted)</h3>
          <table>
            <thead>
              <tr><th>#</th><th>District</th><th>Grade</th><th>Risk</th><th>$/m²</th><th>Value</th></tr>
            </thead>
            <tbody>
              {valueRanked.slice(0, 15).map((d, i) => (
                <tr key={d.id}>
                  <td style={{ color: "#666" }}>{i + 1}</td>
                  <td>
                    <Link href={`/district/${d.id}`}>{d.name}</Link>
                    <div style={{ fontSize: 11, color: "#555" }}>{d.regency}</div>
                  </td>
                  <td><Grade grade={d.investment_grade} /></td>
                  <td style={{ color: riskColor(d.composite_risk) }}>{d.composite_risk.toFixed(1)}</td>
                  <td>${d.avg_land_price_usd_m2.toLocaleString()}</td>
                  <td style={{ fontWeight: 600, color: "#00d4aa" }}>{d.valueScore.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: 24 }}>
        <h3>All Districts by Price</h3>
        <table>
          <thead>
            <tr>
              <th>#</th><th>District</th><th>$/m²</th><th>Grade</th><th>Risk</th>
              <th>Tourism</th><th>Foreign %</th><th>Zone</th>
            </tr>
          </thead>
          <tbody>
            {byPrice.map((d, i) => (
              <tr key={d.id}>
                <td style={{ color: "#666" }}>{i + 1}</td>
                <td>
                  <Link href={`/district/${d.id}`}>{d.name}</Link>
                  <div style={{ fontSize: 11, color: "#555" }}>{d.regency}</div>
                </td>
                <td style={{ fontWeight: 600 }}>${d.avg_land_price_usd_m2.toLocaleString()}</td>
                <td><Grade grade={d.investment_grade} /></td>
                <td style={{ color: riskColor(d.composite_risk) }}>{d.composite_risk.toFixed(1)}</td>
                <td>{(d.tourism_intensity * 100).toFixed(0)}%</td>
                <td>{(d.foreign_investor_density * 100).toFixed(0)}%</td>
                <td style={{ color: "#888" }}>{d.dominant_zone}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
