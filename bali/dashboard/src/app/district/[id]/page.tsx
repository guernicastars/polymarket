import { notFound } from "next/navigation";
import Link from "next/link";
import { getDistrict, getAllDistricts } from "@/lib/data";
import { Grade, RiskBar, riskColor } from "@/components/risk-bar";

export function generateStaticParams() {
  return getAllDistricts().map((d) => ({ id: d.id }));
}

export default async function DistrictDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const d = getDistrict(id);
  if (!d) notFound();

  const factorLabels: Record<string, string> = {
    volcanic: "Volcanic Proximity",
    flood: "Flood Risk",
    tsunami: "Tsunami Risk",
    landslide: "Landslide Risk",
    coastal_erosion: "Coastal Erosion",
    fault_proximity: "Fault Proximity",
    historical_frequency: "Historical Earthquakes",
    liquefaction: "Liquefaction",
    ground_acceleration: "Ground Acceleration (PGA)",
    ownership_pathway: "Ownership Pathway",
    title_security: "Title Security",
    zoning_compliance: "Zoning Compliance",
    dispute_density: "Dispute Density",
    permit_complexity: "Permit Complexity",
    infrastructure_quality: "Infrastructure Gap",
    bureaucratic_complexity: "Bureaucratic Complexity",
    utility_access: "Utility Access Gap",
  };

  return (
    <div>
      <Link href="/districts" style={{ fontSize: 13, color: "#888" }}>
        ← Back to all districts
      </Link>

      <div style={{ display: "flex", alignItems: "center", gap: 16, margin: "16px 0" }}>
        <h2 style={{ margin: 0 }}>{d.name}</h2>
        <Grade grade={d.investment_grade} />
      </div>
      <p style={{ color: "#888", marginBottom: 24 }}>
        {d.regency.charAt(0).toUpperCase() + d.regency.slice(1)} Regency · {d.dominant_zone} zone
      </p>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">Composite Risk</div>
          <div className="value" style={{ color: riskColor(d.composite_risk) }}>
            {d.composite_risk.toFixed(1)}
            <span style={{ fontSize: 14, color: "#666" }}>/100</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="label">Land Price</div>
          <div className="value">${d.avg_land_price_usd_m2.toLocaleString()}/m²</div>
        </div>
        <div className="stat-card">
          <div className="label">Population</div>
          <div className="value">{d.population.toLocaleString()}</div>
        </div>
        <div className="stat-card">
          <div className="label">Area</div>
          <div className="value">{d.area_km2.toFixed(0)} km²</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <div className="card">
          <h3>Risk Summary</h3>
          <RiskBar label="Environmental" score={d.env_risk} />
          <RiskBar label="Seismological" score={d.seismic_risk} />
          <RiskBar label="Legal" score={d.legal_risk} />
          <RiskBar label="Administrative" score={d.admin_risk} />
        </div>

        <div className="card">
          <h3>District Profile</h3>
          <div style={{ display: "grid", gap: 8 }}>
            {[
              ["Elevation", `${d.elevation_m}m`],
              ["Coastal", d.coastal ? "Yes" : "No"],
              ["Volcanic Proximity", `${d.volcanic_proximity_km} km`],
              ["Tourism Intensity", `${(d.tourism_intensity * 100).toFixed(0)}%`],
              ["Infrastructure Index", `${(d.infrastructure_index * 100).toFixed(0)}%`],
              ["Foreign Investor Density", `${(d.foreign_investor_density * 100).toFixed(0)}%`],
              ["Dominant Title", d.dominant_title.toUpperCase()],
              ["Zone", d.dominant_zone],
            ].map(([label, value]) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid var(--border)", fontSize: 14 }}>
                <span style={{ color: "#888" }}>{label}</span>
                <span style={{ fontWeight: 500 }}>{value}</span>
              </div>
            ))}
          </div>
          {d.tags.length > 0 && (
            <div style={{ marginTop: 12 }}>
              {d.tags.map((tag) => (
                <span key={tag} className="tag">{tag}</span>
              ))}
            </div>
          )}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginTop: 24 }}>
        <div className="card">
          <h3>Environmental Factors</h3>
          {Object.entries(d.env_factors).map(([key, score]) => (
            <RiskBar key={key} label={factorLabels[key] || key} score={score} />
          ))}
        </div>
        <div className="card">
          <h3>Seismological Factors</h3>
          {Object.entries(d.seismic_factors).map(([key, score]) => (
            <RiskBar key={key} label={factorLabels[key] || key} score={score} />
          ))}
        </div>
        <div className="card">
          <h3>Legal Factors</h3>
          {Object.entries(d.legal_factors).map(([key, score]) => (
            <RiskBar key={key} label={factorLabels[key] || key} score={score} />
          ))}
        </div>
        <div className="card">
          <h3>Administrative Factors</h3>
          {Object.entries(d.admin_factors).map(([key, score]) => (
            <RiskBar key={key} label={factorLabels[key] || key} score={score} />
          ))}
        </div>
      </div>
    </div>
  );
}
