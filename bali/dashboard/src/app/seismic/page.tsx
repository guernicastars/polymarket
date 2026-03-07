import Link from "next/link";
import { getAllDistricts } from "@/lib/data";
import { RiskBar, riskColor } from "@/components/risk-bar";

export default function SeismicPage() {
  const districts = getAllDistricts()
    .sort((a, b) => b.seismic_risk - a.seismic_risk);

  const volcanoData = [
    { name: "Mount Agung", elevation: 3142, lastEruption: "2019", status: "Normal", lat: -8.3433, lng: 115.5083 },
    { name: "Mount Batur", elevation: 1717, lastEruption: "2000", status: "Normal", lat: -8.2417, lng: 115.3750 },
  ];

  const seismicSources = [
    { name: "Sunda Megathrust", type: "Subduction", maxMag: 8.5, distance: "~250 km south" },
    { name: "Flores Back-Arc Thrust", type: "Thrust", maxMag: 7.5, distance: "~100 km north" },
    { name: "Bali Fault Zone", type: "Strike-slip", maxMag: 6.5, distance: "Traverses island" },
  ];

  return (
    <div>
      <h2>Seismic Risk Assessment</h2>
      <p style={{ color: "#888", marginBottom: 24 }}>
        Bali sits on the Sunda Arc — one of the most seismically active regions on Earth
      </p>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">Active Volcanoes</div>
          <div className="value" style={{ color: "#ff4466" }}>2</div>
        </div>
        <div className="stat-card">
          <div className="label">Seismic Zone</div>
          <div className="value">Zone 4 (High)</div>
        </div>
        <div className="stat-card">
          <div className="label">Design PGA</div>
          <div className="value">0.3–0.4g</div>
        </div>
        <div className="stat-card">
          <div className="label">Max Historical</div>
          <div className="value">M6.9 (2018)</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <div className="card">
          <h3>Volcanoes</h3>
          <table>
            <thead>
              <tr><th>Volcano</th><th>Elevation</th><th>Last Eruption</th><th>Status</th></tr>
            </thead>
            <tbody>
              {volcanoData.map((v) => (
                <tr key={v.name}>
                  <td style={{ fontWeight: 600, color: "#ff4466" }}>{v.name}</td>
                  <td>{v.elevation.toLocaleString()}m</td>
                  <td>{v.lastEruption}</td>
                  <td style={{ color: "#00d4aa" }}>{v.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Seismic Sources</h3>
          <table>
            <thead>
              <tr><th>Source</th><th>Type</th><th>Max M</th><th>Distance</th></tr>
            </thead>
            <tbody>
              {seismicSources.map((s) => (
                <tr key={s.name}>
                  <td style={{ fontWeight: 500 }}>{s.name}</td>
                  <td style={{ color: "#888" }}>{s.type}</td>
                  <td style={{ color: "#ff8844", fontWeight: 600 }}>M{s.maxMag}</td>
                  <td>{s.distance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: 24 }}>
        <h3>Districts by Seismic Risk</h3>
        <table>
          <thead>
            <tr>
              <th>#</th><th>District</th><th>Seismic Risk</th>
              <th>Fault</th><th>Historical</th><th>Liquefaction</th><th>PGA</th>
              <th>Volcanic Dist.</th>
            </tr>
          </thead>
          <tbody>
            {districts.slice(0, 20).map((d, i) => (
              <tr key={d.id}>
                <td style={{ color: "#666" }}>{i + 1}</td>
                <td><Link href={`/district/${d.id}`}>{d.name}</Link></td>
                <td style={{ color: riskColor(d.seismic_risk), fontWeight: 600 }}>
                  {d.seismic_risk.toFixed(1)}
                </td>
                <td style={{ color: riskColor(d.seismic_factors.fault_proximity) }}>
                  {d.seismic_factors.fault_proximity.toFixed(0)}
                </td>
                <td style={{ color: riskColor(d.seismic_factors.historical_frequency) }}>
                  {d.seismic_factors.historical_frequency.toFixed(0)}
                </td>
                <td style={{ color: riskColor(d.seismic_factors.liquefaction) }}>
                  {d.seismic_factors.liquefaction.toFixed(0)}
                </td>
                <td style={{ color: riskColor(d.seismic_factors.ground_acceleration) }}>
                  {d.seismic_factors.ground_acceleration.toFixed(0)}
                </td>
                <td>{d.volcanic_proximity_km} km</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
