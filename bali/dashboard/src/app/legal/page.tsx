import Link from "next/link";
import { getAllDistricts } from "@/lib/data";
import { riskColor } from "@/components/risk-bar";

export default function LegalPage() {
  const districts = getAllDistricts().sort((a, b) => b.legal_risk - a.legal_risk);

  const ownershipPathways = [
    { name: "Hak Pakai (Right to Use)", risk: "Low", term: "25 + 20 + 20 years", description: "Safest legal option for foreigners. Direct registration at BPN. Limited to residential use, one property per person." },
    { name: "PT PMA (Foreign Company)", risk: "Medium", term: "HGB: 30 + 20 + 20 years", description: "Set up an Indonesian company with foreign ownership (via BKPM/OSS). Can hold HGB title. Requires minimum investment and local director." },
    { name: "Leasehold (Hak Sewa)", risk: "Low", term: "25-30 years typical", description: "Long-term lease from Indonesian owner. No ownership rights but simple and clear. Common for villas and commercial properties." },
    { name: "Nominee Arrangement", risk: "EXTREME", term: "N/A (illegal)", description: "Using an Indonesian nominee to hold SHM title. Technically illegal under Indonesian law. Zero legal protection — nominee can claim the property at any time." },
  ];

  const titleTypes = [
    { name: "SHM (Sertifikat Hak Milik)", security: "Highest", who: "Indonesian citizens only", note: "Full freehold ownership. Strongest title. Cannot be held by foreigners." },
    { name: "HGB (Hak Guna Bangunan)", security: "High", who: "PT PMA companies", note: "Right to build. 30+20+20 year terms. Must be actively used." },
    { name: "Hak Pakai", security: "High", who: "Foreigners directly", note: "Right to use. 25+20+20 years. One property only." },
    { name: "Girik (Customary)", security: "Low", who: "Traditional holders", note: "Unregistered customary land. Common in rural areas. Boundary disputes frequent. Requires conversion before sale." },
    { name: "Strata Title", security: "Medium-High", who: "Anyone (apartments)", note: "For condominium/apartment units. Building-dependent. Foreign quota limits may apply." },
  ];

  return (
    <div>
      <h2>Legal Risk Guide for Foreign Investors</h2>
      <p style={{ color: "#888", marginBottom: 24 }}>
        Indonesian law prohibits foreign freehold ownership. Understanding legal pathways is critical.
      </p>

      <div className="card" style={{ marginBottom: 24, borderColor: "#ff4466" }}>
        <h3 style={{ color: "#ff4466" }}>Critical Warning</h3>
        <p style={{ fontSize: 14, lineHeight: 1.6 }}>
          <strong>Nominee arrangements are illegal</strong> under Indonesian Agrarian Law (UUPA No. 5/1960).
          Despite being widely used, nominees can legally claim full ownership of the property at any time.
          There is <strong>zero legal recourse</strong> for the foreign investor in this scenario.
          Always use a properly structured Hak Pakai or PT PMA arrangement.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <h3>Ownership Pathways for Foreigners</h3>
        <table>
          <thead>
            <tr><th>Pathway</th><th>Risk Level</th><th>Term</th><th>Description</th></tr>
          </thead>
          <tbody>
            {ownershipPathways.map((p) => (
              <tr key={p.name}>
                <td style={{ fontWeight: 600 }}>{p.name}</td>
                <td style={{ color: p.risk === "EXTREME" ? "#ff4466" : p.risk === "Medium" ? "#ffe66d" : "#00d4aa", fontWeight: 600 }}>
                  {p.risk}
                </td>
                <td style={{ whiteSpace: "nowrap" }}>{p.term}</td>
                <td style={{ fontSize: 13, color: "#aaa" }}>{p.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <h3>Land Title Types</h3>
        <table>
          <thead>
            <tr><th>Title Type</th><th>Security</th><th>Who Can Hold</th><th>Notes</th></tr>
          </thead>
          <tbody>
            {titleTypes.map((t) => (
              <tr key={t.name}>
                <td style={{ fontWeight: 600 }}>{t.name}</td>
                <td style={{ color: t.security === "Highest" || t.security === "High" ? "#00d4aa" : t.security === "Low" ? "#ff4466" : "#ffe66d" }}>
                  {t.security}
                </td>
                <td>{t.who}</td>
                <td style={{ fontSize: 13, color: "#aaa" }}>{t.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Districts by Legal Risk</h3>
        <table>
          <thead>
            <tr>
              <th>#</th><th>District</th><th>Legal Risk</th>
              <th>Ownership</th><th>Title</th><th>Zoning</th><th>Disputes</th>
              <th>Foreign %</th>
            </tr>
          </thead>
          <tbody>
            {districts.slice(0, 25).map((d, i) => (
              <tr key={d.id}>
                <td style={{ color: "#666" }}>{i + 1}</td>
                <td><Link href={`/district/${d.id}`}>{d.name}</Link></td>
                <td style={{ color: riskColor(d.legal_risk), fontWeight: 600 }}>
                  {d.legal_risk.toFixed(1)}
                </td>
                <td style={{ color: riskColor(d.legal_factors.ownership_pathway) }}>
                  {d.legal_factors.ownership_pathway.toFixed(0)}
                </td>
                <td style={{ color: riskColor(d.legal_factors.title_security) }}>
                  {d.legal_factors.title_security.toFixed(0)}
                </td>
                <td style={{ color: riskColor(d.legal_factors.zoning_compliance) }}>
                  {d.legal_factors.zoning_compliance.toFixed(0)}
                </td>
                <td style={{ color: riskColor(d.legal_factors.dispute_density) }}>
                  {d.legal_factors.dispute_density.toFixed(0)}
                </td>
                <td>{(d.foreign_investor_density * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
