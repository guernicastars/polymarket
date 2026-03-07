"use client";

import { useState } from "react";
import Link from "next/link";
import { getAllDistricts, getRegencies } from "@/lib/data";
import { Grade, riskColor } from "@/components/risk-bar";

type SortKey =
  | "composite_risk"
  | "avg_land_price_usd_m2"
  | "env_risk"
  | "seismic_risk"
  | "legal_risk"
  | "admin_risk"
  | "population"
  | "name";

export default function DistrictsPage() {
  const allDistricts = getAllDistricts();
  const regencies = getRegencies();
  const [search, setSearch] = useState("");
  const [regency, setRegency] = useState("all");
  const [sortBy, setSortBy] = useState<SortKey>("composite_risk");
  const [sortAsc, setSortAsc] = useState(true);

  const filtered = allDistricts
    .filter((d) => {
      if (search && !d.name.toLowerCase().includes(search.toLowerCase())) return false;
      if (regency !== "all" && d.regency !== regency) return false;
      return true;
    })
    .sort((a, b) => {
      const aVal = sortBy === "name" ? a.name : (a[sortBy] as number);
      const bVal = sortBy === "name" ? b.name : (b[sortBy] as number);
      if (typeof aVal === "string") return sortAsc ? aVal.localeCompare(bVal as string) : (bVal as string).localeCompare(aVal);
      return sortAsc ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
    });

  function toggleSort(key: SortKey) {
    if (sortBy === key) setSortAsc(!sortAsc);
    else { setSortBy(key); setSortAsc(key === "name"); }
  }

  return (
    <div>
      <h2>All Districts</h2>
      <p style={{ color: "#888", marginBottom: 16 }}>
        Compare risk scores across {allDistricts.length} kecamatan
      </p>

      <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
        <input
          className="search-input"
          placeholder="Search district..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ maxWidth: 300 }}
        />
        <select
          value={regency}
          onChange={(e) => setRegency(e.target.value)}
          style={{
            background: "var(--bg)", border: "1px solid var(--border)",
            borderRadius: 8, padding: "10px 14px", color: "var(--text)", fontSize: 14,
          }}
        >
          <option value="all">All Regencies</option>
          {regencies.map((r) => (
            <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
          ))}
        </select>
      </div>

      <div className="card">
        <table>
          <thead>
            <tr>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("name")}>#</th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("name")}>District</th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("composite_risk")}>Risk {sortBy === "composite_risk" ? (sortAsc ? "↑" : "↓") : ""}</th>
              <th>Grade</th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("env_risk")}>Env</th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("seismic_risk")}>Seismic</th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("legal_risk")}>Legal</th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("admin_risk")}>Admin</th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("avg_land_price_usd_m2")}>$/m²</th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("population")}>Pop.</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((d, i) => (
              <tr key={d.id}>
                <td style={{ color: "#666" }}>{i + 1}</td>
                <td>
                  <Link href={`/district/${d.id}`}>{d.name}</Link>
                  <div style={{ fontSize: 11, color: "#555" }}>{d.regency}</div>
                </td>
                <td style={{ color: riskColor(d.composite_risk), fontWeight: 600 }}>
                  {d.composite_risk.toFixed(1)}
                </td>
                <td><Grade grade={d.investment_grade} /></td>
                <td style={{ color: riskColor(d.env_risk) }}>{d.env_risk.toFixed(0)}</td>
                <td style={{ color: riskColor(d.seismic_risk) }}>{d.seismic_risk.toFixed(0)}</td>
                <td style={{ color: riskColor(d.legal_risk) }}>{d.legal_risk.toFixed(0)}</td>
                <td style={{ color: riskColor(d.admin_risk) }}>{d.admin_risk.toFixed(0)}</td>
                <td>${d.avg_land_price_usd_m2.toLocaleString()}</td>
                <td>{(d.population / 1000).toFixed(0)}k</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ padding: 12, fontSize: 13, color: "#666" }}>
          Showing {filtered.length} of {allDistricts.length} districts
        </div>
      </div>
    </div>
  );
}
