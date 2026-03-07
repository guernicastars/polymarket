import type { District, RiskData } from "@/types";
import riskDataJson from "../../risk-data.json";

const riskData = riskDataJson as RiskData;

export function getAllDistricts(): District[] {
  return riskData.districts;
}

export function getDistrict(id: string): District | undefined {
  return riskData.districts.find((d) => d.id === id);
}

export function getDistrictsByRegency(regency: string): District[] {
  return riskData.districts.filter((d) => d.regency === regency);
}

export function getRegencies(): string[] {
  return [...new Set(riskData.districts.map((d) => d.regency))].sort();
}

export function getOverviewStats() {
  const districts = riskData.districts;
  const grades: Record<string, number> = {};
  districts.forEach((d) => {
    grades[d.investment_grade] = (grades[d.investment_grade] || 0) + 1;
  });

  return {
    totalDistricts: districts.length,
    avgRisk: Math.round(
      districts.reduce((s, d) => s + d.composite_risk, 0) / districts.length * 10
    ) / 10,
    avgPrice: Math.round(
      districts.reduce((s, d) => s + d.avg_land_price_usd_m2, 0) / districts.length
    ),
    grades,
    totalPopulation: districts.reduce((s, d) => s + d.population, 0),
    coastalCount: districts.filter((d) => d.coastal).length,
  };
}

export function getGeneratedAt(): string {
  return riskData.generated_at;
}
