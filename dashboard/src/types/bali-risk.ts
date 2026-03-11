// Bali Real Estate Risk Monitor — Type Definitions

export type RiskSeverity = "critical" | "high" | "medium" | "low";
export type RiskTrend = "worsening" | "stable" | "improving";
export type RiskCategory =
  | "legal"
  | "climate"
  | "political"
  | "financial"
  | "regulatory"
  | "market";

export interface RiskIndicator {
  id: string;
  category: RiskCategory;
  title: string;
  description: string;
  severity: RiskSeverity;
  trend: RiskTrend;
  score: number; // 0-100, higher = more risk
  last_updated: string;
  source: string;
  details: string;
  mitigation: string;
}

export interface RiskOverview {
  overall_score: number; // 0-100
  total_indicators: number;
  critical_count: number;
  high_count: number;
  medium_count: number;
  low_count: number;
  top_risk_category: RiskCategory;
  trend: RiskTrend;
}

export interface LegalRisk extends RiskIndicator {
  category: "legal";
  legal_area: string; // e.g., "Ownership Structure", "Nominee Risk", "Land Title"
  affected_structures: string[]; // e.g., ["Hak Pakai", "PT PMA", "Leasehold"]
  precedent_cases: number;
}

export interface ClimateRisk extends RiskIndicator {
  category: "climate";
  hazard_type: string; // e.g., "Flood", "Earthquake", "Volcanic", "Tsunami"
  affected_regions: string[];
  probability_10yr: number; // 0-1
  estimated_damage_pct: number; // property value impact %
}

export interface PoliticalRisk extends RiskIndicator {
  category: "political";
  risk_type: string; // e.g., "Policy Change", "Nationalism", "Corruption"
  affected_parties: string[];
  likelihood: number; // 0-1
}

export interface FinancialRisk extends RiskIndicator {
  category: "financial";
  risk_type: string; // e.g., "Currency", "Liquidity", "Credit"
  metric_value: string;
  metric_label: string;
}

export interface RegulatoryRisk extends RiskIndicator {
  category: "regulatory";
  regulation: string;
  effective_date: string;
  compliance_deadline: string;
  penalty: string;
}

export interface MarketRisk extends RiskIndicator {
  category: "market";
  risk_type: string; // e.g., "Oversupply", "Tourism Dependency", "Rental Yield"
  metric_value: string;
  metric_label: string;
}

export type AnyRisk =
  | LegalRisk
  | ClimateRisk
  | PoliticalRisk
  | FinancialRisk
  | RegulatoryRisk
  | MarketRisk;

export const SEVERITY_CONFIG: Record<
  RiskSeverity,
  { label: string; color: string; bgColor: string }
> = {
  critical: {
    label: "Critical",
    color: "text-red-400",
    bgColor: "bg-red-400/10",
  },
  high: {
    label: "High",
    color: "text-orange-400",
    bgColor: "bg-orange-400/10",
  },
  medium: {
    label: "Medium",
    color: "text-amber-400",
    bgColor: "bg-amber-400/10",
  },
  low: { label: "Low", color: "text-emerald-400", bgColor: "bg-emerald-400/10" },
};

export const TREND_CONFIG: Record<
  RiskTrend,
  { label: string; color: string; icon: string }
> = {
  worsening: { label: "Worsening", color: "text-red-400", icon: "arrow-up" },
  stable: { label: "Stable", color: "text-muted-foreground", icon: "minus" },
  improving: {
    label: "Improving",
    color: "text-emerald-400",
    icon: "arrow-down",
  },
};

export const CATEGORY_CONFIG: Record<
  RiskCategory,
  { label: string; color: string; description: string }
> = {
  legal: {
    label: "Legal",
    color: "text-red-400",
    description: "Ownership structures, nominee fraud, land title disputes",
  },
  climate: {
    label: "Climate",
    color: "text-blue-400",
    description: "Natural disasters, flooding, volcanic activity, sea level rise",
  },
  political: {
    label: "Political",
    color: "text-violet-400",
    description: "Policy changes, nationalism, foreign investment sentiment",
  },
  financial: {
    label: "Financial",
    color: "text-amber-400",
    description: "Currency risk, liquidity, mortgage access, capital controls",
  },
  regulatory: {
    label: "Regulatory",
    color: "text-orange-400",
    description: "Zoning, building codes, licensing, PP 28/2025 compliance",
  },
  market: {
    label: "Market",
    color: "text-emerald-400",
    description: "Oversupply, tourism dependency, rental yields, infrastructure",
  },
};
