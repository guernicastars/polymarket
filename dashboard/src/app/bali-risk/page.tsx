import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  BaliRiskStatsCards,
} from "@/components/bali-risk-stats-cards";
import { BaliRiskTabs } from "@/components/bali-risk-tabs";
import { BaliRiskHeatmap } from "@/components/bali-risk-heatmap";
import {
  getRiskOverview,
  getRisksByCategory,
} from "@/lib/bali-risk-data";
import type { RiskCategory } from "@/types/bali-risk";

export const dynamic = "force-dynamic";

const CATEGORIES: RiskCategory[] = [
  "legal",
  "climate",
  "political",
  "financial",
  "regulatory",
  "market",
];

export default function BaliRiskPage() {
  const overview = getRiskOverview();
  const risksByCategory = Object.fromEntries(
    CATEGORIES.map((cat) => [cat, getRisksByCategory(cat)])
  ) as Record<RiskCategory, ReturnType<typeof getRisksByCategory>>;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <h1 className="text-2xl font-bold tracking-tight">
            Bali Real Estate Risk Monitor
          </h1>
          <span className="text-xs bg-amber-400/10 text-amber-400 px-2 py-0.5 rounded-full">
            Live Intel
          </span>
        </div>
        <p className="text-muted-foreground text-sm mt-1">
          Comprehensive risk intelligence for foreign real estate investment in
          Bali, Indonesia. Monitors legal, climate, political, financial,
          regulatory, and market risks with actionable mitigation strategies.
        </p>
      </div>

      {/* Stats Cards */}
      <BaliRiskStatsCards overview={overview} />

      {/* Risk Heatmap */}
      <BaliRiskHeatmap risksByCategory={risksByCategory} />

      {/* Risk Detail Tabs */}
      <Card className="bg-[#111118] border-[#1e1e2e]">
        <CardHeader>
          <CardTitle className="text-base">
            Risk Indicators by Category
          </CardTitle>
          <p className="text-xs text-muted-foreground">
            Click any risk to expand details and mitigation strategies
          </p>
        </CardHeader>
        <CardContent>
          <BaliRiskTabs risksByCategory={risksByCategory} />
        </CardContent>
      </Card>

      {/* Disclaimer */}
      <div className="rounded-lg border border-[#1e1e2e] bg-[#111118] p-4">
        <p className="text-xs text-muted-foreground leading-relaxed">
          <span className="font-medium text-amber-400/80">Disclaimer:</span>{" "}
          This risk monitor provides general information based on publicly
          available data, government regulations, and industry reports. It does
          not constitute legal, financial, or investment advice. Risk scores are
          indicative estimates. Always consult qualified Indonesian property
          lawyers, certified PPAT notaries, and licensed tax consultants before
          making any investment decisions. Data sources include Indonesian
          government agencies (BPN, BKPM, OJK), Transparency International,
          BNPB, BMKG, World Bank, and Bali provincial government publications.
        </p>
      </div>
    </div>
  );
}
