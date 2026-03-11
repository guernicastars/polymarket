import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BaliRiskStatsCards } from "@/components/bali-risk-stats-cards";
import { BaliRiskTabs } from "@/components/bali-risk-tabs";
import { BaliRiskHeatmap } from "@/components/bali-risk-heatmap";
import { BaliRiskMap } from "@/components/bali-risk-map";
import { BaliRiskNews } from "@/components/bali-risk-news";
import { BaliBayesianPanel } from "@/components/bali-bayesian-panel";
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
          <span className="flex items-center gap-1.5 text-[10px] bg-emerald-500/10 text-emerald-400 px-2.5 py-0.5 rounded-full font-mono">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            LIVE INTEL
          </span>
        </div>
        <p className="text-muted-foreground text-sm mt-1 max-w-3xl">
          Comprehensive risk intelligence for foreign real estate investment in
          Bali, Indonesia. Bayesian risk modeling with 7-component vulnerability scoring,
          cascade analysis, and regulatory enforcement probability estimation.
        </p>
      </div>

      {/* Stats Cards */}
      <BaliRiskStatsCards overview={overview} />

      {/* Interactive Map */}
      <BaliRiskMap />

      {/* Two-column: Heatmap + News */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <BaliRiskHeatmap risksByCategory={risksByCategory} />
        </div>
        <Card className="bg-[#111118] border-[#1e1e2e]">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Intelligence Feed</CardTitle>
          </CardHeader>
          <CardContent>
            <BaliRiskNews />
          </CardContent>
        </Card>
      </div>

      {/* Bayesian Model */}
      <BaliBayesianPanel />

      {/* Risk Detail Tabs */}
      <Card className="bg-[#111118] border-[#1e1e2e]">
        <CardHeader>
          <CardTitle className="text-base">
            Risk Indicators by Category
          </CardTitle>
          <p className="text-xs text-muted-foreground">
            Click any risk to expand details, sources, and mitigation strategies
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
          available data, government regulations, and industry reports. Risk scores
          are computed using Bayesian posterior estimation and 7-component vulnerability
          modeling adapted from conflict network analysis. It does not constitute legal,
          financial, or investment advice. Always consult qualified Indonesian property
          lawyers, certified PPAT notaries, and licensed tax consultants before
          making any investment decisions. Data sources include Indonesian
          government agencies (BPN, BKPM, OJK), Transparency International,
          BNPB, BMKG, World Bank, and Bali provincial government publications.
        </p>
      </div>
    </div>
  );
}
