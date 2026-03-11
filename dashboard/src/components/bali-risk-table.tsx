"use client";

import { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  ArrowUp,
  ArrowDown,
  Minus,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import type { AnyRisk, RiskSeverity, RiskTrend } from "@/types/bali-risk";
import { SEVERITY_CONFIG, TREND_CONFIG } from "@/types/bali-risk";

function SeverityBadge({ severity }: { severity: RiskSeverity }) {
  const cfg = SEVERITY_CONFIG[severity];
  return (
    <Badge
      variant="secondary"
      className={`text-xs border-0 ${cfg.bgColor} ${cfg.color}`}
    >
      {cfg.label}
    </Badge>
  );
}

function TrendIndicator({ trend }: { trend: RiskTrend }) {
  const cfg = TREND_CONFIG[trend];
  return (
    <div className={`flex items-center gap-1 ${cfg.color}`}>
      {trend === "worsening" ? (
        <ArrowUp className="h-3 w-3" />
      ) : trend === "improving" ? (
        <ArrowDown className="h-3 w-3" />
      ) : (
        <Minus className="h-3 w-3" />
      )}
      <span className="text-xs">{cfg.label}</span>
    </div>
  );
}

function ScoreBar({ score }: { score: number }) {
  const getColor = (s: number) => {
    if (s >= 80) return "bg-red-500";
    if (s >= 60) return "bg-orange-500";
    if (s >= 40) return "bg-amber-500";
    return "bg-emerald-500";
  };

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${getColor(score)}`}
          style={{ width: `${score}%` }}
        />
      </div>
      <span className="text-xs font-mono">{score}</span>
    </div>
  );
}

interface ExpandedRowProps {
  risk: AnyRisk;
}

function ExpandedRow({ risk }: ExpandedRowProps) {
  return (
    <TableRow className="border-[#1e1e2e] bg-[#0d0d14]">
      <TableCell colSpan={6} className="py-4 px-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-1">
              Details
            </h4>
            <p className="text-sm leading-relaxed">{risk.details}</p>
          </div>
          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-1">
              Mitigation
            </h4>
            <p className="text-sm leading-relaxed text-emerald-300/80">
              {risk.mitigation}
            </p>
          </div>
          <div className="md:col-span-2 flex flex-wrap gap-4 text-xs text-muted-foreground pt-2 border-t border-[#1e1e2e]">
            <span>
              Source: <span className="text-foreground/70">{risk.source}</span>
            </span>
            <span>
              Updated:{" "}
              <span className="text-foreground/70">{risk.last_updated}</span>
            </span>
            {risk.category === "legal" && (
              <>
                <span>
                  Legal Area:{" "}
                  <span className="text-foreground/70">
                    {(risk as any).legal_area}
                  </span>
                </span>
                <span>
                  Precedent Cases:{" "}
                  <span className="text-foreground/70">
                    {(risk as any).precedent_cases}
                  </span>
                </span>
                <span>
                  Affected:{" "}
                  <span className="text-foreground/70">
                    {(risk as any).affected_structures?.join(", ")}
                  </span>
                </span>
              </>
            )}
            {risk.category === "climate" && (
              <>
                <span>
                  10yr Probability:{" "}
                  <span className="text-foreground/70">
                    {((risk as any).probability_10yr * 100).toFixed(0)}%
                  </span>
                </span>
                <span>
                  Est. Damage:{" "}
                  <span className="text-foreground/70">
                    {(risk as any).estimated_damage_pct}% of value
                  </span>
                </span>
                <span>
                  Regions:{" "}
                  <span className="text-foreground/70">
                    {(risk as any).affected_regions?.join(", ")}
                  </span>
                </span>
              </>
            )}
            {risk.category === "political" && (
              <>
                <span>
                  Likelihood:{" "}
                  <span className="text-foreground/70">
                    {((risk as any).likelihood * 100).toFixed(0)}%
                  </span>
                </span>
                <span>
                  Type:{" "}
                  <span className="text-foreground/70">
                    {(risk as any).risk_type}
                  </span>
                </span>
              </>
            )}
            {risk.category === "financial" && (
              <>
                <span>
                  {(risk as any).metric_label}:{" "}
                  <span className="text-foreground/70">
                    {(risk as any).metric_value}
                  </span>
                </span>
              </>
            )}
            {risk.category === "regulatory" && (
              <>
                <span>
                  Regulation:{" "}
                  <span className="text-foreground/70">
                    {(risk as any).regulation}
                  </span>
                </span>
                <span>
                  Deadline:{" "}
                  <span className="text-foreground/70">
                    {(risk as any).compliance_deadline}
                  </span>
                </span>
                <span>
                  Penalty:{" "}
                  <span className="text-red-400/80">
                    {(risk as any).penalty}
                  </span>
                </span>
              </>
            )}
            {risk.category === "market" && (
              <>
                <span>
                  {(risk as any).metric_label}:{" "}
                  <span className="text-foreground/70">
                    {(risk as any).metric_value}
                  </span>
                </span>
              </>
            )}
          </div>
        </div>
      </TableCell>
    </TableRow>
  );
}

interface BaliRiskTableProps {
  data: AnyRisk[];
  categoryLabel: string;
}

export function BaliRiskTable({ data, categoryLabel }: BaliRiskTableProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No {categoryLabel.toLowerCase()} risks identified
      </div>
    );
  }

  const sorted = [...data].sort((a, b) => b.score - a.score);

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="border-[#1e1e2e] hover:bg-transparent">
            <TableHead className="text-muted-foreground w-8"></TableHead>
            <TableHead className="text-muted-foreground">Risk</TableHead>
            <TableHead className="text-muted-foreground">Severity</TableHead>
            <TableHead className="text-muted-foreground">Score</TableHead>
            <TableHead className="text-muted-foreground">Trend</TableHead>
            <TableHead className="text-muted-foreground">Description</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((risk) => (
            <>
              <TableRow
                key={risk.id}
                className="border-[#1e1e2e] cursor-pointer hover:bg-[#1a1a2e] transition-colors"
                onClick={() =>
                  setExpandedId(expandedId === risk.id ? null : risk.id)
                }
              >
                <TableCell className="w-8">
                  {expandedId === risk.id ? (
                    <ChevronUp className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  )}
                </TableCell>
                <TableCell>
                  <div className="font-medium text-sm">{risk.title}</div>
                </TableCell>
                <TableCell>
                  <SeverityBadge severity={risk.severity} />
                </TableCell>
                <TableCell>
                  <ScoreBar score={risk.score} />
                </TableCell>
                <TableCell>
                  <TrendIndicator trend={risk.trend} />
                </TableCell>
                <TableCell className="max-w-[300px]">
                  <div className="text-xs text-muted-foreground line-clamp-2">
                    {risk.description}
                  </div>
                </TableCell>
              </TableRow>
              {expandedId === risk.id && (
                <ExpandedRow key={`${risk.id}-expanded`} risk={risk} />
              )}
            </>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
