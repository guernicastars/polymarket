"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ShieldAlert,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Eye,
  BarChart3,
} from "lucide-react";
import type {
  ManipulationAlertData,
  ManipulationSeverity,
} from "@/types/causal";

// ── Constants ─────────────────────────────────────────────

const SEVERITY_CONFIG: Record<
  ManipulationSeverity,
  { label: string; color: string; bgColor: string; borderColor: string }
> = {
  low: {
    label: "Low",
    color: "#6b7280",
    bgColor: "bg-[#6b7280]/10",
    borderColor: "border-[#6b7280]/20",
  },
  medium: {
    label: "Medium",
    color: "#f59e0b",
    bgColor: "bg-amber-400/10",
    borderColor: "border-amber-400/20",
  },
  high: {
    label: "High",
    color: "#ff4466",
    bgColor: "bg-[#ff4466]/10",
    borderColor: "border-[#ff4466]/20",
  },
  critical: {
    label: "Critical",
    color: "#ef4444",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/30",
  },
};

// ── Helpers ───────────────────────────────────────────────

function getSeverity(riskScore: number): ManipulationSeverity {
  if (riskScore >= 0.75) return "critical";
  if (riskScore >= 0.5) return "high";
  if (riskScore >= 0.25) return "medium";
  return "low";
}

function formatRelativeTime(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffSec = Math.floor((now - then) / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return `${Math.floor(diffHr / 24)}d ago`;
}

// ── Props ─────────────────────────────────────────────────

interface ManipulationAlertProps {
  alerts: ManipulationAlertData[];
}

// ── Component ─────────────────────────────────────────────

export function ManipulationAlert({ alerts }: ManipulationAlertProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (!alerts || alerts.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <ShieldAlert className="h-8 w-8 mx-auto mb-3 opacity-40" />
        <p>No manipulation alerts detected</p>
        <p className="text-xs mt-1">Markets are operating within normal parameters</p>
      </div>
    );
  }

  // Sort by risk score descending
  const sorted = [...alerts].sort((a, b) => b.riskScore - a.riskScore);

  // Summary stats
  const criticalCount = sorted.filter((a) => getSeverity(a.riskScore) === "critical").length;
  const highCount = sorted.filter((a) => getSeverity(a.riskScore) === "high").length;
  const avgRisk = sorted.reduce((sum, a) => sum + a.riskScore, 0) / sorted.length;

  return (
    <div className="space-y-4">
      {/* Summary bar */}
      <div className="flex items-center gap-4 text-xs">
        <span className="text-muted-foreground">
          {sorted.length} market{sorted.length !== 1 ? "s" : ""} flagged
        </span>
        {criticalCount > 0 && (
          <Badge variant="secondary" className="text-xs border-0 bg-red-500/10 text-red-400">
            {criticalCount} Critical
          </Badge>
        )}
        {highCount > 0 && (
          <Badge variant="secondary" className="text-xs border-0 bg-[#ff4466]/10 text-[#ff4466]">
            {highCount} High
          </Badge>
        )}
        <span className="text-muted-foreground ml-auto font-mono">
          Avg risk: {(avgRisk * 100).toFixed(0)}%
        </span>
      </div>

      {/* Alert cards */}
      <div className="space-y-2">
        {sorted.map((alert) => {
          const severity = getSeverity(alert.riskScore);
          const config = SEVERITY_CONFIG[severity];
          const isExpanded = expandedId === alert.marketId;

          return (
            <Card
              key={alert.marketId}
              className={`bg-[#111118] border-[#1e1e2e] ${
                severity === "critical" ? "border-red-500/30" : ""
              }`}
            >
              <CardContent className="pt-4 pb-3 px-4">
                {/* Main row */}
                <div
                  className="flex items-center gap-3 cursor-pointer"
                  onClick={() =>
                    setExpandedId(isExpanded ? null : alert.marketId)
                  }
                >
                  {/* Severity icon */}
                  <div
                    className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `${config.color}15` }}
                  >
                    {severity === "critical" || severity === "high" ? (
                      <AlertTriangle
                        className="h-4 w-4"
                        style={{ color: config.color }}
                      />
                    ) : (
                      <Eye
                        className="h-4 w-4"
                        style={{ color: config.color }}
                      />
                    )}
                  </div>

                  {/* Market info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm truncate max-w-[300px]">
                        {alert.question}
                      </span>
                      <Badge
                        variant="secondary"
                        className={`text-xs border-0 ${config.bgColor}`}
                        style={{ color: config.color }}
                      >
                        {config.label}
                      </Badge>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {alert.details.length > 0
                        ? alert.details[0]
                        : "Suspicious activity detected"}
                    </p>
                  </div>

                  {/* Score bars */}
                  <div className="flex items-center gap-4 flex-shrink-0">
                    <ScoreMini label="Risk" value={alert.riskScore} color={config.color} />
                    <ScoreMini label="Wash" value={alert.washScore} />
                    <ScoreMini label="Spoof" value={alert.spoofScore} />
                    <ScoreMini label="Anomaly" value={alert.anomalyScore} />
                  </div>

                  {/* Timestamp + expand */}
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className="text-xs text-muted-foreground font-mono whitespace-nowrap">
                      {formatRelativeTime(alert.detectedAt)}
                    </span>
                    {isExpanded ? (
                      <ChevronUp className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    )}
                  </div>
                </div>

                {/* Expanded details */}
                {isExpanded && (
                  <div className="mt-4 pt-3 border-t border-[#1e1e2e] space-y-3">
                    {/* Score breakdown */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <ScoreDetail
                        label="Overall Risk"
                        value={alert.riskScore}
                        icon={<ShieldAlert className="h-3.5 w-3.5" />}
                        color={config.color}
                      />
                      <ScoreDetail
                        label="Wash Trading"
                        value={alert.washScore}
                        icon={<BarChart3 className="h-3.5 w-3.5" />}
                        description="Self-trading + round sizes + volume without impact"
                      />
                      <ScoreDetail
                        label="Spoofing"
                        value={alert.spoofScore}
                        icon={<Eye className="h-3.5 w-3.5" />}
                        description="Disappearing liquidity + asymmetric book patterns"
                      />
                      <ScoreDetail
                        label="Causal Anomaly"
                        value={alert.anomalyScore}
                        icon={<AlertTriangle className="h-3.5 w-3.5" />}
                        description="Price movements unexplained by causal model"
                      />
                    </div>

                    {/* Signal details */}
                    {alert.details.length > 0 && (
                      <div>
                        <p className="text-xs text-muted-foreground font-medium mb-1.5">
                          Detection Signals
                        </p>
                        <ul className="space-y-1">
                          {alert.details.map((detail, i) => (
                            <li
                              key={i}
                              className="flex items-start gap-2 text-xs text-muted-foreground"
                            >
                              <span className="text-amber-400 mt-0.5">\u2022</span>
                              {detail}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Market ID */}
                    <p className="text-xs text-muted-foreground font-mono">
                      Market: {alert.marketId}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}

// ── Score mini bar (inline) ───────────────────────────────

function ScoreMini({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color?: string;
}) {
  const barColor = color ?? getScoreColor(value);
  return (
    <div className="min-w-[60px]">
      <span className="text-xs text-muted-foreground">{label}</span>
      <div className="flex items-center gap-1.5 mt-0.5">
        <div className="flex-1 h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${Math.max(value * 100, 2)}%`,
              backgroundColor: barColor,
            }}
          />
        </div>
        <span className="font-mono text-xs w-7 text-right" style={{ color: barColor }}>
          {(value * 100).toFixed(0)}
        </span>
      </div>
    </div>
  );
}

// ── Score detail card ─────────────────────────────────────

function ScoreDetail({
  label,
  value,
  icon,
  color,
  description,
}: {
  label: string;
  value: number;
  icon: React.ReactNode;
  color?: string;
  description?: string;
}) {
  const barColor = color ?? getScoreColor(value);
  return (
    <div className="bg-[#0d0d14] border border-[#1e1e2e] rounded-lg px-3 py-2.5">
      <div className="flex items-center gap-1.5">
        <span style={{ color: barColor }}>{icon}</span>
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <div className="flex items-center gap-2 mt-1.5">
        <div className="flex-1 h-2 bg-[#1e1e2e] rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${Math.max(value * 100, 2)}%`,
              backgroundColor: barColor,
            }}
          />
        </div>
        <span className="font-mono text-sm font-medium w-10 text-right" style={{ color: barColor }}>
          {(value * 100).toFixed(0)}%
        </span>
      </div>
      {description && (
        <p className="text-xs text-muted-foreground mt-1">{description}</p>
      )}
    </div>
  );
}

// ── Score color helper ────────────────────────────────────

function getScoreColor(value: number): string {
  if (value >= 0.75) return "#ef4444";
  if (value >= 0.5) return "#ff4466";
  if (value >= 0.25) return "#f59e0b";
  return "#6b7280";
}

// ── Skeleton ──────────────────────────────────────────────

export function ManipulationAlertSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 4 }).map((_, i) => (
        <Card key={i} className="bg-[#111118] border-[#1e1e2e]">
          <CardContent className="pt-4 pb-3 px-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-[#1e1e2e] animate-pulse" />
              <div className="flex-1">
                <div className="h-4 w-48 bg-[#1e1e2e] rounded animate-pulse" />
                <div className="h-3 w-64 bg-[#1e1e2e] rounded animate-pulse mt-1.5" />
              </div>
              <div className="flex gap-4">
                {Array.from({ length: 4 }).map((_, j) => (
                  <div key={j} className="w-16">
                    <div className="h-2 w-12 bg-[#1e1e2e] rounded animate-pulse" />
                    <div className="h-1.5 w-full bg-[#1e1e2e] rounded-full animate-pulse mt-1" />
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
