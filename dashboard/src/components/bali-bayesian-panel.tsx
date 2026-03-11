"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getBayesianDetails,
  combinedRegulatoryPosterior,
} from "@/lib/bali-risk-engine";

function ProbBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="w-full h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all"
        style={{ width: `${value * 100}%`, backgroundColor: color }}
      />
    </div>
  );
}

export function BaliBayesianPanel() {
  const details = getBayesianDetails();
  const combined = combinedRegulatoryPosterior();

  return (
    <Card className="bg-[#111118] border-[#1e1e2e]">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          Bayesian Regulatory Risk Model
          <span className="text-[10px] bg-indigo-500/10 text-indigo-400 px-2 py-0.5 rounded-full font-normal">
            P(enforcement | signals)
          </span>
        </CardTitle>
        <p className="text-xs text-muted-foreground mt-1">
          Posterior probability of enforcement action given observed signals.
          Uses Bayes&apos; theorem: P(E|S) = P(S|E)&middot;P(E) / [P(S|E)&middot;P(E) + P(S|&not;E)&middot;P(&not;E)]
        </p>
      </CardHeader>
      <CardContent>
        {/* Combined score */}
        <div className="rounded-lg border border-indigo-500/20 bg-indigo-500/5 p-3 mb-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-mono">
              Combined Regulatory Posterior
            </span>
            <span className="text-lg font-bold font-mono text-indigo-400">
              {(combined * 100).toFixed(1)}%
            </span>
          </div>
          <ProbBar value={combined} color="#6366f1" />
          <p className="text-[10px] text-slate-500 mt-1.5">
            Weighted average of all regulatory enforcement posteriors.
            Higher = more likely enforcement action will affect investments.
          </p>
        </div>

        {/* Individual regulations */}
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-[#1e1e2e]">
                <th className="text-left py-2 font-medium">Regulation</th>
                <th className="text-right py-2 font-medium">
                  Prior P(E)
                </th>
                <th className="text-right py-2 font-medium">
                  Likelihood P(S|E)
                </th>
                <th className="text-right py-2 font-medium">
                  Posterior P(E|S)
                </th>
                <th className="py-2 w-24"></th>
              </tr>
            </thead>
            <tbody>
              {details.map((d) => {
                const color =
                  d.posterior >= 0.7
                    ? "#ef4444"
                    : d.posterior >= 0.5
                      ? "#f97316"
                      : d.posterior >= 0.3
                        ? "#eab308"
                        : "#22c55e";
                return (
                  <tr
                    key={d.key}
                    className="border-b border-[#1e1e2e]/50 hover:bg-[#1a1a2e] transition-colors"
                  >
                    <td className="py-2.5 text-slate-300 font-mono">
                      {d.label}
                    </td>
                    <td className="py-2.5 text-right text-slate-500 font-mono">
                      {(d.prior * 100).toFixed(0)}%
                    </td>
                    <td className="py-2.5 text-right text-slate-500 font-mono">
                      {(d.likelihood * 100).toFixed(0)}%
                    </td>
                    <td className="py-2.5 text-right font-mono font-bold" style={{ color }}>
                      {(d.posterior * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 pl-3">
                      <ProbBar value={d.posterior} color={color} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Formula explanation */}
        <div className="mt-4 pt-3 border-t border-[#1e1e2e]">
          <p className="text-[10px] text-slate-600 font-mono leading-relaxed">
            Model adapted from network/signals/market_signal.py probability blending.
            Priors calibrated from Indonesian government enforcement data (BPN, BKPM, OSS).
            Likelihoods estimated from signal strength (PP 28/2025 implementation,
            media coverage, recent enforcement actions). Updated 2026-03-10.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
