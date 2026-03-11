"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Scale,
  CloudRain,
  Landmark,
  DollarSign,
  FileText,
  TrendingDown,
} from "lucide-react";
import { BaliRiskTable } from "./bali-risk-table";
import type { AnyRisk, RiskCategory } from "@/types/bali-risk";
import { CATEGORY_CONFIG } from "@/types/bali-risk";

interface BaliRiskTabsProps {
  risksByCategory: Record<RiskCategory, AnyRisk[]>;
}

const TAB_CONFIG: {
  value: RiskCategory;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  { value: "legal", icon: Scale },
  { value: "climate", icon: CloudRain },
  { value: "political", icon: Landmark },
  { value: "financial", icon: DollarSign },
  { value: "regulatory", icon: FileText },
  { value: "market", icon: TrendingDown },
];

export function BaliRiskTabs({ risksByCategory }: BaliRiskTabsProps) {
  return (
    <Tabs defaultValue="legal">
      <TabsList>
        {TAB_CONFIG.map(({ value, icon: Icon }) => {
          const cfg = CATEGORY_CONFIG[value];
          const count = risksByCategory[value]?.length ?? 0;
          return (
            <TabsTrigger key={value} value={value}>
              <Icon className={`h-3.5 w-3.5 mr-1 ${cfg.color}`} />
              {cfg.label} ({count})
            </TabsTrigger>
          );
        })}
      </TabsList>
      {TAB_CONFIG.map(({ value }) => {
        const cfg = CATEGORY_CONFIG[value];
        return (
          <TabsContent key={value} value={value}>
            <div className="mb-3 px-1">
              <p className="text-xs text-muted-foreground">{cfg.description}</p>
            </div>
            <BaliRiskTable
              data={risksByCategory[value]}
              categoryLabel={cfg.label}
            />
          </TabsContent>
        );
      })}
    </Tabs>
  );
}
