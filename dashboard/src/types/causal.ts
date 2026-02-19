// ── Causal Analysis Types ─────────────────────────────────
// Types aligned with the pipeline/causal/ backend modules.

// ── CausalGraph types ─────────────────────────────────────

export type CausalEdgeMethod = "granger" | "pc" | "transfer_entropy";

export interface CausalNode {
  id: string;
  label: string;
  /** Optional: market question for tooltips */
  question?: string;
  /** Optional: current price 0-1 */
  price?: number;
  /** Node role from information flow analysis */
  role?: "source" | "derivative" | "neutral";
  /** Simulation position x */
  x?: number;
  /** Simulation position y */
  y?: number;
  /** Velocity x for simulation */
  vx?: number;
  /** Velocity y for simulation */
  vy?: number;
}

export interface CausalEdge {
  source: string;
  target: string;
  /** Discovery method: granger, pc, or transfer_entropy */
  method: CausalEdgeMethod;
  /** Strength of the causal relationship (normalized 0-1) */
  strength: number;
  /** Direction type: directed or bidirectional */
  type: "directed" | "undirected" | "bidirectional";
  /** Optional: p-value for statistical significance */
  pValue?: number;
  /** Optional: F-statistic (Granger) */
  fStat?: number;
  /** Optional: lag (Granger) */
  lag?: number;
  /** Optional: confidence from merged evidence (0-1) */
  confidence?: number;
}

// ── ImpactAnalysis types ──────────────────────────────────

export interface ImpactResult {
  /** Average post-period causal impact */
  pointEffect: number;
  /** Sum of point-wise impacts */
  cumulativeEffect: number;
  /** pointEffect / mean pre-period price */
  relativeEffect: number;
  /** 95% CI lower bound */
  ciLower: number;
  /** 95% CI upper bound */
  ciUpper: number;
  /** Two-sided p-value */
  pValue: number;
  /** Whether the impact is statistically significant */
  significant: boolean;
  /** R-squared of the pre-period model */
  prePeriodR2: number;
  /** Actual prices in the post-period */
  actualPost: number[];
  /** Counterfactual prices in the post-period */
  counterfactualPost: number[];
  /** Point-wise impact (actual - counterfactual) */
  impactSeries: number[];
}

export interface PricePoint {
  timestamp: string;
  price: number;
}

// ── InformationFlow types ─────────────────────────────────

export interface FlowMatrix {
  /** Ordered token IDs */
  tokenIds: string[];
  /** NxN transfer entropy matrix: te[i][j] = TE from i to j */
  teMatrix: number[][];
  /** Maximum transfer entropy observed */
  maxTe: number;
}

export interface MarketFlowInfo {
  tokenId: string;
  label: string;
  totalOutflow: number;
  totalInflow: number;
  netFlow: number;
  role: "source" | "derivative";
}

// ── ManipulationAlert types ───────────────────────────────

export type ManipulationSeverity = "low" | "medium" | "high" | "critical";

export interface ManipulationAlertData {
  /** Market token ID or condition ID */
  marketId: string;
  /** Market question / label */
  question: string;
  /** Aggregate risk score 0-1 */
  riskScore: number;
  /** Wash trading composite score 0-1 */
  washScore: number;
  /** Spoofing composite score 0-1 */
  spoofScore: number;
  /** Causal anomaly score 0-1 */
  anomalyScore: number;
  /** Human-readable signal descriptions */
  details: string[];
  /** When the detection was last run */
  detectedAt: string;
}

// ── CounterfactualView types ──────────────────────────────

export interface SyntheticControlResult {
  /** Donor market weights */
  weights: Record<string, number>;
  /** RMSE of synthetic control in pre-period */
  preFitRmse: number;
  /** Actual treatment prices post-event */
  actualPost: number[];
  /** Synthetic control prices post-event */
  counterfactualPost: number[];
  /** Point-wise actual - counterfactual */
  impactSeries: number[];
  /** Mean of impact series */
  averageEffect: number;
  /** Sum of impact series */
  cumulativeEffect: number;
  /** Timestamps for post-period */
  timestampsPost: string[];
}

export interface CounterfactualEvent {
  id: string;
  label: string;
  timestamp: string;
  description?: string;
}

// ── CausalDashboard types ─────────────────────────────────

export interface CausalMarketOption {
  conditionId: string;
  question: string;
  tokenId?: string;
}

export interface CausalOverview {
  totalEdges: number;
  significantPairs: number;
  sourceMarkets: number;
  derivativeMarkets: number;
  alertCount: number;
  avgRiskScore: number;
}
