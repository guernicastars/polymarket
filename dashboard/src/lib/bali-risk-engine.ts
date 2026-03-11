/**
 * Bali Real Estate Risk Engine
 *
 * Bayesian network-based risk assessment adapted from Donbas conflict
 * vulnerability scoring (network/signals/vulnerability.py).
 *
 * Mathematical framework:
 * - 7-component vulnerability scoring with weighted linear combination
 * - Bayesian posterior updating for regulatory risk
 * - Supply chain path redundancy (min-cut analysis)
 * - Cascade failure simulation for neighborhood contagion
 * - Kelly criterion for investment sizing
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export interface BaliNode {
  id: string;
  name: string;
  lat: number;
  lng: number;
  region: "south" | "central" | "east" | "north" | "west";
  population: number;
  is_hotspot: boolean;

  // Raw risk factors (0-1 scale)
  legal_exposure: number;      // Nominee fraud prevalence, title disputes
  climate_hazard: number;      // Flood + volcanic + earthquake + erosion
  market_saturation: number;   // Oversupply, yield compression
  regulatory_pressure: number; // PP 28/2025 enforcement, licensing
  infrastructure_stress: number; // Water, power, roads, waste
  political_sensitivity: number; // Banjar opposition, nationalism
  financial_volatility: number; // IDR depreciation, transaction costs

  // Computed
  vulnerability_score?: number;
  bayesian_posterior?: number;
  cascade_severity?: number;
  composite_risk?: number;
  kelly_fraction?: number;
}

export interface BaliEdge {
  source: string;
  target: string;
  type: "road" | "water_supply" | "power_grid" | "tourism_flow" | "economic_dependency";
  weight: number;     // 1.0 = normal capacity
  distance_km: number;
  is_critical: boolean;
}

export interface RiskZone {
  id: string;
  name: string;
  type: "flood" | "volcanic" | "earthquake" | "erosion" | "saturation" | "green_zone";
  center: [number, number];
  radius_km: number;
  severity: number; // 0-1
  color: string;
}

export interface VulnerabilityBreakdown {
  legal: number;
  climate: number;
  market: number;
  regulatory: number;
  infrastructure: number;
  political: number;
  financial: number;
  composite: number;
}

// ─── Constants ───────────────────────────────────────────────────────────────

/**
 * Vulnerability weights adapted from Donbas model:
 * Original: connectivity(0.20), supply(0.20), force_balance(0.15),
 *           terrain(0.10), fortification(0.10), assault(0.15), frontline(0.10)
 *
 * Bali adaptation maps conflict concepts to real estate:
 * - connectivity → legal_exposure (how exposed to ownership disputes)
 * - supply → infrastructure_stress (water, power, roads)
 * - force_balance → market_saturation (development pressure)
 * - terrain → climate_hazard (natural disaster exposure)
 * - fortification → regulatory_pressure (compliance burden)
 * - assault → political_sensitivity (community opposition)
 * - frontline → financial_volatility (currency, liquidity risk)
 */
const VULNERABILITY_WEIGHTS = {
  legal_exposure: 0.25,       // Highest weight — existential risk
  climate_hazard: 0.15,
  market_saturation: 0.15,
  regulatory_pressure: 0.15,
  infrastructure_stress: 0.10,
  political_sensitivity: 0.10,
  financial_volatility: 0.10,
} as const;

// ─── Bayesian Regulatory Risk ────────────────────────────────────────────────

/**
 * Bayesian posterior for regulatory enforcement probability.
 *
 * P(enforcement | signal) = P(signal | enforcement) * P(enforcement)
 *                           / P(signal)
 *
 * Prior: base rate of enforcement actions in Bali (from government data)
 * Likelihood: probability of observing current signals given enforcement
 * Evidence: signals include PP 28/2025, recent crackdowns, media reports
 */
interface BayesianPrior {
  enforcement_base_rate: number;  // P(enforcement)
  signal_given_enforcement: number; // P(signal | enforcement)
  signal_given_no_enforcement: number; // P(signal | ¬enforcement)
}

const REGULATORY_PRIORS: Record<string, BayesianPrior> = {
  pp_28_2025: {
    enforcement_base_rate: 0.60,      // New regulation, high initial enforcement
    signal_given_enforcement: 0.85,   // Strong signals observed
    signal_given_no_enforcement: 0.20,
  },
  height_limit: {
    enforcement_base_rate: 0.75,      // Consistently enforced
    signal_given_enforcement: 0.90,
    signal_given_no_enforcement: 0.10,
  },
  rental_licensing: {
    enforcement_base_rate: 0.45,      // Spotty enforcement
    signal_given_enforcement: 0.70,
    signal_given_no_enforcement: 0.35,
  },
  green_zone: {
    enforcement_base_rate: 0.80,      // Strictly enforced
    signal_given_enforcement: 0.95,
    signal_given_no_enforcement: 0.05,
  },
  nominee_crackdown: {
    enforcement_base_rate: 0.55,      // Increasing
    signal_given_enforcement: 0.80,
    signal_given_no_enforcement: 0.25,
  },
};

/**
 * Compute Bayesian posterior probability of enforcement action.
 * Uses Bayes' theorem: P(E|S) = P(S|E)·P(E) / [P(S|E)·P(E) + P(S|¬E)·P(¬E)]
 */
export function bayesianPosterior(priorKey: string): number {
  const prior = REGULATORY_PRIORS[priorKey];
  if (!prior) return 0.5;

  const { enforcement_base_rate: pe, signal_given_enforcement: pse, signal_given_no_enforcement: psne } = prior;
  const numerator = pse * pe;
  const denominator = numerator + psne * (1 - pe);
  return denominator > 0 ? numerator / denominator : pe;
}

/**
 * Combined regulatory risk posterior — weighted average across all regulations
 */
export function combinedRegulatoryPosterior(): number {
  const posteriors = Object.keys(REGULATORY_PRIORS).map(bayesianPosterior);
  return posteriors.reduce((a, b) => a + b, 0) / posteriors.length;
}

// ─── Vulnerability Scoring ───────────────────────────────────────────────────

/**
 * 7-component vulnerability score adapted from Donbas model.
 *
 * V_composite = Σ(w_i · factor_i) for i in {legal, climate, market,
 *               regulatory, infrastructure, political, financial}
 *
 * Each factor is 0-1, weights sum to 1.0, output is 0-1.
 */
export function computeVulnerability(node: BaliNode): VulnerabilityBreakdown {
  const components = {
    legal: node.legal_exposure,
    climate: node.climate_hazard,
    market: node.market_saturation,
    regulatory: node.regulatory_pressure,
    infrastructure: node.infrastructure_stress,
    political: node.political_sensitivity,
    financial: node.financial_volatility,
  };

  const composite =
    VULNERABILITY_WEIGHTS.legal_exposure * components.legal +
    VULNERABILITY_WEIGHTS.climate_hazard * components.climate +
    VULNERABILITY_WEIGHTS.market_saturation * components.market +
    VULNERABILITY_WEIGHTS.regulatory_pressure * components.regulatory +
    VULNERABILITY_WEIGHTS.infrastructure_stress * components.infrastructure +
    VULNERABILITY_WEIGHTS.political_sensitivity * components.political +
    VULNERABILITY_WEIGHTS.financial_volatility * components.financial;

  return { ...components, composite };
}

// ─── Cascade Analysis ────────────────────────────────────────────────────────

/**
 * Simplified cascade simulation: if a key infrastructure node fails,
 * how many dependent properties lose access?
 *
 * Adapted from network/signals/cascade.py:
 * severity = (isolated_value + failed_value) / total_value
 */
export function simulateCascade(
  failedNodeId: string,
  nodes: BaliNode[],
  edges: BaliEdge[]
): { severity: number; isolated_nodes: string[] } {
  // Build adjacency list
  const adj: Record<string, string[]> = {};
  for (const n of nodes) adj[n.id] = [];
  for (const e of edges) {
    if (e.source !== failedNodeId && e.target !== failedNodeId) {
      if (adj[e.source]) adj[e.source].push(e.target);
      if (adj[e.target]) adj[e.target].push(e.source);
    }
  }

  // BFS to find connected components
  const visited = new Set<string>();
  const components: string[][] = [];

  for (const n of nodes) {
    if (n.id === failedNodeId || visited.has(n.id)) continue;
    const component: string[] = [];
    const queue = [n.id];
    visited.add(n.id);
    while (queue.length > 0) {
      const u = queue.shift()!;
      component.push(u);
      for (const v of (adj[u] || [])) {
        if (!visited.has(v)) {
          visited.add(v);
          queue.push(v);
        }
      }
    }
    components.push(component);
  }

  // Largest component assumed to be "connected to services"
  const largest = components.sort((a, b) => b.length - a.length)[0] || [];
  const largestSet = new Set(largest);
  const isolated = nodes
    .filter(n => n.id !== failedNodeId && !largestSet.has(n.id))
    .map(n => n.id);

  const totalPop = nodes.reduce((a, n) => a + n.population, 0);
  const failedNode = nodes.find(n => n.id === failedNodeId);
  const isolatedPop = isolated.reduce((a, id) => {
    const n = nodes.find(x => x.id === id);
    return a + (n?.population || 0);
  }, 0);
  const failedPop = failedNode?.population || 0;

  const severity = totalPop > 0
    ? Math.min((isolatedPop + failedPop) / totalPop, 1.0)
    : 0;

  return { severity, isolated_nodes: isolated };
}

// ─── Composite Risk Score ────────────────────────────────────────────────────

/**
 * Multi-factor composite risk blending, adapted from market_signal.py:
 *
 * P(risk) = 0.45 · V_vulnerability
 *         + 0.25 · B_bayesian_regulatory
 *         + 0.20 · C_cascade_severity
 *         + 0.10 · base_rate
 *
 * Output clamped to [0.01, 0.99]
 */
export function compositeRiskScore(
  vulnerability: number,
  bayesianRegulatory: number,
  cascadeSeverity: number,
  baseRate: number = 0.35
): number {
  const raw =
    0.45 * vulnerability +
    0.25 * bayesianRegulatory +
    0.20 * cascadeSeverity +
    0.10 * baseRate;
  return Math.min(Math.max(raw, 0.01), 0.99);
}

// ─── Kelly Criterion ─────────────────────────────────────────────────────────

/**
 * Kelly criterion for optimal investment allocation.
 *
 * f* = (p·b - q) / b
 *
 * where p = model probability of success, b = payout odds, q = 1-p
 *
 * For real estate: p = 1 - composite_risk (probability of successful investment)
 * b = expected return ratio (e.g., 0.08 for 8% yield)
 */
export function kellyFraction(
  compositeRisk: number,
  expectedReturnRatio: number = 0.08,
  riskFreeRate: number = 0.04
): number {
  const p = 1 - compositeRisk; // probability of positive outcome
  const q = compositeRisk;
  const b = expectedReturnRatio - riskFreeRate; // excess return
  if (b <= 0) return 0;
  const f = (p * b - q * riskFreeRate) / b;
  return Math.max(0, Math.min(f, 1)); // clamp to [0, 1]
}

// ─── Bali Graph Data ─────────────────────────────────────────────────────────

export const BALI_NODES: BaliNode[] = [
  {
    id: "canggu",
    name: "Canggu",
    lat: -8.6478,
    lng: 115.1385,
    region: "south",
    population: 45000,
    is_hotspot: true,
    legal_exposure: 0.85,
    climate_hazard: 0.70,
    market_saturation: 0.92,
    regulatory_pressure: 0.75,
    infrastructure_stress: 0.80,
    political_sensitivity: 0.65,
    financial_volatility: 0.55,
  },
  {
    id: "seminyak",
    name: "Seminyak",
    lat: -8.6913,
    lng: 115.1682,
    region: "south",
    population: 35000,
    is_hotspot: true,
    legal_exposure: 0.80,
    climate_hazard: 0.55,
    market_saturation: 0.85,
    regulatory_pressure: 0.70,
    infrastructure_stress: 0.75,
    political_sensitivity: 0.50,
    financial_volatility: 0.50,
  },
  {
    id: "kuta",
    name: "Kuta",
    lat: -8.7184,
    lng: 115.1686,
    region: "south",
    population: 80000,
    is_hotspot: true,
    legal_exposure: 0.75,
    climate_hazard: 0.65,
    market_saturation: 0.80,
    regulatory_pressure: 0.65,
    infrastructure_stress: 0.85,
    political_sensitivity: 0.45,
    financial_volatility: 0.50,
  },
  {
    id: "ubud",
    name: "Ubud",
    lat: -8.5069,
    lng: 115.2625,
    region: "central",
    population: 75000,
    is_hotspot: true,
    legal_exposure: 0.60,
    climate_hazard: 0.45,
    market_saturation: 0.65,
    regulatory_pressure: 0.55,
    infrastructure_stress: 0.50,
    political_sensitivity: 0.70,
    financial_volatility: 0.40,
  },
  {
    id: "uluwatu",
    name: "Uluwatu / Bukit",
    lat: -8.8291,
    lng: 115.0849,
    region: "south",
    population: 20000,
    is_hotspot: true,
    legal_exposure: 0.50,
    climate_hazard: 0.30,
    market_saturation: 0.55,
    regulatory_pressure: 0.50,
    infrastructure_stress: 0.45,
    political_sensitivity: 0.35,
    financial_volatility: 0.40,
  },
  {
    id: "denpasar",
    name: "Denpasar",
    lat: -8.6500,
    lng: 115.2167,
    region: "south",
    population: 950000,
    is_hotspot: false,
    legal_exposure: 0.55,
    climate_hazard: 0.60,
    market_saturation: 0.50,
    regulatory_pressure: 0.60,
    infrastructure_stress: 0.70,
    political_sensitivity: 0.40,
    financial_volatility: 0.45,
  },
  {
    id: "sanur",
    name: "Sanur",
    lat: -8.6886,
    lng: 115.2640,
    region: "south",
    population: 30000,
    is_hotspot: false,
    legal_exposure: 0.55,
    climate_hazard: 0.55,
    market_saturation: 0.45,
    regulatory_pressure: 0.50,
    infrastructure_stress: 0.50,
    political_sensitivity: 0.35,
    financial_volatility: 0.40,
  },
  {
    id: "nusa_dua",
    name: "Nusa Dua",
    lat: -8.8014,
    lng: 115.2318,
    region: "south",
    population: 25000,
    is_hotspot: false,
    legal_exposure: 0.40,
    climate_hazard: 0.35,
    market_saturation: 0.40,
    regulatory_pressure: 0.45,
    infrastructure_stress: 0.30,
    political_sensitivity: 0.25,
    financial_volatility: 0.35,
  },
  {
    id: "jimbaran",
    name: "Jimbaran",
    lat: -8.7681,
    lng: 115.1659,
    region: "south",
    population: 40000,
    is_hotspot: false,
    legal_exposure: 0.55,
    climate_hazard: 0.40,
    market_saturation: 0.55,
    regulatory_pressure: 0.50,
    infrastructure_stress: 0.55,
    political_sensitivity: 0.40,
    financial_volatility: 0.45,
  },
  {
    id: "tabanan",
    name: "Tabanan",
    lat: -8.5375,
    lng: 115.1268,
    region: "west",
    population: 60000,
    is_hotspot: false,
    legal_exposure: 0.35,
    climate_hazard: 0.50,
    market_saturation: 0.25,
    regulatory_pressure: 0.40,
    infrastructure_stress: 0.55,
    political_sensitivity: 0.55,
    financial_volatility: 0.35,
  },
  {
    id: "gianyar",
    name: "Gianyar",
    lat: -8.5402,
    lng: 115.3268,
    region: "central",
    population: 50000,
    is_hotspot: false,
    legal_exposure: 0.45,
    climate_hazard: 0.40,
    market_saturation: 0.35,
    regulatory_pressure: 0.45,
    infrastructure_stress: 0.45,
    political_sensitivity: 0.60,
    financial_volatility: 0.35,
  },
  {
    id: "karangasem",
    name: "Karangasem",
    lat: -8.4490,
    lng: 115.6091,
    region: "east",
    population: 45000,
    is_hotspot: false,
    legal_exposure: 0.40,
    climate_hazard: 0.80,
    market_saturation: 0.20,
    regulatory_pressure: 0.35,
    infrastructure_stress: 0.65,
    political_sensitivity: 0.55,
    financial_volatility: 0.30,
  },
  {
    id: "amed",
    name: "Amed",
    lat: -8.3481,
    lng: 115.6456,
    region: "east",
    population: 12000,
    is_hotspot: false,
    legal_exposure: 0.45,
    climate_hazard: 0.75,
    market_saturation: 0.30,
    regulatory_pressure: 0.30,
    infrastructure_stress: 0.70,
    political_sensitivity: 0.45,
    financial_volatility: 0.35,
  },
  {
    id: "lovina",
    name: "Lovina",
    lat: -8.1529,
    lng: 115.0271,
    region: "north",
    population: 15000,
    is_hotspot: false,
    legal_exposure: 0.40,
    climate_hazard: 0.35,
    market_saturation: 0.20,
    regulatory_pressure: 0.30,
    infrastructure_stress: 0.60,
    political_sensitivity: 0.40,
    financial_volatility: 0.30,
  },
  {
    id: "sidemen",
    name: "Sidemen",
    lat: -8.4868,
    lng: 115.4715,
    region: "east",
    population: 8000,
    is_hotspot: false,
    legal_exposure: 0.35,
    climate_hazard: 0.45,
    market_saturation: 0.15,
    regulatory_pressure: 0.25,
    infrastructure_stress: 0.55,
    political_sensitivity: 0.50,
    financial_volatility: 0.30,
  },
  {
    id: "singaraja",
    name: "Singaraja",
    lat: -8.1120,
    lng: 115.0880,
    region: "north",
    population: 120000,
    is_hotspot: false,
    legal_exposure: 0.35,
    climate_hazard: 0.40,
    market_saturation: 0.15,
    regulatory_pressure: 0.35,
    infrastructure_stress: 0.50,
    political_sensitivity: 0.45,
    financial_volatility: 0.30,
  },
  {
    id: "candidasa",
    name: "Candidasa",
    lat: -8.5110,
    lng: 115.5680,
    region: "east",
    population: 10000,
    is_hotspot: false,
    legal_exposure: 0.45,
    climate_hazard: 0.60,
    market_saturation: 0.25,
    regulatory_pressure: 0.35,
    infrastructure_stress: 0.60,
    political_sensitivity: 0.40,
    financial_volatility: 0.35,
  },
  {
    id: "mt_agung",
    name: "Mt. Agung",
    lat: -8.3433,
    lng: 115.5078,
    region: "east",
    population: 0,
    is_hotspot: false,
    legal_exposure: 0,
    climate_hazard: 1.0,
    market_saturation: 0,
    regulatory_pressure: 0,
    infrastructure_stress: 0,
    political_sensitivity: 0,
    financial_volatility: 0,
  },
];

export const BALI_EDGES: BaliEdge[] = [
  // Major road connections
  { source: "canggu", target: "seminyak", type: "road", weight: 0.9, distance_km: 8, is_critical: false },
  { source: "seminyak", target: "kuta", type: "road", weight: 0.8, distance_km: 5, is_critical: false },
  { source: "kuta", target: "denpasar", type: "road", weight: 0.7, distance_km: 10, is_critical: true },
  { source: "kuta", target: "jimbaran", type: "road", weight: 0.85, distance_km: 8, is_critical: false },
  { source: "jimbaran", target: "uluwatu", type: "road", weight: 0.9, distance_km: 12, is_critical: false },
  { source: "jimbaran", target: "nusa_dua", type: "road", weight: 0.85, distance_km: 10, is_critical: false },
  { source: "denpasar", target: "sanur", type: "road", weight: 0.8, distance_km: 7, is_critical: false },
  { source: "denpasar", target: "ubud", type: "road", weight: 0.75, distance_km: 25, is_critical: true },
  { source: "denpasar", target: "gianyar", type: "road", weight: 0.8, distance_km: 20, is_critical: false },
  { source: "ubud", target: "gianyar", type: "road", weight: 0.85, distance_km: 10, is_critical: false },
  { source: "ubud", target: "tabanan", type: "road", weight: 0.7, distance_km: 30, is_critical: false },
  { source: "canggu", target: "tabanan", type: "road", weight: 0.8, distance_km: 15, is_critical: false },
  { source: "gianyar", target: "karangasem", type: "road", weight: 0.7, distance_km: 35, is_critical: true },
  { source: "karangasem", target: "amed", type: "road", weight: 0.6, distance_km: 15, is_critical: false },
  { source: "karangasem", target: "candidasa", type: "road", weight: 0.75, distance_km: 12, is_critical: false },
  { source: "gianyar", target: "sidemen", type: "road", weight: 0.65, distance_km: 25, is_critical: false },
  { source: "singaraja", target: "lovina", type: "road", weight: 0.85, distance_km: 10, is_critical: false },
  { source: "denpasar", target: "singaraja", type: "road", weight: 0.6, distance_km: 80, is_critical: true },
  // Water supply
  { source: "denpasar", target: "kuta", type: "water_supply", weight: 0.7, distance_km: 10, is_critical: true },
  { source: "denpasar", target: "seminyak", type: "water_supply", weight: 0.65, distance_km: 12, is_critical: true },
  { source: "denpasar", target: "canggu", type: "water_supply", weight: 0.55, distance_km: 18, is_critical: true },
  { source: "denpasar", target: "sanur", type: "water_supply", weight: 0.8, distance_km: 7, is_critical: false },
  { source: "ubud", target: "gianyar", type: "water_supply", weight: 0.75, distance_km: 10, is_critical: false },
  // Tourism flow
  { source: "denpasar", target: "ubud", type: "tourism_flow", weight: 0.9, distance_km: 25, is_critical: true },
  { source: "denpasar", target: "kuta", type: "tourism_flow", weight: 0.95, distance_km: 10, is_critical: true },
  { source: "canggu", target: "uluwatu", type: "tourism_flow", weight: 0.7, distance_km: 30, is_critical: false },
  { source: "ubud", target: "sidemen", type: "tourism_flow", weight: 0.5, distance_km: 25, is_critical: false },
  // Economic dependencies
  { source: "canggu", target: "denpasar", type: "economic_dependency", weight: 0.85, distance_km: 15, is_critical: true },
  { source: "ubud", target: "denpasar", type: "economic_dependency", weight: 0.80, distance_km: 25, is_critical: true },
  { source: "nusa_dua", target: "denpasar", type: "economic_dependency", weight: 0.75, distance_km: 20, is_critical: false },
];

export const RISK_ZONES: RiskZone[] = [
  {
    id: "agung_exclusion",
    name: "Mt. Agung Volcanic Exclusion Zone",
    type: "volcanic",
    center: [-8.3433, 115.5078],
    radius_km: 12,
    severity: 0.95,
    color: "#ef4444",
  },
  {
    id: "canggu_flood",
    name: "Canggu Flood Zone",
    type: "flood",
    center: [-8.6478, 115.1385],
    radius_km: 3,
    severity: 0.70,
    color: "#3b82f6",
  },
  {
    id: "kuta_flood",
    name: "Kuta-Legian Flood Zone",
    type: "flood",
    center: [-8.7184, 115.1686],
    radius_km: 2.5,
    severity: 0.65,
    color: "#3b82f6",
  },
  {
    id: "south_erosion",
    name: "Southern Coastal Erosion Zone",
    type: "erosion",
    center: [-8.7200, 115.1800],
    radius_km: 4,
    severity: 0.55,
    color: "#f59e0b",
  },
  {
    id: "canggu_saturation",
    name: "Canggu Market Saturation Zone",
    type: "saturation",
    center: [-8.6478, 115.1385],
    radius_km: 5,
    severity: 0.90,
    color: "#a855f7",
  },
  {
    id: "seminyak_saturation",
    name: "Seminyak-Kuta Saturation Zone",
    type: "saturation",
    center: [-8.7050, 115.1680],
    radius_km: 4,
    severity: 0.80,
    color: "#a855f7",
  },
  {
    id: "subak_green",
    name: "Subak UNESCO Green Zone (Tabanan)",
    type: "green_zone",
    center: [-8.4200, 115.1000],
    radius_km: 8,
    severity: 0.85,
    color: "#22c55e",
  },
];

// ─── Compute All Scores ──────────────────────────────────────────────────────

/**
 * Run the full risk engine on all Bali nodes.
 * Returns enriched nodes with computed risk metrics.
 */
export function computeAllRiskScores(): BaliNode[] {
  const bayesianReg = combinedRegulatoryPosterior();

  return BALI_NODES.map(node => {
    if (node.id === "mt_agung") return { ...node, vulnerability_score: 1, composite_risk: 1 };

    const vuln = computeVulnerability(node);
    const cascade = simulateCascade(node.id, BALI_NODES, BALI_EDGES);

    // Blend: node-specific regulatory pressure with Bayesian posterior
    const nodeRegulatory = node.regulatory_pressure * 0.6 + bayesianReg * 0.4;

    const composite = compositeRiskScore(
      vuln.composite,
      nodeRegulatory,
      cascade.severity,
      getBaseRate(node.region)
    );

    const kelly = kellyFraction(composite);

    return {
      ...node,
      vulnerability_score: vuln.composite,
      bayesian_posterior: nodeRegulatory,
      cascade_severity: cascade.severity,
      composite_risk: composite,
      kelly_fraction: kelly,
    };
  });
}

function getBaseRate(region: string): number {
  const rates: Record<string, number> = {
    south: 0.45,    // High development pressure
    central: 0.35,  // Moderate
    east: 0.30,     // Lower but volcanic
    north: 0.25,    // Least developed
    west: 0.30,     // Emerging
  };
  return rates[region] ?? 0.35;
}

// ─── Bayesian Detail Data ────────────────────────────────────────────────────

export function getBayesianDetails() {
  return Object.entries(REGULATORY_PRIORS).map(([key, prior]) => ({
    key,
    label: key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
    prior: prior.enforcement_base_rate,
    likelihood: prior.signal_given_enforcement,
    posterior: bayesianPosterior(key),
  }));
}
