export interface District {
  id: string;
  name: string;
  regency: string;
  lat: number;
  lng: number;
  population: number;
  elevation_m: number;
  coastal: boolean;
  volcanic_proximity_km: number;
  avg_land_price_usd_m2: number;
  tourism_intensity: number;
  infrastructure_index: number;
  foreign_investor_density: number;
  dominant_zone: string;
  dominant_title: string;
  tags: string[];
  area_km2: number;
  composite_risk: number;
  investment_grade: string;
  env_risk: number;
  env_factors: Record<string, number>;
  seismic_risk: number;
  seismic_factors: Record<string, number>;
  legal_risk: number;
  legal_factors: Record<string, number>;
  admin_risk: number;
  admin_factors: Record<string, number>;
}

export interface RiskData {
  districts: District[];
  generated_at: string;
}
