import type {
  LegalRisk,
  ClimateRisk,
  PoliticalRisk,
  FinancialRisk,
  RegulatoryRisk,
  MarketRisk,
  RiskOverview,
  AnyRisk,
  RiskCategory,
} from "@/types/bali-risk";

// ── Legal Risks ──────────────────────────────────────────────────────────────

export const legalRisks: LegalRisk[] = [
  {
    id: "legal-001",
    category: "legal",
    title: "Nominee Arrangement Fraud",
    description:
      "Indonesian nominees can legally claim full ownership of property held on behalf of foreigners. Courts consistently rule nominee arrangements unenforceable under Basic Agrarian Law No. 5/1960.",
    severity: "critical",
    trend: "worsening",
    score: 95,
    last_updated: "2026-03-10",
    source: "Indonesian Supreme Court rulings, AREBI reports",
    details:
      "Nominee may renege on agreement, claim property as their own, use it as loan collateral without knowledge, or lose it through personal bankruptcy/divorce/death. Side agreements are unenforceable in Indonesian courts. Government crackdown on nominee structures intensified in 2025-2026. AREBI recorded 180+ formal complaints in Bali in 2024 alone, 30%+ increase in real estate fraud since 2021.",
    mitigation:
      "Never use nominee arrangements. Use PT PMA for commercial properties or Hak Pakai for residential. Engage certified PPAT notary and independent property lawyer.",
    legal_area: "Ownership Structure",
    affected_structures: ["Nominee", "SHM (via nominee)"],
    precedent_cases: 47,
  },
  {
    id: "legal-002",
    category: "legal",
    title: "Hak Pakai Renewal Uncertainty",
    description:
      "Hak Pakai (Right to Use) initial 30-year term + 20-year extension + 30-year renewal (80 years max). Renewal requires landowner consent and government approval — neither guaranteed.",
    severity: "high",
    trend: "stable",
    score: 72,
    last_updated: "2026-03-10",
    source: "National Land Agency (BPN), PP 18/2021",
    details:
      "While Indonesian law provides for up to 80 years of Hak Pakai, each renewal requires active application, government approval, and — critically — landowner consent. If the underlying land ownership changes (death, sale), the new landowner may refuse extension. Indonesian banks will NOT lend against Hak Pakai titles, limiting refinancing options. Only one property per foreigner. Requires valid KITAS/KITAP residential permit.",
    mitigation:
      "Ensure the initial lease agreement includes binding renewal clauses notarized by PPAT. Maintain continuous valid residential permit. Budget for renewal costs.",
    legal_area: "Land Title",
    affected_structures: ["Hak Pakai"],
    precedent_cases: 12,
  },
  {
    id: "legal-003",
    category: "legal",
    title: "Leasehold Extension Risk",
    description:
      "Lease extensions are \"based on mutual agreement\" — weak legal position for foreigners if original lessor is unwilling to renew or demands exploitative terms.",
    severity: "high",
    trend: "stable",
    score: 68,
    last_updated: "2026-03-10",
    source: "Indonesian Civil Code, Property litigation records",
    details:
      "Leasehold (Hak Sewa) contracts typically run 25-30 years, extendable to 80-99 years. However, extension language in standard contracts is intentionally vague. Lessors have been known to demand 200-400% increases at renewal, demand additional payments, or simply refuse. Heirs of deceased lessors may not honor original agreements. No statutory right to renewal exists.",
    mitigation:
      "Negotiate specific renewal terms upfront and notarize them. Include pre-agreed renewal pricing or arbitration clauses. Use escrow for upfront payments. Work with Indonesian property lawyer to draft ironclad extension terms.",
    legal_area: "Lease Agreement",
    affected_structures: ["Leasehold", "Hak Sewa"],
    precedent_cases: 23,
  },
  {
    id: "legal-004",
    category: "legal",
    title: "Double-Selling & Title Fraud",
    description:
      "Properties sold to multiple buyers simultaneously, or with forged ownership certificates. Particularly prevalent in undeveloped land transactions.",
    severity: "high",
    trend: "worsening",
    score: 78,
    last_updated: "2026-03-10",
    source: "Indonesian Ministry of Trade, Bali Police reports",
    details:
      "Forged SHM/HGB certificates are sophisticated and hard to detect without BPN verification. Double-selling occurs when sellers exploit the gap between agreement signing and title transfer. Phantom listings with AI-generated photos target foreign buyers. Notable case: Saudi princess lost $37M to Indonesian real estate scammers. Indonesian Ministry of Trade reports 30%+ increase in real estate fraud since 2021.",
    mitigation:
      "Always verify land title directly at BPN office. Use independent legal counsel (not seller's). Never pay full amount before title transfer. Use escrow services. Inspect property physically. Cross-reference with village (banjar) records.",
    legal_area: "Title Verification",
    affected_structures: ["SHM", "HGB", "Hak Pakai", "Leasehold"],
    precedent_cases: 180,
  },
  {
    id: "legal-005",
    category: "legal",
    title: "PT PMA Regulatory Scrutiny",
    description:
      "Using PT PMA as passive land-holding vehicle without real business operations attracts increasing regulatory scrutiny from BKPM and tax authorities.",
    severity: "medium",
    trend: "worsening",
    score: 62,
    last_updated: "2026-03-10",
    source: "BKPM (Investment Coordinating Board), PP 28/2025",
    details:
      "PT PMA companies must demonstrate genuine business activity matching their stated business classification (KBLI codes). Minimum investment thresholds apply. Annual reporting requirements. Companies operating solely as property holding vehicles face audit risk, license revocation, and forced divestiture. New PP 28/2025 regulation ties licensing to zoning, environment, building approval, and operations into a digital compliance trail.",
    mitigation:
      "Ensure PT PMA has genuine operational activity. Maintain proper KBLI classification. File all annual reports on time. Work with tax consultant and corporate lawyer familiar with BKPM requirements.",
    legal_area: "Corporate Compliance",
    affected_structures: ["PT PMA", "HGB (via PT PMA)"],
    precedent_cases: 8,
  },
  {
    id: "legal-006",
    category: "legal",
    title: "Inheritance & Succession Risk",
    description:
      "Foreign property rights in Indonesia do not automatically transfer to heirs. Without proper estate planning, properties face forced resale or government seizure.",
    severity: "high",
    trend: "stable",
    score: 70,
    last_updated: "2026-03-10",
    source: "Indonesian Inheritance Law, Notarial practice records",
    details:
      "Hak Pakai is personal to the holder — does not auto-inherit. Heirs must independently qualify (valid KITAS/KITAP) or the right lapses. PT PMA shares can transfer but require BKPM approval and compliance verification. Leasehold rights may or may not survive depending on contract terms. Probate in Indonesia is slow and complex for foreign estates.",
    mitigation:
      "Create Indonesian-law-compliant will with certified notary. For PT PMA, maintain updated shareholder agreement with succession clauses. Consider life insurance to cover forced-sale losses. Consult cross-border estate planning specialist.",
    legal_area: "Estate Planning",
    affected_structures: ["Hak Pakai", "PT PMA", "Leasehold"],
    precedent_cases: 15,
  },
];

// ── Climate Risks ────────────────────────────────────────────────────────────

export const climateRisks: ClimateRisk[] = [
  {
    id: "climate-001",
    category: "climate",
    title: "Flood Risk — Monsoon & Urban",
    description:
      "Bali experienced worst floods in a decade in late 2024. River flood hazard classified as MEDIUM (>20% chance of damaging flood within 10 years). Climate change intensifying monsoon patterns.",
    severity: "high",
    trend: "worsening",
    score: 78,
    last_updated: "2026-03-10",
    source: "BNPB, ThinkHazard, Global Climate Risks",
    details:
      "Monsoon season (Nov-March) arrived in September 2024 — unprecedented shift. Warmer Indian Ocean waters fuel intense tropical downpours. Rapid urbanization in Canggu, Denpasar, Ubud, and Sanur has destroyed natural drainage. Deforestation removes water absorption capacity. Poor waste management blocks drainage channels. Properties in low-lying areas face annual flooding. Insurance costs rising 15-25% annually for flood-prone zones.",
    mitigation:
      "Avoid low-lying areas in Canggu, south Denpasar, and river valleys. Invest in elevated construction (minimum 1m above grade). Install proper drainage systems. Verify flood zone maps before purchase. Budget for comprehensive property insurance.",
    hazard_type: "Flood",
    affected_regions: ["Canggu", "Denpasar", "Ubud", "Sanur", "Kuta"],
    probability_10yr: 0.22,
    estimated_damage_pct: 15,
  },
  {
    id: "climate-002",
    category: "climate",
    title: "Volcanic Activity — Mount Agung",
    description:
      "Mount Agung (3,031m) is an active stratovolcano that erupted in 2017-2019. Located in northeast Bali, it poses ashfall, lahar, and pyroclastic risk to surrounding areas.",
    severity: "medium",
    trend: "stable",
    score: 55,
    last_updated: "2026-03-10",
    source: "PVMBG (Indonesian Volcanology Center), BMKG",
    details:
      "The 2017 eruption caused mass evacuations (140,000+ people), airport closures, and property damage from ashfall. Eruption exclusion zones extend 6-12km from crater. Lahar (volcanic mudflow) follows river channels during rain. Ashfall can reach south Bali in heavy eruptions. Tourism drops 40-60% during eruption periods, devastating rental income. Mount Batur also intermittently active.",
    mitigation:
      "Avoid properties within 15km of Mount Agung or Mount Batur. South Bali (Seminyak, Kuta, Uluwatu) has lowest volcanic risk. Ensure property insurance covers volcanic damage. Have evacuation plan. Diversify rental income sources beyond tourism.",
    hazard_type: "Volcanic",
    affected_regions: [
      "Karangasem",
      "Bangli",
      "Klungkung",
      "East Bali",
    ],
    probability_10yr: 0.35,
    estimated_damage_pct: 25,
  },
  {
    id: "climate-003",
    category: "climate",
    title: "Earthquake Risk",
    description:
      "Bali sits on the Pacific Ring of Fire. The 2018 Lombok earthquakes (M6.4-6.9) caused significant structural damage in north Bali. Subduction zone capable of M8+ events.",
    severity: "high",
    trend: "stable",
    score: 65,
    last_updated: "2026-03-10",
    source: "BMKG, USGS, ThinkHazard",
    details:
      "Indonesia experiences 5,000+ earthquakes annually. Bali's proximity to the Java-Bali subduction zone creates persistent seismic risk. Many Bali properties lack earthquake-resistant construction standards. Traditional construction (unreinforced masonry) is particularly vulnerable. The 2018 Lombok sequence damaged structures across northern Bali. Liquefaction risk in coastal reclaimed areas (Benoa, Serangan).",
    mitigation:
      "Ensure construction meets SNI 1726-2019 earthquake resistance standards. Avoid unreinforced masonry structures. Commission structural engineering assessment before purchase. Avoid reclaimed land areas. Include earthquake coverage in property insurance.",
    hazard_type: "Earthquake",
    affected_regions: [
      "All Bali",
      "North Bali (highest)",
      "Coastal areas (liquefaction)",
    ],
    probability_10yr: 0.45,
    estimated_damage_pct: 20,
  },
  {
    id: "climate-004",
    category: "climate",
    title: "Coastal Erosion & Sea Level Rise",
    description:
      "Bali losing 1-3 meters of coastline annually in some areas. Sea level projected to rise 30-60cm by 2100. Beach-front properties face long-term devaluation.",
    severity: "medium",
    trend: "worsening",
    score: 58,
    last_updated: "2026-03-10",
    source: "IPCC AR6, Indonesian Maritime Affairs Ministry",
    details:
      "Kuta, Sanur, and Candidasa beaches show accelerated erosion. Coral reef destruction (bleaching, dynamite fishing, pollution) removes natural wave barriers. Sand mining for construction depletes beach resources. Some beachfront properties have lost 5-10m of setback over 20 years. Storm surge during monsoon season increasingly damaging. Mangrove removal in Benoa Bay area increases flood exposure.",
    mitigation:
      "Avoid beachfront properties with less than 50m setback. Invest in clifftop properties (Uluwatu, Bukit Peninsula) instead. Verify coastal erosion rates from local government data. Factor in 30-year coastal retreat projections. Consider properties elevated above sea level.",
    hazard_type: "Coastal Erosion",
    affected_regions: ["Kuta", "Sanur", "Candidasa", "Benoa", "Lovina"],
    probability_10yr: 0.85,
    estimated_damage_pct: 12,
  },
  {
    id: "climate-005",
    category: "climate",
    title: "Water Scarcity & Infrastructure Stress",
    description:
      "Bali's freshwater aquifers depleting at unsustainable rates. Tourism consumes 65% of water supply. Dry season water shortages increasingly common in developed areas.",
    severity: "medium",
    trend: "worsening",
    score: 52,
    last_updated: "2026-03-10",
    source: "Bali Water Authority, UNESCO water studies",
    details:
      "Groundwater extraction exceeds recharge rates by 200-300% in southern Bali. Saltwater intrusion affecting coastal wells. Hotels and villas consume far more water per capita than local communities. Government considering water use restrictions for new developments. Properties relying solely on well water face increasing risk. Rice paddy conversion to development removes water catchment.",
    mitigation:
      "Install rainwater harvesting systems. Consider properties with access to PDAM (public water supply). Invest in water recycling infrastructure. Avoid areas with known saltwater intrusion. Budget for water storage tanks and treatment.",
    hazard_type: "Water Scarcity",
    affected_regions: [
      "South Bali",
      "Canggu",
      "Seminyak",
      "Kuta",
      "Nusa Dua",
    ],
    probability_10yr: 0.7,
    estimated_damage_pct: 5,
  },
];

// ── Political Risks ──────────────────────────────────────────────────────────

export const politicalRisks: PoliticalRisk[] = [
  {
    id: "political-001",
    category: "political",
    title: "Foreign Ownership Restriction Tightening",
    description:
      "Growing political sentiment to further restrict foreign property ownership. Multiple legislative proposals to reduce Hak Pakai terms or eliminate foreign property rights entirely.",
    severity: "high",
    trend: "worsening",
    score: 75,
    last_updated: "2026-03-10",
    source: "DPR (Parliament) committee minutes, Jakarta Post",
    details:
      "Nationalist political parties gaining influence. Campaign promises to 'protect Indonesian land from foreigners' resonate with voters. Previous government (Jokowi) was more investment-friendly; current administration signals stricter stance. Local Balinese politicians advocate for limits on foreign development. Ban on foreign freehold (SHM) is already law — further restrictions on Hak Pakai, HGB, and leasehold terms are under discussion.",
    mitigation:
      "Diversify across multiple ownership structures. Maintain strong local partnerships. Stay informed on legislative developments. Consider shorter-term investment horizons that don't depend on 80-year tenure. Engage local political consultants.",
    risk_type: "Policy Change",
    affected_parties: ["Foreign investors", "PT PMA holders", "Lease holders"],
    likelihood: 0.35,
  },
  {
    id: "political-002",
    category: "political",
    title: "Local Community (Banjar) Opposition",
    description:
      "Traditional Balinese village councils (banjar) hold significant informal power. Can block development, deny access, or create hostile operating environments for unwelcome foreign projects.",
    severity: "high",
    trend: "stable",
    score: 70,
    last_updated: "2026-03-10",
    source: "Bali Governor's office, Cultural anthropology reports",
    details:
      "Banjar (traditional village councils) control local community life, religious ceremonies, and customary law (adat). They can effectively block construction by denying access, disrupting ceremonies, or mobilizing community opposition. Sacred land, temple zones, and ceremonially significant areas are non-negotiable. Disputes between foreign developers and banjar can persist for years with no legal resolution. Some banjar have forcibly evicted developers who ignored warnings.",
    mitigation:
      "Always consult with local banjar before purchasing. Attend community meetings. Contribute to local temple and ceremony funds. Hire local Balinese project manager. Respect sacred zones and temple buffer areas. Never build on tanah ayahan desa (community-owned land).",
    risk_type: "Community Relations",
    affected_parties: ["Foreign developers", "New investors", "Short-term renters"],
    likelihood: 0.4,
  },
  {
    id: "political-003",
    category: "political",
    title: "Corruption & Bureaucratic Risk",
    description:
      "Indonesia ranks 110/180 on Transparency International's CPI. Property transactions involve multiple government offices, each presenting corruption opportunity.",
    severity: "medium",
    trend: "stable",
    score: 60,
    last_updated: "2026-03-10",
    source: "Transparency International CPI 2025, KPK reports",
    details:
      "Property transactions require approvals from BPN (land agency), local planning office, building department, environmental agency, and tax office. Unofficial 'facilitation fees' are common. Government officials may delay permits indefinitely without payment. Land mafia (mafia tanah) networks operate in some regions, colluding with officials to invalidate legitimate titles. KPK (Anti-Corruption Commission) has limited reach outside Jakarta.",
    mitigation:
      "Work exclusively through reputable legal counsel. Document all payments. Use official channels and receipts. Report corruption attempts to KPK. Build timeline buffers for permit processes. Budget for proper legal and notarial fees.",
    risk_type: "Corruption",
    affected_parties: ["All foreign investors"],
    likelihood: 0.55,
  },
  {
    id: "political-004",
    category: "political",
    title: "Tourism Policy Volatility",
    description:
      "Government tourism policies directly impact Bali property values. Digital nomad visa changes, tourist taxes, and COVID-era closures demonstrated vulnerability.",
    severity: "medium",
    trend: "stable",
    score: 55,
    last_updated: "2026-03-10",
    source: "Indonesian Tourism Ministry, Bali Governor's office",
    details:
      "Bali's economy is 80%+ tourism-dependent. Government controls tourist visa policies, tourist taxes (IDR 150K entry fee), and development zones. COVID-19 border closure (2020-2022) devastated rental income for 2+ years. Digital nomad visa (B211A) policies shift frequently. New regulations can restrict short-term rental operations overnight. Bali tourism tax changes affect visitor numbers and property demand.",
    mitigation:
      "Don't assume continuous tourism growth. Stress-test rental projections with 30-50% occupancy scenarios. Diversify income (long-term + short-term rental mix). Keep 18-24 months operating reserves. Monitor Ministry of Tourism announcements.",
    risk_type: "Policy Change",
    affected_parties: ["Rental property investors", "Villa operators", "Tourism-dependent properties"],
    likelihood: 0.45,
  },
];

// ── Financial Risks ──────────────────────────────────────────────────────────

export const financialRisks: FinancialRisk[] = [
  {
    id: "financial-001",
    category: "financial",
    title: "IDR Currency Depreciation",
    description:
      "Indonesian Rupiah has lost ~40% against USD over 10 years. Property valued in IDR but marketed in USD creates valuation gap. Rental income in IDR erodes against foreign obligations.",
    severity: "high",
    trend: "worsening",
    score: 72,
    last_updated: "2026-03-10",
    source: "Bank Indonesia, Bloomberg FX data",
    details:
      "IDR/USD moved from ~11,000 (2014) to ~16,200 (2026). Real estate transactions often quoted in USD but legally settled in IDR. Rental agreements in IDR lose purchasing power. Repatriation of sale proceeds at unfavorable rates. Bank Indonesia maintains managed float but limited reserves for sustained intervention. Indonesian inflation typically 3-6% vs 2-3% in USD-denominated economies.",
    mitigation:
      "Negotiate USD-denominated lease agreements where possible. Use natural hedging (USD rental income if targeting foreign tenants). Consider forward contracts for large transactions. Keep reserves in diversified currencies. Factor 5% annual IDR depreciation into projections.",
    risk_type: "Currency",
    metric_value: "IDR 16,200/USD",
    metric_label: "Current Exchange Rate",
  },
  {
    id: "financial-002",
    category: "financial",
    title: "No Mortgage Access for Foreigners",
    description:
      "Indonesian banks will NOT provide mortgages on Hak Pakai properties. Foreigners must purchase with 100% cash or offshore financing at higher rates.",
    severity: "high",
    trend: "stable",
    score: 68,
    last_updated: "2026-03-10",
    source: "Bank Indonesia regulations, OJK guidelines",
    details:
      "Only SHM and HGB titles qualify as bank collateral in Indonesia. Foreigners cannot hold SHM. HGB requires PT PMA. Even with PT PMA and HGB, loan-to-value ratios are typically 50-60% with high interest rates (10-14% IDR). Offshore banks rarely accept Indonesian property as collateral. This creates 100% equity requirement, increasing capital at risk and reducing diversification.",
    mitigation:
      "Plan for 100% cash purchase. Explore home equity lines from home country against other assets. Consider Singapore-based banks for regional property lending. PT PMA with HGB offers best (though limited) financing options.",
    risk_type: "Credit",
    metric_value: "0% LTV",
    metric_label: "Foreign Mortgage Availability",
  },
  {
    id: "financial-003",
    category: "financial",
    title: "Capital Repatriation Restrictions",
    description:
      "While Indonesia allows capital repatriation, the process involves tax clearance, Bank Indonesia reporting, and can be delayed weeks to months.",
    severity: "medium",
    trend: "stable",
    score: 52,
    last_updated: "2026-03-10",
    source: "Bank Indonesia foreign exchange regulations",
    details:
      "Sale proceeds above $25,000 equivalent require Bank Indonesia reporting. Tax clearance certificate needed (can take 30-90 days). Withholding tax of 2.5% on property sales by foreigners. VAT implications for PT PMA property dispositions. Transfer fees and bank charges add 1-3%. Documentation requirements are extensive. In practice, large transfers can be flagged and held for review.",
    mitigation:
      "Maintain clear tax records throughout ownership. Work with Indonesian tax consultant before selling. Budget 3-6 months for complete repatriation process. Consider structuring sale through PT PMA for cleaner corporate transfer.",
    risk_type: "Liquidity",
    metric_value: "30-90 days",
    metric_label: "Avg. Repatriation Time",
  },
  {
    id: "financial-004",
    category: "financial",
    title: "Property Tax & Fee Escalation",
    description:
      "Indonesian government increasing property tax assessments (NJOP) to capture rising real estate values. New local taxes and fees applied retroactively.",
    severity: "medium",
    trend: "worsening",
    score: 48,
    last_updated: "2026-03-10",
    source: "Bali Provincial Tax Office, UU HKPD 2022",
    details:
      "Property transfer tax (BPHTB) is 5% of sale value. Annual property tax (PBB) based on NJOP which government can increase unilaterally. New tourism tax (IDR 150K per tourist) shifts costs into ecosystem. PT PMA annual reporting costs $2,000-5,000. Notarial and PPAT fees typically 1% of transaction value. These aggregate costs reduce net yield significantly.",
    mitigation:
      "Budget 7-10% of purchase price for transaction costs. Include escalating tax projections in financial models. Maintain accurate NJOP records. Engage local tax consultant for annual filing. Monitor regional tax regulation changes.",
    risk_type: "Tax",
    metric_value: "~8%",
    metric_label: "Total Transaction Cost",
  },
];

// ── Regulatory Risks ─────────────────────────────────────────────────────────

export const regulatoryRisks: RegulatoryRisk[] = [
  {
    id: "regulatory-001",
    category: "regulatory",
    title: "PP 28/2025 — New Risk-Based Licensing",
    description:
      "New government regulation PP 28/2025 creates digital compliance trail linking zoning, environmental, building approval, and operations licensing. Non-compliance risks asset seizure.",
    severity: "critical",
    trend: "worsening",
    score: 82,
    last_updated: "2026-03-10",
    source: "PP 28/2025, OSS (Online Single Submission) system",
    details:
      "PP 28/2025 replaces fragmented licensing with integrated risk-based framework. All property activities (purchase, development, rental operations) must comply with layered digital approvals. Existing properties may need retroactive compliance. Enforcement through the OSS system means violations are automatically flagged. Penalties include operational suspension, fines, and forced property divestiture. Foreign investors operating rental businesses without proper NIB (business registration) and KBLI codes face immediate enforcement.",
    mitigation:
      "Register all business activities through OSS immediately. Ensure correct KBLI code classification. Obtain all environmental impact assessments (AMDAL/UKL-UPL). Hire compliance consultant familiar with PP 28/2025. Budget 6-12 months for full compliance.",
    regulation: "PP 28/2025",
    effective_date: "2025-06-01",
    compliance_deadline: "2026-12-31",
    penalty: "Business suspension, forced divestiture, fines up to IDR 5B",
  },
  {
    id: "regulatory-002",
    category: "regulatory",
    title: "15m Building Height Restriction",
    description:
      "Bali enforces strict 15-meter (approximately 4 stories) building height limit island-wide. Violations result in forced demolition.",
    severity: "high",
    trend: "stable",
    score: 65,
    last_updated: "2026-03-10",
    source: "Bali Governor Regulation, DPRD Bali",
    details:
      "The 15m height limit is rooted in Hindu-Balinese cultural principles (no structure taller than a coconut palm). Strictly enforced with demolition orders issued for violations. Some developers build illegal upper floors hoping for amnesty — this has never been granted. Properties purchased with illegal floors face demolition risk and value loss. Basement construction also regulated in many zones.",
    mitigation:
      "Verify building permits (IMB/PBG) match actual construction before purchase. Never purchase properties exceeding height limits. Commission independent building survey. Check all floors against approved building plans.",
    regulation: "Perda Provinsi Bali No. 16/2009",
    effective_date: "2009-01-01",
    compliance_deadline: "Ongoing",
    penalty: "Forced demolition, fines, criminal prosecution",
  },
  {
    id: "regulatory-003",
    category: "regulatory",
    title: "Short-Term Rental Licensing Requirements",
    description:
      "Operating a villa/property as short-term rental requires specific business licensing (Pondok Wisata). Operating without license = illegal business activity.",
    severity: "high",
    trend: "worsening",
    score: 70,
    last_updated: "2026-03-10",
    source: "Bali Tourism Office, PP 28/2025",
    details:
      "A Pondok Wisata license is required for any rental operation. Requires: zoning compliance, building permit, fire safety certificate, health certificate, and environmental assessment. PT PMA is required for foreigner-operated businesses. Operating on tourist visa with rental income is technically tax evasion. Government cracking down on unlicensed Airbnb-style rentals. Penalties include property closure, deportation, and criminal charges.",
    mitigation:
      "Obtain Pondok Wisata license before operating. Set up PT PMA for rental business. Use licensed property management company. Ensure all staff are properly employed with BPJS. Maintain all required safety certificates.",
    regulation: "Pergub Bali No. 25/2020, PP 28/2025",
    effective_date: "2020-01-01",
    compliance_deadline: "Ongoing",
    penalty: "Business closure, fines, deportation for illegal business activity",
  },
  {
    id: "regulatory-004",
    category: "regulatory",
    title: "Green Zone / Agricultural Land Protection",
    description:
      "Bali's zoning designates large areas as green zones (agricultural, sacred, forest). Building on green zone land is illegal regardless of land ownership.",
    severity: "high",
    trend: "stable",
    score: 68,
    last_updated: "2026-03-10",
    source: "RTRW Bali (Spatial Planning), Bali BPN",
    details:
      "Bali's RTRW (Regional Spatial Plan) classifies land as green zone (protected, agricultural, sacred, forest) or development zone. Building permits are NOT issued for green zone land. However, sellers may misrepresent zoning status or sell agricultural land at 'development' prices. Subak (rice terrace irrigation system) land is UNESCO-recognized and absolutely non-developable. Converting agricultural land requires provincial governor approval — rarely granted.",
    mitigation:
      "Verify zoning classification through DPMPTSP (local planning office) before purchase. Never buy based on seller's zoning claims. Request IPPT (land use permission) as first step. Engage certified surveyor to confirm boundaries against spatial plan.",
    regulation: "RTRW Provinsi Bali",
    effective_date: "2009-01-01",
    compliance_deadline: "Ongoing",
    penalty: "Demolition, land reversion, criminal prosecution for environmental damage",
  },
];

// ── Market Risks ─────────────────────────────────────────────────────────────

export const marketRisks: MarketRisk[] = [
  {
    id: "market-001",
    category: "market",
    title: "Oversupply in Key Tourist Areas",
    description:
      "Villa and accommodation oversupply in Canggu, Seminyak, and Ubud driving occupancy rates down and creating deflationary pressure on rental yields.",
    severity: "high",
    trend: "worsening",
    score: 72,
    last_updated: "2026-03-10",
    source: "Bali Tourism Board, STR Global data",
    details:
      "Canggu has seen 300%+ increase in villa inventory since 2019. Average occupancy rates dropped from 75% (2019) to 55% (2025) in saturated areas. New construction continues despite oversupply signals. Many villa owners competing on price, eroding margins. Budget villa segment particularly oversaturated. Luxury segment showing more resilience but requires $500K+ investment. Airbnb listings in Bali grew from 25,000 (2019) to 80,000+ (2025).",
    mitigation:
      "Target underserved niches (long-term digital nomad, wellness retreat, family villas). Avoid Canggu unless ultra-premium segment. Consider emerging areas (Tabanan, Sidemen, Amed). Focus on unique value proposition and superior management.",
    risk_type: "Oversupply",
    metric_value: "55%",
    metric_label: "Avg. Occupancy Rate (Canggu)",
  },
  {
    id: "market-002",
    category: "market",
    title: "Tourism Dependency — Single Sector Risk",
    description:
      "Bali's economy is 80%+ dependent on tourism. Any disruption (pandemic, geopolitical, natural disaster) causes cascading property value decline.",
    severity: "high",
    trend: "stable",
    score: 75,
    last_updated: "2026-03-10",
    source: "BPS Bali (Statistics), World Bank Indonesia",
    details:
      "COVID-19 demonstrated catastrophic vulnerability: 2 years of near-zero tourism income, property prices dropped 20-40% in tourist areas, many forced sales by distressed owners. The island has minimal economic diversification — no significant manufacturing, mining, or agricultural export base. China market dependency (pre-COVID ~25% of tourists) creates geopolitical exposure. Australian tourist share (~25%) subject to AUD/IDR volatility.",
    mitigation:
      "Price properties assuming 50% tourism disruption scenario. Maintain 24-month operating reserves. Target Indonesian domestic market (growing middle class). Consider mixed-use properties with non-tourism income. Don't leverage properties based on peak occupancy projections.",
    risk_type: "Tourism Dependency",
    metric_value: "80%+",
    metric_label: "Economy Tourism Dependency",
  },
  {
    id: "market-003",
    category: "market",
    title: "Infrastructure Bottlenecks",
    description:
      "Bali's infrastructure (roads, power, water, waste) has not kept pace with development. Traffic congestion, power outages, and waste management failures affect property desirability.",
    severity: "medium",
    trend: "worsening",
    score: 58,
    last_updated: "2026-03-10",
    source: "Bali Public Works, World Bank infrastructure assessment",
    details:
      "Average commute times in south Bali have tripled in 10 years. Power grid unreliable outside major tourist zones — most villas require backup generators. Waste management crisis: illegal dumping, ocean pollution, limited recycling. Water supply inadequate (see Climate risks). Limited public transportation forces car dependency. Planned toll roads and new airport face political opposition and funding delays.",
    mitigation:
      "Invest in areas with existing infrastructure capacity. Install solar panels and battery backup. Include generator in property. Factor in waste management costs. Consider distance to airport (current DPS and planned north Bali airport).",
    risk_type: "Infrastructure",
    metric_value: "3x",
    metric_label: "Commute Time Increase (10yr)",
  },
  {
    id: "market-004",
    category: "market",
    title: "Rental Yield Compression",
    description:
      "Gross rental yields declining from 12-15% (2017) to 6-8% (2025) as property prices rose faster than rental rates, particularly in established tourist areas.",
    severity: "medium",
    trend: "worsening",
    score: 55,
    last_updated: "2026-03-10",
    source: "Rumah.com, Colliers Indonesia",
    details:
      "Property purchase prices have increased 40-60% in Canggu/Seminyak since 2019. Nightly rental rates have only increased 10-20% in the same period. Property management fees (20-30%) further reduce net yields. Seasonal variation extreme: Dec-Mar peak vs May-Sep trough can see 50% rate difference. Long-term rental yields lower (4-6%) but more stable. Capital appreciation may compensate but is not guaranteed.",
    mitigation:
      "Run conservative yield calculations at 60% occupancy. Factor all costs: management (25%), maintenance (5%), taxes (3%), insurance (2%). Target net yield above 5% post-costs. Consider long-term rental for stable income. Focus on areas with limited competing supply.",
    risk_type: "Rental Yield",
    metric_value: "6-8%",
    metric_label: "Gross Rental Yield (2025)",
  },
];

// ── Aggregation Functions ────────────────────────────────────────────────────

export function getAllRisks(): AnyRisk[] {
  return [
    ...legalRisks,
    ...climateRisks,
    ...politicalRisks,
    ...financialRisks,
    ...regulatoryRisks,
    ...marketRisks,
  ];
}

export function getRisksByCategory(category: RiskCategory): AnyRisk[] {
  const map: Record<RiskCategory, AnyRisk[]> = {
    legal: legalRisks,
    climate: climateRisks,
    political: politicalRisks,
    financial: financialRisks,
    regulatory: regulatoryRisks,
    market: marketRisks,
  };
  return map[category] || [];
}

export function getRiskOverview(): RiskOverview {
  const all = getAllRisks();
  const scores = all.map((r) => r.score);
  const overall = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);

  const critical = all.filter((r) => r.severity === "critical").length;
  const high = all.filter((r) => r.severity === "high").length;
  const medium = all.filter((r) => r.severity === "medium").length;
  const low = all.filter((r) => r.severity === "low").length;

  // Find top risk category by average score
  const categories: RiskCategory[] = [
    "legal",
    "climate",
    "political",
    "financial",
    "regulatory",
    "market",
  ];
  let topCategory: RiskCategory = "legal";
  let topAvg = 0;
  for (const cat of categories) {
    const catRisks = getRisksByCategory(cat);
    const avg =
      catRisks.reduce((a, r) => a + r.score, 0) / catRisks.length;
    if (avg > topAvg) {
      topAvg = avg;
      topCategory = cat;
    }
  }

  // Overall trend based on worsening count
  const worsening = all.filter((r) => r.trend === "worsening").length;
  const improving = all.filter((r) => r.trend === "improving").length;
  const trend =
    worsening > improving + 3
      ? "worsening"
      : improving > worsening + 3
        ? "improving"
        : "stable";

  return {
    overall_score: overall,
    total_indicators: all.length,
    critical_count: critical,
    high_count: high,
    medium_count: medium,
    low_count: low,
    top_risk_category: topCategory,
    trend: trend as "worsening" | "stable" | "improving",
  };
}
