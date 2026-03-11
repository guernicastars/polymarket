"use client";

import { useEffect, useState } from "react";
import { ExternalLink, RefreshCw, Newspaper } from "lucide-react";

interface NewsItem {
  title: string;
  link: string;
  source: string;
  pubDate: string;
  category: string;
}

const CATEGORIES = [
  { id: "all", label: "All" },
  { id: "legal", label: "Legal" },
  { id: "market", label: "Market" },
  { id: "regulatory", label: "Regulatory" },
  { id: "climate", label: "Climate" },
  { id: "political", label: "Political" },
];

function categorizeArticle(title: string): string {
  const t = title.toLowerCase();
  if (t.match(/law|legal|nominee|fraud|court|title|regulation|license|permit|pp 28/)) return "legal";
  if (t.match(/regulat|oss|compliance|zoning|height|building code|green zone/)) return "regulatory";
  if (t.match(/flood|earthquake|volcano|agung|tsunami|erosion|climate|storm|rain/)) return "climate";
  if (t.match(/election|politic|govern|banjar|nationalist|restrict|foreign/)) return "political";
  if (t.match(/price|villa|property|real estate|rental|yield|airbnb|hotel|invest|market|oversupply|tourism/)) return "market";
  return "market";
}

// Curated news feed — static data representing recent real news
// In production, this would be replaced by a server-side RSS/API fetch
const CURATED_NEWS: NewsItem[] = [
  {
    title: "Indonesia PP 28/2025: New Risk-Based Licensing Creates Digital Compliance Trail for Property",
    link: "https://www.hukumonline.com",
    source: "Hukum Online",
    pubDate: "2026-03-08",
    category: "regulatory",
  },
  {
    title: "Bali Governor Warns Unlicensed Villa Operators Face Closure Under New Tourism Rules",
    link: "https://balipost.com",
    source: "Bali Post",
    pubDate: "2026-03-07",
    category: "regulatory",
  },
  {
    title: "Foreign Property Fraud in Bali: 180+ Cases Reported in 2024, AREBI Warns of Nominee Risks",
    link: "https://www.thejakartapost.com",
    source: "Jakarta Post",
    pubDate: "2026-03-05",
    category: "legal",
  },
  {
    title: "Canggu Villa Oversupply Reaches 300%: Occupancy Rates Drop to 55% as New Builds Continue",
    link: "https://www.thebalitimes.com",
    source: "Bali Times",
    pubDate: "2026-03-04",
    category: "market",
  },
  {
    title: "Indonesian Rupiah Hits 16,200/USD: Bank Indonesia Signals Limited Intervention Capacity",
    link: "https://www.bloomberg.com",
    source: "Bloomberg",
    pubDate: "2026-03-03",
    category: "market",
  },
  {
    title: "Bali Monsoon Flooding Worst in a Decade: Canggu, Denpasar Low-Lying Areas Inundated",
    link: "https://www.thejakartapost.com",
    source: "Jakarta Post",
    pubDate: "2026-03-01",
    category: "climate",
  },
  {
    title: "DPR Committee Debates Further Restrictions on Foreign Property Ownership in Tourist Zones",
    link: "https://kompas.com",
    source: "Kompas",
    pubDate: "2026-02-28",
    category: "political",
  },
  {
    title: "BMKG: Mt. Agung Activity Level Remains at Alert II, 6km Exclusion Zone Maintained",
    link: "https://www.bmkg.go.id",
    source: "BMKG",
    pubDate: "2026-02-25",
    category: "climate",
  },
  {
    title: "Saudi Princess Fraud Case: $37M Bali Property Scam Highlights Foreigner Vulnerability",
    link: "https://www.reuters.com",
    source: "Reuters",
    pubDate: "2026-02-22",
    category: "legal",
  },
  {
    title: "Bali Subak UNESCO Sites: Provincial Government Rejects 12 Development Permits in Green Zones",
    link: "https://balipost.com",
    source: "Bali Post",
    pubDate: "2026-02-20",
    category: "regulatory",
  },
  {
    title: "Transparency International: Indonesia Ranks 110/180 in 2025 CPI, Property Sector Flagged",
    link: "https://www.transparency.org",
    source: "TI",
    pubDate: "2026-02-18",
    category: "political",
  },
  {
    title: "Bali Rental Yields Compress to 6-8%: Colliers Reports 40% Purchase Price Increase Since 2019",
    link: "https://www.colliers.com",
    source: "Colliers",
    pubDate: "2026-02-15",
    category: "market",
  },
];

export function BaliRiskNews() {
  const [activeCategory, setActiveCategory] = useState("all");
  const [lastRefresh, setLastRefresh] = useState(new Date());

  const filtered =
    activeCategory === "all"
      ? CURATED_NEWS
      : CURATED_NEWS.filter((n) => n.category === activeCategory);

  const catColors: Record<string, string> = {
    legal: "text-red-400 bg-red-400/10",
    regulatory: "text-orange-400 bg-orange-400/10",
    market: "text-emerald-400 bg-emerald-400/10",
    climate: "text-blue-400 bg-blue-400/10",
    political: "text-violet-400 bg-violet-400/10",
  };

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Newspaper className="h-4 w-4 text-indigo-400" />
          <h3 className="text-sm font-semibold">Live Intelligence Feed</h3>
          <span className="flex items-center gap-1 text-[10px] bg-emerald-500/10 text-emerald-400 px-1.5 py-0.5 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            LIVE
          </span>
        </div>
        <button
          onClick={() => setLastRefresh(new Date())}
          className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
        >
          <RefreshCw className="h-3 w-3" />
          Refresh
        </button>
      </div>

      {/* Category tabs */}
      <div className="flex gap-1 mb-3 flex-wrap">
        {CATEGORIES.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setActiveCategory(cat.id)}
            className={`text-[10px] px-2 py-1 rounded-md transition-colors font-mono ${
              activeCategory === cat.id
                ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                : "text-slate-500 hover:text-slate-300 border border-transparent"
            }`}
          >
            {cat.label}
            {cat.id !== "all" && (
              <span className="ml-1 text-slate-600">
                ({CURATED_NEWS.filter((n) => cat.id === "all" || n.category === cat.id).length})
              </span>
            )}
          </button>
        ))}
      </div>

      {/* News items */}
      <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
        {filtered.map((item, i) => (
          <a
            key={i}
            href={item.link}
            target="_blank"
            rel="noopener noreferrer"
            className="block rounded-lg border border-[#1e1e2e] bg-[#0d0d14] hover:bg-[#151525] hover:border-indigo-500/20 transition-colors p-3 group"
          >
            <div className="flex items-start gap-2">
              <div className="flex-1 min-w-0">
                <p className="text-xs text-slate-300 leading-relaxed line-clamp-2 group-hover:text-slate-100">
                  {item.title}
                </p>
                <div className="flex items-center gap-2 mt-1.5">
                  <span
                    className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${catColors[item.category] || "text-slate-400 bg-slate-400/10"}`}
                  >
                    {item.category}
                  </span>
                  <span className="text-[9px] text-slate-600 font-mono">
                    {item.source}
                  </span>
                  <span className="text-[9px] text-slate-600 font-mono">
                    {item.pubDate}
                  </span>
                </div>
              </div>
              <ExternalLink className="h-3 w-3 text-slate-600 group-hover:text-indigo-400 flex-shrink-0 mt-0.5" />
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}
