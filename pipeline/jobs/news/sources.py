"""News source definitions and reliability scores.

Each source has a base URL, fetch strategy, and credibility weighting.
Sources are ranked by timeliness × accuracy for conflict intelligence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SourceType(str, Enum):
    RSS = "rss"
    API = "api"
    SCRAPE = "scrape"
    TELEGRAM = "telegram"


@dataclass
class NewsSource:
    """A news source configuration."""
    name: str
    source_type: SourceType
    base_url: str
    reliability: float          # 0-1, higher = more trustworthy
    timeliness: float           # 0-1, higher = faster reporting
    language: str = "en"
    region_focus: str = ""      # primary region covered
    fetch_interval_sec: int = 300
    enabled: bool = True
    headers: dict = field(default_factory=dict)
    description: str = ""


# === Source Registry ===
# Ordered by overall signal quality (reliability × timeliness)

SOURCES: dict[str, NewsSource] = {
    # --- Tier 1: Primary OSINT (highest signal quality) ---

    "deepstate": NewsSource(
        name="DeepState Map",
        source_type=SourceType.API,
        base_url="https://deepstatemap.live",
        reliability=0.85,
        timeliness=0.90,
        region_focus="donbas,south,kursk",
        fetch_interval_sec=300,
        description="Ukrainian OSINT map with near-real-time frontline updates. "
                    "Best source for control status changes.",
    ),
    "isw": NewsSource(
        name="Institute for the Study of War",
        source_type=SourceType.RSS,
        base_url="https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment",
        reliability=0.90,
        timeliness=0.60,
        region_focus="all",
        fetch_interval_sec=3600,
        description="Daily analytical assessment. High reliability, moderate latency. "
                    "Best for strategic context and confirmed territorial changes.",
    ),

    # --- Tier 2: Wire services (fast, verified) ---

    "reuters_ukraine": NewsSource(
        name="Reuters Ukraine",
        source_type=SourceType.RSS,
        base_url="https://www.reuters.com/world/europe/",
        reliability=0.85,
        timeliness=0.80,
        fetch_interval_sec=600,
        description="Major wire service. Fast, verified reporting.",
    ),
    "ukrinform": NewsSource(
        name="Ukrinform",
        source_type=SourceType.RSS,
        base_url="https://www.ukrinform.net/rubric-ato",
        reliability=0.75,
        timeliness=0.85,
        language="en",
        region_focus="all",
        fetch_interval_sec=600,
        description="Ukrainian state news agency. Good frontline coverage, "
                    "pro-Ukrainian bias to account for.",
    ),

    # --- Tier 3: OSINT aggregators (fast, variable quality) ---

    "liveuamap": NewsSource(
        name="LiveUAMap",
        source_type=SourceType.SCRAPE,
        base_url="https://liveuamap.com",
        reliability=0.70,
        timeliness=0.90,
        region_focus="all",
        fetch_interval_sec=300,
        description="Crowdsourced conflict map. Very fast, needs verification.",
    ),
    "militaryland": NewsSource(
        name="MilitaryLand.net",
        source_type=SourceType.RSS,
        base_url="https://militaryland.net",
        reliability=0.75,
        timeliness=0.70,
        region_focus="donbas,south",
        fetch_interval_sec=1800,
        description="Detailed frontline analysis with maps.",
    ),

    # --- Tier 4: Social media / Telegram (fastest, least reliable) ---

    "rybar": NewsSource(
        name="Rybar (RU milblogger)",
        source_type=SourceType.TELEGRAM,
        base_url="https://t.me/ryaborvoting",
        reliability=0.55,
        timeliness=0.95,
        language="ru",
        region_focus="all",
        fetch_interval_sec=300,
        description="Influential Russian milblogger. Fast but pro-RU bias. "
                    "Good early warning for RU offensives.",
    ),
    "deepstate_ua_tg": NewsSource(
        name="DeepState UA (Telegram)",
        source_type=SourceType.TELEGRAM,
        base_url="https://t.me/DeepStateUA",
        reliability=0.80,
        timeliness=0.95,
        language="uk",
        region_focus="donbas,south",
        fetch_interval_sec=300,
        description="Official DeepState Telegram. Fastest confirmed frontline updates.",
    ),
    "war_monitor": NewsSource(
        name="WarMonitor",
        source_type=SourceType.TELEGRAM,
        base_url="https://t.me/WarMonitor3",
        reliability=0.60,
        timeliness=0.90,
        language="en",
        region_focus="all",
        fetch_interval_sec=600,
        description="English-language OSINT aggregator.",
    ),
}


# Settlement name aliases for entity extraction
SETTLEMENT_ALIASES: dict[str, list[str]] = {
    "pokrovsk": ["pokrovsk", "покровськ", "покровск"],
    "chasiv_yar": ["chasiv yar", "часів яр", "часов яр", "chasov yar"],
    "toretsk": ["toretsk", "торецьк", "торецк", "dzerzhynsk"],
    "kupiansk": ["kupiansk", "купʼянськ", "купянск", "kupyansk"],
    "zaporizhzhia": ["zaporizhzhia", "запоріжжя", "запорожье", "zaporozhye"],
    "orikhiv": ["orikhiv", "оріхів", "орехов", "orekhov"],
    "hryshyne": ["hryshyne", "гришине", "гришино", "hrysyne"],
    "kramatorsk": ["kramatorsk", "краматорськ", "краматорск"],
    "sloviansk": ["sloviansk", "словʼянськ", "славянск", "slovyansk"],
    "bakhmut": ["bakhmut", "бахмут", "artomovsk", "артемовськ"],
    "avdiivka": ["avdiivka", "авдіївка", "авдеевка", "avdeevka"],
    "vuhledar": ["vuhledar", "вугледар", "угледар", "ugledar"],
    "kurakhove": ["kurakhove", "курахове", "курахово"],
    "selydove": ["selydove", "селидове", "селидово"],
    "myrnohrad": ["myrnohrad", "мирноград"],
    "siversk": ["siversk", "сіверськ", "северск"],
    "lyman": ["lyman", "лиман"],
    "sudzha": ["sudzha", "суджа"],
    "kherson_city": ["kherson", "херсон"],
    "mariupol": ["mariupol", "маріуполь", "мариуполь"],
    "melitopol": ["melitopol", "мелітополь", "мелитополь"],
}

# Keywords that indicate control changes
CONTROL_KEYWORDS = {
    "captured": ("RU", 0.7),
    "liberated": ("UA", 0.7),
    "recaptured": ("UA", 0.8),
    "fell": ("RU", 0.6),
    "retreated from": ("UA_LOST", 0.6),
    "withdrawn from": ("UA_LOST", 0.6),
    "entered": ("RU_ADVANCE", 0.5),
    "raised flag": ("RU", 0.8),
    "hoisted flag": ("RU", 0.8),
    "fierce fighting": ("CONTESTED", 0.5),
    "heavy clashes": ("CONTESTED", 0.5),
    "contested": ("CONTESTED", 0.4),
    "encirclement": ("RU_ADVANCE", 0.6),
    "surrounded": ("RU_ADVANCE", 0.6),
    "counterattack": ("UA_ADVANCE", 0.5),
    "pushed back": ("UA_ADVANCE", 0.5),
}

# Sentiment keywords for scoring
SENTIMENT_KEYWORDS = {
    # Pro-UA / negative for RU
    "counteroffensive": 0.6,
    "liberated": 0.7,
    "recaptured": 0.7,
    "destroyed": 0.3,
    "intercepted": 0.4,
    "repelled": 0.5,
    "casualties": -0.2,
    "ammunition delivery": 0.5,
    "aid package": 0.5,

    # Pro-RU / negative for UA
    "captured": -0.6,
    "fell": -0.7,
    "retreated": -0.5,
    "encircled": -0.6,
    "breakthrough": -0.5,
    "glide bomb": -0.3,
    "shelling": -0.3,
    "missile strike": -0.4,

    # Neutral / escalation
    "ceasefire": 0.1,
    "negotiations": 0.2,
    "mobilization": -0.1,
    "nuclear": -0.4,
}
