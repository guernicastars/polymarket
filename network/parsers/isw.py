"""ISW (Institute for Study of War) parser — extracts settlement mentions.

TODO Phase 2:
- Parse daily ISW reports for settlement mentions
- Extract assault/shelling intensity signals from text
- NLP-based sentiment on settlement status
"""

from __future__ import annotations


class ISWParser:
    """Stub: parse ISW daily reports for settlement intelligence."""

    BASE_URL = "https://www.understandingwar.org/"

    def fetch_latest_report(self) -> str:
        """Fetch latest ISW daily report text."""
        raise NotImplementedError("ISW parser not yet implemented")

    def extract_settlement_mentions(self, text: str) -> list[dict]:
        """Extract settlement names and context from report text."""
        raise NotImplementedError("ISW parser not yet implemented")
