"""Folium map rendering of the Donbas network graph.

TODO Phase 2:
- Render settlements as colored markers (UA=blue, RU=red, CONTESTED=orange)
- Draw edges as polylines with weight-based thickness
- Add popups with vulnerability scores and supply info
- Heatmap overlay for assault intensity
- Export to HTML for dashboard embedding
"""

from __future__ import annotations


class MapRenderer:
    """Stub: render Donbas graph on an interactive Folium map."""

    CENTER = (48.5, 37.5)  # Donbas center
    ZOOM = 8

    def render(self) -> str:
        """Render map to HTML string."""
        raise NotImplementedError("Map renderer not yet implemented")

    def save_html(self, path: str) -> None:
        """Save rendered map to HTML file."""
        raise NotImplementedError("Map renderer not yet implemented")
