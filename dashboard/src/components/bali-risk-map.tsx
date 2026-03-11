"use client";

import { useEffect, useRef, useState } from "react";
import type { BaliNode, RiskZone } from "@/lib/bali-risk-engine";
import {
  computeAllRiskScores,
  BALI_EDGES,
  RISK_ZONES,
  computeVulnerability,
  BALI_NODES,
} from "@/lib/bali-risk-engine";

function riskColor(score: number): string {
  if (score >= 0.7) return "#ef4444";
  if (score >= 0.5) return "#f97316";
  if (score >= 0.35) return "#eab308";
  return "#22c55e";
}

function riskLabel(score: number): string {
  if (score >= 0.7) return "CRITICAL";
  if (score >= 0.5) return "HIGH";
  if (score >= 0.35) return "MEDIUM";
  return "LOW";
}

const ZONE_TYPE_LABELS: Record<string, string> = {
  flood: "Flood Zone",
  volcanic: "Volcanic Zone",
  earthquake: "Earthquake Zone",
  erosion: "Coastal Erosion",
  saturation: "Market Saturation",
  green_zone: "Protected Green Zone",
};

interface MapLayer {
  id: string;
  label: string;
  enabled: boolean;
  color: string;
}

export function BaliRiskMap() {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const layerGroupsRef = useRef<Record<string, any>>({});
  const [selectedNode, setSelectedNode] = useState<BaliNode | null>(null);
  const [layers, setLayers] = useState<MapLayer[]>([
    { id: "nodes", label: "Risk Nodes", enabled: true, color: "#e2e8f0" },
    { id: "edges", label: "Connections", enabled: true, color: "#475569" },
    { id: "flood", label: "Flood Zones", enabled: true, color: "#3b82f6" },
    { id: "volcanic", label: "Volcanic Zones", enabled: true, color: "#ef4444" },
    { id: "saturation", label: "Saturation Zones", enabled: false, color: "#a855f7" },
    { id: "erosion", label: "Erosion Zones", enabled: false, color: "#f59e0b" },
    { id: "green_zone", label: "Green Zones", enabled: false, color: "#22c55e" },
  ]);

  const enrichedNodes = computeAllRiskScores();

  // Load Leaflet CSS
  useEffect(() => {
    if (typeof document !== "undefined" && !document.getElementById("leaflet-css")) {
      const link = document.createElement("link");
      link.id = "leaflet-css";
      link.rel = "stylesheet";
      link.href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css";
      document.head.appendChild(link);
    }
  }, []);

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    let L: any;
    const initMap = async () => {
      L = (await import("leaflet")).default;

      const map = L.map(mapRef.current!, {
        center: [-8.5, 115.3],
        zoom: 10,
        zoomControl: false,
        attributionControl: true,
      });

      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        {
          attribution: "&copy; OSM &copy; CARTO",
          subdomains: "abcd",
          maxZoom: 16,
        }
      ).addTo(map);

      L.control.zoom({ position: "bottomright" }).addTo(map);

      mapInstanceRef.current = map;

      // Create layer groups
      const nodeLayer = L.layerGroup().addTo(map);
      const edgeLayer = L.layerGroup().addTo(map);
      const floodLayer = L.layerGroup().addTo(map);
      const volcanicLayer = L.layerGroup().addTo(map);
      const saturationLayer = L.layerGroup();
      const erosionLayer = L.layerGroup();
      const greenZoneLayer = L.layerGroup();

      layerGroupsRef.current = {
        nodes: nodeLayer,
        edges: edgeLayer,
        flood: floodLayer,
        volcanic: volcanicLayer,
        saturation: saturationLayer,
        erosion: erosionLayer,
        green_zone: greenZoneLayer,
      };

      // ── Risk Zone Overlays ──
      RISK_ZONES.forEach((zone: RiskZone) => {
        const targetLayer = layerGroupsRef.current[zone.type];
        if (!targetLayer) return;

        // Outer glow
        L.circle(zone.center, {
          radius: zone.radius_km * 1000 * 1.15,
          fillColor: zone.color,
          fillOpacity: zone.severity * 0.06,
          stroke: false,
          interactive: false,
        }).addTo(targetLayer);

        // Main zone
        L.circle(zone.center, {
          radius: zone.radius_km * 1000,
          fillColor: zone.color,
          fillOpacity: zone.severity * 0.12,
          color: zone.color,
          weight: 1,
          opacity: 0.3,
          dashArray: "5 5",
          interactive: false,
        }).addTo(targetLayer);

        // Zone label
        const zoneIcon = L.divIcon({
          className: "",
          html: `<div style="
            color: ${zone.color};
            font-size: 9px;
            font-weight: 600;
            text-shadow: 0 0 6px rgba(0,0,0,0.9);
            white-space: nowrap;
            opacity: 0.6;
            font-family: monospace;
          ">${zone.name}</div>`,
          iconSize: [0, 0],
          iconAnchor: [0, 0],
        });
        L.marker(
          [zone.center[0] + zone.radius_km * 0.005, zone.center[1]],
          { icon: zoneIcon, interactive: false }
        ).addTo(targetLayer);
      });

      // ── Edges ──
      const nodeMap = Object.fromEntries(
        enrichedNodes.map((n) => [n.id, n])
      );

      BALI_EDGES.forEach((edge) => {
        const src = nodeMap[edge.source];
        const tgt = nodeMap[edge.target];
        if (!src || !tgt) return;

        const typeColors: Record<string, string> = {
          road: "#475569",
          water_supply: "#3b82f6",
          power_grid: "#eab308",
          tourism_flow: "#22c55e",
          economic_dependency: "#a855f7",
        };

        L.polyline(
          [
            [src.lat, src.lng],
            [tgt.lat, tgt.lng],
          ],
          {
            color: typeColors[edge.type] || "#475569",
            weight: edge.is_critical ? 2 : 1,
            opacity: edge.is_critical ? 0.4 : 0.15,
            dashArray: edge.type === "water_supply" ? "4 6" : edge.type === "tourism_flow" ? "2 4" : undefined,
            interactive: false,
          }
        ).addTo(edgeLayer);
      });

      // ── Nodes ──
      enrichedNodes.forEach((node) => {
        if (node.id === "mt_agung") {
          // Volcano marker
          const volcIcon = L.divIcon({
            className: "",
            html: `<div style="
              font-size: 20px;
              text-shadow: 0 0 10px rgba(239,68,68,0.8);
              pointer-events: none;
            ">🌋</div>`,
            iconSize: [24, 24],
            iconAnchor: [12, 12],
          });
          L.marker([node.lat, node.lng], { icon: volcIcon, interactive: false }).addTo(nodeLayer);
          const volcLabel = L.divIcon({
            className: "",
            html: `<div style="
              color: #ef4444;
              font-size: 11px;
              font-weight: 700;
              text-shadow: 0 0 8px rgba(0,0,0,0.9);
              white-space: nowrap;
              font-family: monospace;
            ">MT. AGUNG (3,031m)</div>`,
            iconSize: [0, 0],
            iconAnchor: [0, -18],
          });
          L.marker([node.lat, node.lng], { icon: volcLabel, interactive: false }).addTo(nodeLayer);
          return;
        }

        const risk = node.composite_risk ?? 0;
        const color = riskColor(risk);
        const r = node.is_hotspot ? 10 : 7;

        // Outer glow for hotspots
        if (node.is_hotspot) {
          L.circleMarker([node.lat, node.lng], {
            radius: r + 8,
            fillColor: color,
            fillOpacity: 0.12,
            stroke: false,
            interactive: false,
            pane: "shadowPane",
          }).addTo(nodeLayer);
        }

        // Risk ring
        L.circleMarker([node.lat, node.lng], {
          radius: r + 3,
          color: color,
          weight: 1.5,
          fillColor: "transparent",
          fillOpacity: 0,
          opacity: 0.5,
          interactive: false,
        }).addTo(nodeLayer);

        // Main marker
        const marker = L.circleMarker([node.lat, node.lng], {
          radius: r,
          fillColor: color,
          fillOpacity: 0.85,
          color: "rgba(255,255,255,0.3)",
          weight: 1,
        }).addTo(nodeLayer);

        // Label
        const labelIcon = L.divIcon({
          className: "",
          html: `<div style="
            color: ${node.is_hotspot ? "#e2e8f0" : "rgba(226,232,240,0.6)"};
            font-size: ${node.is_hotspot ? "11px" : "9px"};
            font-weight: ${node.is_hotspot ? "700" : "400"};
            text-shadow: 0 0 6px rgba(0,0,0,0.9), 0 0 12px rgba(0,0,0,0.7);
            white-space: nowrap;
            text-align: center;
            pointer-events: none;
            font-family: 'SF Mono', Monaco, monospace;
          ">${node.name}</div>`,
          iconSize: [0, 0],
          iconAnchor: [0, -(r + 10)],
        });
        L.marker([node.lat, node.lng], {
          icon: labelIcon,
          interactive: false,
        }).addTo(nodeLayer);

        // Score label
        const scoreIcon = L.divIcon({
          className: "",
          html: `<div style="
            color: ${color};
            font-size: 9px;
            font-weight: 700;
            font-family: 'SF Mono', Monaco, monospace;
            text-shadow: 0 0 4px rgba(0,0,0,0.9);
            pointer-events: none;
          ">${(risk * 100).toFixed(0)}</div>`,
          iconSize: [0, 0],
          iconAnchor: [0, r + 4],
        });
        L.marker([node.lat, node.lng], {
          icon: scoreIcon,
          interactive: false,
        }).addTo(nodeLayer);

        // Click handler
        marker.on("click", () => {
          setSelectedNode(node);
          map.flyTo([node.lat, node.lng], 12, { duration: 0.5 });
        });

        // Hover
        marker.on("mouseover", () => {
          marker.setStyle({ fillOpacity: 1, weight: 2.5, color: "#fff" });
        });
        marker.on("mouseout", () => {
          marker.setStyle({
            fillOpacity: 0.85,
            weight: 1,
            color: "rgba(255,255,255,0.3)",
          });
        });
      });
    };

    initMap();

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Toggle layers
  const toggleLayer = (id: string) => {
    setLayers((prev) =>
      prev.map((l) => (l.id === id ? { ...l, enabled: !l.enabled } : l))
    );
    const map = mapInstanceRef.current;
    const group = layerGroupsRef.current[id];
    if (!map || !group) return;
    const layer = layers.find((l) => l.id === id);
    if (layer?.enabled) {
      map.removeLayer(group);
    } else {
      map.addLayer(group);
    }
  };

  const vuln = selectedNode
    ? computeVulnerability(selectedNode)
    : null;

  return (
    <div className="relative rounded-xl overflow-hidden border border-[#1e1e2e]" style={{ height: 600 }}>
      <div ref={mapRef} className="absolute inset-0 z-0" />

      {/* Title overlay */}
      <div className="absolute top-4 left-4 z-[1000] bg-[#0a0e1a]/92 backdrop-blur-xl border border-indigo-500/20 rounded-xl px-5 py-3 shadow-[0_0_40px_rgba(99,102,241,0.1)]">
        <h2
          className="text-base font-bold"
          style={{
            background: "linear-gradient(135deg, #818cf8, #6366f1)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Bali Risk Intelligence Map
        </h2>
        <p className="text-[10px] text-slate-400 font-mono mt-0.5">
          {enrichedNodes.filter((n) => n.id !== "mt_agung").length} nodes &middot;{" "}
          {BALI_EDGES.length} connections &middot;{" "}
          {RISK_ZONES.length} risk zones
        </p>
      </div>

      {/* Layer Controls */}
      <div className="absolute top-4 right-4 z-[1000] bg-[#0a0e1a]/92 backdrop-blur-xl border border-indigo-500/20 rounded-xl px-4 py-3 shadow-[0_0_40px_rgba(99,102,241,0.1)]">
        <h3 className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
          Layers
        </h3>
        <div className="space-y-1.5">
          {layers.map((layer) => (
            <label
              key={layer.id}
              className="flex items-center gap-2 cursor-pointer text-xs text-slate-300 hover:text-white transition-colors"
            >
              <input
                type="checkbox"
                checked={layer.enabled}
                onChange={() => toggleLayer(layer.id)}
                className="rounded border-slate-600 bg-transparent text-indigo-500 focus:ring-indigo-500/30 h-3 w-3"
              />
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: layer.color }}
              />
              {layer.label}
            </label>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-[1000] bg-[#0a0e1a]/92 backdrop-blur-xl border border-indigo-500/20 rounded-xl px-4 py-3 shadow-[0_0_40px_rgba(99,102,241,0.1)]">
        <h3 className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
          Risk Level
        </h3>
        <div className="flex items-center gap-3">
          {[
            { label: "Low", color: "#22c55e" },
            { label: "Medium", color: "#eab308" },
            { label: "High", color: "#f97316" },
            { label: "Critical", color: "#ef4444" },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-1">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: item.color }}
              />
              <span className="text-[10px] text-slate-400 font-mono">
                {item.label}
              </span>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-3 mt-1.5 pt-1.5 border-t border-slate-700/50">
          {[
            { label: "Road", style: "solid", color: "#475569" },
            { label: "Water", style: "dashed", color: "#3b82f6" },
            { label: "Tourism", style: "dotted", color: "#22c55e" },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-1">
              <div
                className="w-4 h-0.5"
                style={{
                  backgroundColor: item.color,
                  borderTop:
                    item.style === "dashed"
                      ? `2px dashed ${item.color}`
                      : item.style === "dotted"
                        ? `2px dotted ${item.color}`
                        : "none",
                }}
              />
              <span className="text-[10px] text-slate-400 font-mono">
                {item.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Selected Node Detail */}
      {selectedNode && selectedNode.id !== "mt_agung" && vuln && (
        <div className="absolute bottom-4 right-4 z-[1000] bg-[#0a0e1a]/95 backdrop-blur-xl border border-indigo-500/30 rounded-xl px-5 py-4 shadow-[0_0_40px_rgba(99,102,241,0.15)] w-[300px]">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-bold text-slate-200">
              {selectedNode.name}
            </h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-slate-500 hover:text-slate-300 text-xs"
            >
              ✕
            </button>
          </div>

          {/* Composite score */}
          <div className="flex items-center gap-2 mb-3">
            <span
              className="text-2xl font-bold font-mono"
              style={{
                color: riskColor(selectedNode.composite_risk ?? 0),
              }}
            >
              {((selectedNode.composite_risk ?? 0) * 100).toFixed(0)}
            </span>
            <div className="text-[10px] text-slate-400">
              <div>
                composite risk &middot;{" "}
                <span
                  style={{
                    color: riskColor(selectedNode.composite_risk ?? 0),
                  }}
                >
                  {riskLabel(selectedNode.composite_risk ?? 0)}
                </span>
              </div>
              <div>
                Kelly fraction:{" "}
                <span className="text-slate-300 font-mono">
                  {((selectedNode.kelly_fraction ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          {/* Vulnerability breakdown bars */}
          <div className="space-y-1.5">
            {(
              [
                { key: "legal", label: "Legal", weight: "25%" },
                { key: "climate", label: "Climate", weight: "15%" },
                { key: "market", label: "Market", weight: "15%" },
                { key: "regulatory", label: "Regulatory", weight: "15%" },
                { key: "infrastructure", label: "Infra", weight: "10%" },
                { key: "political", label: "Political", weight: "10%" },
                { key: "financial", label: "Financial", weight: "10%" },
              ] as const
            ).map(({ key, label, weight }) => {
              const val = vuln[key];
              return (
                <div key={key} className="flex items-center gap-2">
                  <span className="text-[9px] text-slate-500 w-16 text-right font-mono">
                    {label}
                  </span>
                  <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${val * 100}%`,
                        backgroundColor: riskColor(val),
                      }}
                    />
                  </div>
                  <span className="text-[9px] text-slate-500 font-mono w-7">
                    {(val * 100).toFixed(0)}
                  </span>
                  <span className="text-[8px] text-slate-600 font-mono w-6">
                    {weight}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Cascade & Bayesian */}
          <div className="mt-3 pt-2 border-t border-slate-700/50 grid grid-cols-2 gap-2 text-[10px]">
            <div>
              <span className="text-slate-500">Cascade severity</span>
              <div className="text-slate-300 font-mono font-bold">
                {((selectedNode.cascade_severity ?? 0) * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <span className="text-slate-500">Bayesian P(reg)</span>
              <div className="text-slate-300 font-mono font-bold">
                {((selectedNode.bayesian_posterior ?? 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
