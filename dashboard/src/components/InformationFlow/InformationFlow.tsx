"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import type { MarketFlowInfo } from "@/types/causal";

// ── Constants ─────────────────────────────────────────────

const SOURCE_COLOR = "#00d4aa";
const DERIVATIVE_COLOR = "#ff4466";
const NEUTRAL_COLOR = "#6b7280";
const EDGE_COLOR = "#374151";
const NODE_RADIUS = 28;
const ARROW_SIZE = 6;

// ── Props ─────────────────────────────────────────────────

interface InformationFlowProps {
  /** NxN transfer entropy matrix */
  flowMatrix: number[][];
  /** Market names/labels for display (same order as matrix) */
  marketNames: string[];
  /** Market IDs (same order as matrix) */
  marketIds: string[];
  /** Indices of source markets (net outflow > 0) */
  sourceMarkets: number[];
  /** Indices of derivative markets (net inflow > outflow) */
  derivativeMarkets: number[];
  /** Optional: max TE for normalization */
  maxTe?: number;
  /** Width of the SVG */
  width?: number;
  /** Height of the SVG */
  height?: number;
  /** Minimum TE threshold to show an edge */
  minThreshold?: number;
}

// ── Force layout helper ───────────────────────────────────

interface LayoutNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  idx: number;
  role: "source" | "derivative" | "neutral";
}

function layoutNodes(
  n: number,
  sourceIndices: Set<number>,
  derivativeIndices: Set<number>,
  flowMatrix: number[][],
  width: number,
  height: number,
): LayoutNode[] {
  // Place sources on the left, derivatives on the right
  const nodes: LayoutNode[] = [];
  const sources = Array.from(sourceIndices);
  const derivatives = Array.from(derivativeIndices);
  const neutrals: number[] = [];

  for (let i = 0; i < n; i++) {
    if (!sourceIndices.has(i) && !derivativeIndices.has(i)) {
      neutrals.push(i);
    }
  }

  // Layout: sources left-column, neutrals center, derivatives right
  const colWidth = width / 4;

  sources.forEach((idx, i) => {
    const ySpacing = height / (sources.length + 1);
    nodes[idx] = {
      x: colWidth * 0.8,
      y: ySpacing * (i + 1),
      vx: 0,
      vy: 0,
      idx,
      role: "source",
    };
  });

  derivatives.forEach((idx, i) => {
    const ySpacing = height / (derivatives.length + 1);
    nodes[idx] = {
      x: colWidth * 3.2,
      y: ySpacing * (i + 1),
      vx: 0,
      vy: 0,
      idx,
      role: "derivative",
    };
  });

  neutrals.forEach((idx, i) => {
    const ySpacing = height / (neutrals.length + 1);
    nodes[idx] = {
      x: colWidth * 2,
      y: ySpacing * (i + 1),
      vx: 0,
      vy: 0,
      idx,
      role: "neutral",
    };
  });

  // Simple repulsion pass to avoid overlap
  for (let iter = 0; iter < 50; iter++) {
    const alpha = 1 - iter / 50;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!nodes[i] || !nodes[j]) continue;
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        if (dist < NODE_RADIUS * 3) {
          const force = ((NODE_RADIUS * 3 - dist) / dist) * alpha * 2;
          nodes[i].x += (dx / dist) * force;
          nodes[i].y += (dy / dist) * force;
          nodes[j].x -= (dx / dist) * force;
          nodes[j].y -= (dy / dist) * force;
        }
      }
    }

    // Keep within bounds
    for (const node of nodes) {
      if (!node) continue;
      node.x = Math.max(NODE_RADIUS + 10, Math.min(width - NODE_RADIUS - 10, node.x));
      node.y = Math.max(NODE_RADIUS + 10, Math.min(height - NODE_RADIUS - 10, node.y));
    }
  }

  return nodes;
}

// ── Component ─────────────────────────────────────────────

export function InformationFlow({
  flowMatrix,
  marketNames,
  marketIds,
  sourceMarkets,
  derivativeMarkets,
  maxTe,
  width = 800,
  height = 450,
  minThreshold = 0.001,
}: InformationFlowProps) {
  const [animOffset, setAnimOffset] = useState(0);
  const animRef = useRef<number | null>(null);
  const [hoveredNode, setHoveredNode] = useState<number | null>(null);

  const n = marketNames.length;
  const sourceSet = useMemo(() => new Set(sourceMarkets), [sourceMarkets]);
  const derivativeSet = useMemo(() => new Set(derivativeMarkets), [derivativeMarkets]);

  // Layout nodes
  const nodes = useMemo(
    () => layoutNodes(n, sourceSet, derivativeSet, flowMatrix, width, height),
    [n, sourceSet, derivativeSet, flowMatrix, width, height],
  );

  // Normalize TE values
  const maxTeValue = useMemo(() => {
    if (maxTe && maxTe > 0) return maxTe;
    let max = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j && flowMatrix[i]?.[j] > max) {
          max = flowMatrix[i][j];
        }
      }
    }
    return Math.max(max, 0.001);
  }, [flowMatrix, n, maxTe]);

  // Build visible edges
  const edges = useMemo(() => {
    const result: {
      from: number;
      to: number;
      te: number;
      normalized: number;
    }[] = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const te = flowMatrix[i]?.[j] ?? 0;
        if (te >= minThreshold) {
          result.push({
            from: i,
            to: j,
            te,
            normalized: te / maxTeValue,
          });
        }
      }
    }
    // Sort by strength so stronger edges render on top
    result.sort((a, b) => a.normalized - b.normalized);
    return result;
  }, [flowMatrix, n, maxTeValue, minThreshold]);

  // Animate flow particles
  useEffect(() => {
    let lastTime = 0;
    const animate = (time: number) => {
      if (time - lastTime > 40) {
        setAnimOffset((prev) => (prev + 1) % 30);
        lastTime = time;
      }
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, []);

  if (n === 0 || flowMatrix.length === 0) {
    return (
      <div className="flex items-center justify-center h-[450px] text-muted-foreground">
        No information flow data available. Run transfer entropy analysis first.
      </div>
    );
  }

  // Get edges connected to hovered node
  const hoveredEdges = new Set<string>();
  const hoveredNodeIds = new Set<number>();
  if (hoveredNode !== null) {
    hoveredNodeIds.add(hoveredNode);
    for (const e of edges) {
      if (e.from === hoveredNode || e.to === hoveredNode) {
        hoveredEdges.add(`${e.from}-${e.to}`);
        hoveredNodeIds.add(e.from);
        hoveredNodeIds.add(e.to);
      }
    }
  }

  return (
    <div className="space-y-3">
      {/* Legend */}
      <div className="flex items-center gap-6 text-xs">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: SOURCE_COLOR }} />
          Source (net outflow)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: DERIVATIVE_COLOR }} />
          Derivative (net inflow)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: NEUTRAL_COLOR }} />
          Neutral
        </span>
        <span className="text-muted-foreground ml-auto">
          Arrow thickness = transfer entropy strength
        </span>
      </div>

      {/* SVG */}
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full rounded-lg bg-[#0d0d14] border border-[#1e1e2e]"
        style={{ maxHeight: `${height}px` }}
      >
        <defs>
          <marker
            id="flow-arrow"
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth={ARROW_SIZE}
            markerHeight={ARROW_SIZE}
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280" opacity={0.7} />
          </marker>
          <marker
            id="flow-arrow-highlight"
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth={ARROW_SIZE + 2}
            markerHeight={ARROW_SIZE + 2}
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="white" opacity={0.8} />
          </marker>
        </defs>

        {/* Edges with animated flow */}
        {edges.map((edge) => {
          const fromNode = nodes[edge.from];
          const toNode = nodes[edge.to];
          if (!fromNode || !toNode) return null;

          const edgeKey = `${edge.from}-${edge.to}`;
          const isHighlighted = hoveredEdges.size > 0 && hoveredEdges.has(edgeKey);
          const isDimmed = hoveredEdges.size > 0 && !isHighlighted;

          const dx = toNode.x - fromNode.x;
          const dy = toNode.y - fromNode.y;
          const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
          const ux = dx / dist;
          const uy = dy / dist;

          const x1 = fromNode.x + ux * (NODE_RADIUS + 2);
          const y1 = fromNode.y + uy * (NODE_RADIUS + 2);
          const x2 = toNode.x - ux * (NODE_RADIUS + ARROW_SIZE + 2);
          const y2 = toNode.y - uy * (NODE_RADIUS + ARROW_SIZE + 2);

          const strokeWidth = 1 + edge.normalized * 5;
          const dashLength = 6 + edge.normalized * 4;

          return (
            <g key={edgeKey}>
              {/* Background line */}
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={isHighlighted ? "white" : EDGE_COLOR}
                strokeWidth={strokeWidth}
                opacity={isDimmed ? 0.05 : isHighlighted ? 0.8 : 0.25}
                markerEnd={`url(#flow-arrow${isHighlighted ? "-highlight" : ""})`}
              />
              {/* Animated flow particles */}
              {!isDimmed && (
                <line
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={
                    sourceSet.has(edge.from)
                      ? SOURCE_COLOR
                      : derivativeSet.has(edge.from)
                        ? DERIVATIVE_COLOR
                        : NEUTRAL_COLOR
                  }
                  strokeWidth={Math.max(strokeWidth * 0.6, 1)}
                  opacity={isHighlighted ? 0.9 : 0.5}
                  strokeDasharray={`${dashLength} ${dashLength * 2}`}
                  strokeDashoffset={-animOffset * 2}
                  markerEnd={`url(#flow-arrow${isHighlighted ? "-highlight" : ""})`}
                />
              )}
            </g>
          );
        })}

        {/* Nodes */}
        {nodes.map((node) => {
          if (!node) return null;
          const isHighlighted = hoveredNodeIds.size > 0 && hoveredNodeIds.has(node.idx);
          const isDimmed = hoveredNodeIds.size > 0 && !isHighlighted;
          const color =
            node.role === "source"
              ? SOURCE_COLOR
              : node.role === "derivative"
                ? DERIVATIVE_COLOR
                : NEUTRAL_COLOR;
          const name = marketNames[node.idx] ?? `Market ${node.idx}`;

          return (
            <g
              key={node.idx}
              transform={`translate(${node.x}, ${node.y})`}
              className="cursor-pointer"
              onMouseEnter={() => setHoveredNode(node.idx)}
              onMouseLeave={() => setHoveredNode(null)}
            >
              {/* Glow ring */}
              <circle
                r={NODE_RADIUS + 3}
                fill="none"
                stroke={color}
                strokeWidth={1.5}
                opacity={isDimmed ? 0.05 : isHighlighted ? 0.7 : 0.2}
              />
              {/* Main circle */}
              <circle
                r={NODE_RADIUS}
                fill="#111118"
                stroke={color}
                strokeWidth={2}
                opacity={isDimmed ? 0.2 : 1}
              />
              <circle
                r={NODE_RADIUS - 3}
                fill={color}
                opacity={isDimmed ? 0.03 : 0.12}
              />
              {/* Label */}
              <text
                textAnchor="middle"
                dy={-2}
                fill={isDimmed ? "#333" : "white"}
                fontSize={8}
                fontWeight={500}
              >
                {name.length > 10 ? name.slice(0, 9) + "\u2026" : name}
              </text>
              {/* Role indicator */}
              <text
                textAnchor="middle"
                dy={10}
                fill={isDimmed ? "#222" : color}
                fontSize={7}
                fontFamily="monospace"
              >
                {node.role === "source"
                  ? "\u25B2 SRC"
                  : node.role === "derivative"
                    ? "\u25BC DRV"
                    : "\u2500 NEU"}
              </text>
            </g>
          );
        })}

        {/* Tooltip for hovered node */}
        {hoveredNode !== null && nodes[hoveredNode] && (
          <g>
            <rect
              x={Math.min(nodes[hoveredNode].x + NODE_RADIUS + 8, width - 180)}
              y={nodes[hoveredNode].y - 30}
              width={170}
              height={48}
              rx={6}
              fill="#1e1e2e"
              stroke="#2a2a3e"
              strokeWidth={1}
            />
            <text
              x={Math.min(nodes[hoveredNode].x + NODE_RADIUS + 16, width - 172)}
              y={nodes[hoveredNode].y - 12}
              fill="white"
              fontSize={10}
              fontWeight={500}
            >
              {marketNames[hoveredNode]?.slice(0, 22) ?? `Market ${hoveredNode}`}
            </text>
            <text
              x={Math.min(nodes[hoveredNode].x + NODE_RADIUS + 16, width - 172)}
              y={nodes[hoveredNode].y + 4}
              fill="#6b7280"
              fontSize={9}
              fontFamily="monospace"
            >
              {(() => {
                const outflow = edges
                  .filter((e) => e.from === hoveredNode)
                  .reduce((sum, e) => sum + e.te, 0);
                const inflow = edges
                  .filter((e) => e.to === hoveredNode)
                  .reduce((sum, e) => sum + e.te, 0);
                return `Out: ${outflow.toFixed(4)}  In: ${inflow.toFixed(4)}`;
              })()}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}

// ── Skeleton ──────────────────────────────────────────────

export function InformationFlowSkeleton() {
  return (
    <div className="space-y-3">
      <div className="flex gap-6">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="h-4 w-24 bg-[#1e1e2e] rounded animate-pulse" />
        ))}
      </div>
      <div className="h-[450px] rounded-lg bg-[#1e1e2e] animate-pulse" />
    </div>
  );
}
