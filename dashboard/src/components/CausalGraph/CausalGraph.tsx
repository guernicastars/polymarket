"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { CausalNode, CausalEdge, CausalEdgeMethod } from "@/types/causal";

// ── Constants ─────────────────────────────────────────────

const METHOD_COLORS: Record<CausalEdgeMethod, string> = {
  granger: "#6366f1",        // indigo
  pc: "#00d4aa",             // teal/green
  transfer_entropy: "#f59e0b", // amber
};

const METHOD_LABELS: Record<CausalEdgeMethod, string> = {
  granger: "Granger",
  pc: "PC Algorithm",
  transfer_entropy: "Transfer Entropy",
};

const ROLE_COLORS: Record<string, string> = {
  source: "#00d4aa",
  derivative: "#ff4466",
  neutral: "#6b7280",
};

const NODE_RADIUS = 24;
const ARROW_SIZE = 8;

// ── Force simulation helpers ──────────────────────────────

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function runSimulation(
  nodes: CausalNode[],
  edges: CausalEdge[],
  width: number,
  height: number,
  iterations: number = 120,
): CausalNode[] {
  // Initialize positions in a circle if not set
  const simNodes = nodes.map((n, i) => ({
    ...n,
    x: n.x ?? width / 2 + (width / 3) * Math.cos((2 * Math.PI * i) / nodes.length),
    y: n.y ?? height / 2 + (height / 3) * Math.sin((2 * Math.PI * i) / nodes.length),
    vx: 0,
    vy: 0,
  }));

  const nodeMap = new Map(simNodes.map((n) => [n.id, n]));

  for (let iter = 0; iter < iterations; iter++) {
    const alpha = 1 - iter / iterations; // cooling
    const k = Math.sqrt((width * height) / Math.max(simNodes.length, 1));

    // Repulsive force between all pairs
    for (let i = 0; i < simNodes.length; i++) {
      for (let j = i + 1; j < simNodes.length; j++) {
        const a = simNodes[i];
        const b = simNodes[j];
        let dx = a.x! - b.x!;
        let dy = a.y! - b.y!;
        const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        const force = (k * k) / dist;
        dx = (dx / dist) * force * alpha * 0.15;
        dy = (dy / dist) * force * alpha * 0.15;
        a.vx! += dx;
        a.vy! += dy;
        b.vx! -= dx;
        b.vy! -= dy;
      }
    }

    // Attractive force along edges
    for (const edge of edges) {
      const a = nodeMap.get(edge.source);
      const b = nodeMap.get(edge.target);
      if (!a || !b) continue;
      let dx = b.x! - a.x!;
      let dy = b.y! - a.y!;
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const force = (dist * dist) / k;
      const strength = 0.05 + edge.strength * 0.1;
      dx = (dx / dist) * force * alpha * strength;
      dy = (dy / dist) * force * alpha * strength;
      a.vx! += dx;
      a.vy! += dy;
      b.vx! -= dx;
      b.vy! -= dy;
    }

    // Center gravity
    for (const node of simNodes) {
      node.vx! += (width / 2 - node.x!) * alpha * 0.01;
      node.vy! += (height / 2 - node.y!) * alpha * 0.01;
    }

    // Apply velocities with damping
    for (const node of simNodes) {
      node.vx! *= 0.8;
      node.vy! *= 0.8;
      node.x! += node.vx!;
      node.y! += node.vy!;
      // Keep within bounds
      node.x = clamp(node.x!, NODE_RADIUS + 10, width - NODE_RADIUS - 10);
      node.y = clamp(node.y!, NODE_RADIUS + 10, height - NODE_RADIUS - 10);
    }
  }

  return simNodes;
}

// ── Props ─────────────────────────────────────────────────

interface CausalGraphProps {
  nodes: CausalNode[];
  edges: CausalEdge[];
  onNodeClick?: (node: CausalNode) => void;
  onEdgeClick?: (edge: CausalEdge) => void;
  width?: number;
  height?: number;
}

// ── Component ─────────────────────────────────────────────

export function CausalGraph({
  nodes,
  edges,
  onNodeClick,
  onEdgeClick,
  width = 800,
  height = 500,
}: CausalGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [simulatedNodes, setSimulatedNodes] = useState<CausalNode[]>([]);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [enabledMethods, setEnabledMethods] = useState<Set<CausalEdgeMethod>>(
    new Set(["granger", "pc", "transfer_entropy"]),
  );

  // Run simulation when nodes/edges change
  useEffect(() => {
    if (nodes.length === 0) return;
    const result = runSimulation(nodes, edges, width, height);
    setSimulatedNodes(result);
  }, [nodes, edges, width, height]);

  const nodeMap = new Map(simulatedNodes.map((n) => [n.id, n]));

  const filteredEdges = edges.filter((e) => enabledMethods.has(e.method));

  const toggleMethod = useCallback((method: CausalEdgeMethod) => {
    setEnabledMethods((prev) => {
      const next = new Set(prev);
      if (next.has(method)) {
        next.delete(method);
      } else {
        next.add(method);
      }
      return next;
    });
  }, []);

  const handleNodeClick = useCallback(
    (node: CausalNode) => {
      setSelectedNode((prev) => (prev === node.id ? null : node.id));
      onNodeClick?.(node);
    },
    [onNodeClick],
  );

  // Get edges connected to hovered or selected node for highlighting
  const highlightNodeId = hoveredNode ?? selectedNode;
  const connectedEdgeKeys = new Set<string>();
  const connectedNodeIds = new Set<string>();
  if (highlightNodeId) {
    connectedNodeIds.add(highlightNodeId);
    for (const e of filteredEdges) {
      if (e.source === highlightNodeId || e.target === highlightNodeId) {
        connectedEdgeKeys.add(`${e.source}-${e.target}-${e.method}`);
        connectedNodeIds.add(e.source);
        connectedNodeIds.add(e.target);
      }
    }
  }

  if (nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-[500px] text-muted-foreground">
        No causal relationships discovered. Run causal analysis on selected markets.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Method filter legend */}
      <div className="flex flex-wrap items-center gap-4 text-xs">
        <span className="text-muted-foreground font-medium">Edge type:</span>
        {(Object.keys(METHOD_COLORS) as CausalEdgeMethod[]).map((method) => {
          const enabled = enabledMethods.has(method);
          const count = edges.filter((e) => e.method === method).length;
          return (
            <button
              key={method}
              onClick={() => toggleMethod(method)}
              className={`flex items-center gap-1.5 px-2 py-1 rounded transition-colors ${
                enabled
                  ? "bg-[#1e1e2e]"
                  : "bg-transparent opacity-40"
              }`}
            >
              <span
                className="inline-block w-3 h-0.5 rounded-full"
                style={{ backgroundColor: METHOD_COLORS[method] }}
              />
              <span className={enabled ? "text-foreground" : "text-muted-foreground"}>
                {METHOD_LABELS[method]} ({count})
              </span>
            </button>
          );
        })}
        {/* Node role legend */}
        <span className="ml-4 text-muted-foreground font-medium">Node role:</span>
        {(["source", "derivative", "neutral"] as const).map((role) => (
          <span key={role} className="flex items-center gap-1.5">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: ROLE_COLORS[role] }}
            />
            <span className="text-muted-foreground capitalize">{role}</span>
          </span>
        ))}
      </div>

      {/* SVG Graph */}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full rounded-lg bg-[#0d0d14] border border-[#1e1e2e]"
        style={{ maxHeight: `${height}px` }}
      >
        <defs>
          {/* Arrowhead markers per method */}
          {(Object.keys(METHOD_COLORS) as CausalEdgeMethod[]).map((method) => (
            <marker
              key={method}
              id={`arrow-${method}`}
              viewBox="0 0 10 10"
              refX="10"
              refY="5"
              markerWidth={ARROW_SIZE}
              markerHeight={ARROW_SIZE}
              orient="auto-start-reverse"
            >
              <path
                d="M 0 0 L 10 5 L 0 10 z"
                fill={METHOD_COLORS[method]}
                opacity={0.8}
              />
            </marker>
          ))}
          {/* Highlighted arrowhead */}
          <marker
            id="arrow-highlight"
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth={ARROW_SIZE + 2}
            markerHeight={ARROW_SIZE + 2}
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="white" opacity={0.9} />
          </marker>
        </defs>

        {/* Edges */}
        {filteredEdges.map((edge) => {
          const sourceNode = nodeMap.get(edge.source);
          const targetNode = nodeMap.get(edge.target);
          if (!sourceNode || !targetNode) return null;

          const edgeKey = `${edge.source}-${edge.target}-${edge.method}`;
          const isHighlighted = connectedEdgeKeys.size > 0 && connectedEdgeKeys.has(edgeKey);
          const isDimmed = connectedEdgeKeys.size > 0 && !isHighlighted;

          // Calculate edge endpoints (offset from node center by radius)
          const dx = targetNode.x! - sourceNode.x!;
          const dy = targetNode.y! - sourceNode.y!;
          const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
          const ux = dx / dist;
          const uy = dy / dist;

          const x1 = sourceNode.x! + ux * (NODE_RADIUS + 2);
          const y1 = sourceNode.y! + uy * (NODE_RADIUS + 2);
          const x2 = targetNode.x! - ux * (NODE_RADIUS + ARROW_SIZE + 2);
          const y2 = targetNode.y! - uy * (NODE_RADIUS + ARROW_SIZE + 2);

          const strokeWidth = 1 + edge.strength * 4;

          return (
            <g key={edgeKey}>
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={isHighlighted ? "white" : METHOD_COLORS[edge.method]}
                strokeWidth={isHighlighted ? strokeWidth + 1 : strokeWidth}
                opacity={isDimmed ? 0.1 : isHighlighted ? 1 : 0.6}
                markerEnd={
                  edge.type !== "undirected"
                    ? `url(#arrow-${isHighlighted ? "highlight" : edge.method})`
                    : undefined
                }
                className="cursor-pointer transition-opacity"
                onClick={(e) => {
                  e.stopPropagation();
                  onEdgeClick?.(edge);
                }}
                onMouseEnter={() => setHoveredEdge(edgeKey)}
                onMouseLeave={() => setHoveredEdge(null)}
              />
              {/* Edge label on hover */}
              {hoveredEdge === edgeKey && (
                <g>
                  <rect
                    x={(x1 + x2) / 2 - 40}
                    y={(y1 + y2) / 2 - 12}
                    width={80}
                    height={24}
                    rx={4}
                    fill="#1e1e2e"
                    stroke="#2a2a3e"
                    strokeWidth={1}
                  />
                  <text
                    x={(x1 + x2) / 2}
                    y={(y1 + y2) / 2 + 4}
                    textAnchor="middle"
                    fill="white"
                    fontSize={10}
                    fontFamily="monospace"
                  >
                    {METHOD_LABELS[edge.method]} ({(edge.strength * 100).toFixed(0)}%)
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* Nodes */}
        {simulatedNodes.map((node) => {
          const isHighlighted = connectedNodeIds.size > 0 && connectedNodeIds.has(node.id);
          const isDimmed = connectedNodeIds.size > 0 && !isHighlighted;
          const isSelected = selectedNode === node.id;
          const role = node.role ?? "neutral";
          const fillColor = ROLE_COLORS[role];

          return (
            <g
              key={node.id}
              transform={`translate(${node.x}, ${node.y})`}
              className="cursor-pointer"
              onClick={() => handleNodeClick(node)}
              onMouseEnter={() => setHoveredNode(node.id)}
              onMouseLeave={() => setHoveredNode(null)}
            >
              {/* Selection ring */}
              {isSelected && (
                <circle
                  r={NODE_RADIUS + 5}
                  fill="none"
                  stroke="white"
                  strokeWidth={2}
                  opacity={0.6}
                />
              )}
              {/* Outer ring (glow) */}
              <circle
                r={NODE_RADIUS + 2}
                fill="none"
                stroke={fillColor}
                strokeWidth={1.5}
                opacity={isDimmed ? 0.1 : isHighlighted ? 0.8 : 0.3}
              />
              {/* Main circle */}
              <circle
                r={NODE_RADIUS}
                fill="#111118"
                stroke={fillColor}
                strokeWidth={2}
                opacity={isDimmed ? 0.2 : 1}
              />
              {/* Inner fill with role color */}
              <circle
                r={NODE_RADIUS - 3}
                fill={fillColor}
                opacity={isDimmed ? 0.05 : 0.15}
              />
              {/* Label */}
              <text
                textAnchor="middle"
                dy={-3}
                fill={isDimmed ? "#333" : "white"}
                fontSize={9}
                fontWeight={500}
              >
                {node.label.length > 8 ? node.label.slice(0, 7) + "\u2026" : node.label}
              </text>
              {/* Price subtext */}
              {node.price !== undefined && (
                <text
                  textAnchor="middle"
                  dy={9}
                  fill={isDimmed ? "#222" : "#6b7280"}
                  fontSize={8}
                  fontFamily="monospace"
                >
                  {(node.price * 100).toFixed(0)}\u00A2
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Selected node detail panel */}
      {selectedNode && (
        <SelectedNodePanel
          node={nodeMap.get(selectedNode)!}
          edges={filteredEdges}
          nodeMap={nodeMap}
        />
      )}
    </div>
  );
}

// ── Node detail panel ─────────────────────────────────────

function SelectedNodePanel({
  node,
  edges,
  nodeMap,
}: {
  node: CausalNode;
  edges: CausalEdge[];
  nodeMap: Map<string, CausalNode>;
}) {
  const outgoing = edges.filter((e) => e.source === node.id);
  const incoming = edges.filter((e) => e.target === node.id);

  return (
    <div className="bg-[#111118] border border-[#1e1e2e] rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="font-medium text-sm">{node.label}</h4>
          {node.question && (
            <p className="text-xs text-muted-foreground mt-0.5 max-w-lg truncate">
              {node.question}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {node.price !== undefined && (
            <span className="font-mono text-sm">{(node.price * 100).toFixed(1)}\u00A2</span>
          )}
          <span
            className="text-xs px-2 py-0.5 rounded-full capitalize"
            style={{
              backgroundColor: `${ROLE_COLORS[node.role ?? "neutral"]}20`,
              color: ROLE_COLORS[node.role ?? "neutral"],
            }}
          >
            {node.role ?? "neutral"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 text-xs">
        {/* Outgoing */}
        <div>
          <span className="text-muted-foreground font-medium">
            Causes ({outgoing.length})
          </span>
          {outgoing.length === 0 ? (
            <p className="text-muted-foreground mt-1">No outgoing causal edges</p>
          ) : (
            <ul className="mt-1 space-y-1">
              {outgoing.map((e) => {
                const target = nodeMap.get(e.target);
                return (
                  <li key={`${e.target}-${e.method}`} className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-0.5 rounded-full inline-block"
                      style={{ backgroundColor: METHOD_COLORS[e.method] }}
                    />
                    <span>{target?.label ?? e.target}</span>
                    <span className="text-muted-foreground font-mono">
                      ({(e.strength * 100).toFixed(0)}%)
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {/* Incoming */}
        <div>
          <span className="text-muted-foreground font-medium">
            Caused by ({incoming.length})
          </span>
          {incoming.length === 0 ? (
            <p className="text-muted-foreground mt-1">No incoming causal edges</p>
          ) : (
            <ul className="mt-1 space-y-1">
              {incoming.map((e) => {
                const source = nodeMap.get(e.source);
                return (
                  <li key={`${e.source}-${e.method}`} className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-0.5 rounded-full inline-block"
                      style={{ backgroundColor: METHOD_COLORS[e.method] }}
                    />
                    <span>{source?.label ?? e.source}</span>
                    <span className="text-muted-foreground font-mono">
                      ({(e.strength * 100).toFixed(0)}%)
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
