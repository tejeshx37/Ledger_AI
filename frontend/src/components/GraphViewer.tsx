import { useEffect, useRef, useState } from "react";
import cytoscape, { type Core } from "cytoscape";
import type { Evidence, EvidenceEdge } from "../types";
import { GitBranch, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";

interface GraphViewerProps {
  evidence: Evidence | null;
  highlightedNodeId?: string | null;
  onNodeClick?: (nodeId: string) => void;
}

const ROLE_COLORS: Record<string, string> = {
  source: "#10b981",      // emerald
  mule: "#f59e0b",        // amber
  aggregator: "#8b5cf6",  // violet
  destination: "#f43f5e", // rose
};

function buildCyElements(evidence: Evidence, visibleEdgeIds: Set<string>) {
  const nodes = evidence.nodes.map((n) => ({
    data: {
      id: n.id,
      label: n.id,
      role: n.role,
      color: ROLE_COLORS[n.role] ?? "#64748b",
    },
  }));

  const edges = evidence.edges
    .filter((e) => visibleEdgeIds.has(e.id))
    .map((e) => ({
      data: {
        id: e.id,
        source: e.src,
        target: e.dst,
        label: `$${(e.amount / 1000).toFixed(1)}k`,
        weight: Math.max(1, Math.min(8, e.amount / 15000)),
      },
    }));

  return [...nodes, ...edges];
}

export default function GraphViewer({
  evidence,
  highlightedNodeId,
  onNodeClick,
}: GraphViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [sliderValue, setSliderValue] = useState(100);

  // Sorted edges by timestamp for timeline scrubber
  const sortedEdges: EvidenceEdge[] = evidence
    ? [...evidence.edges].sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      )
    : [];

  const visibleCount = Math.max(
    1,
    Math.ceil((sliderValue / 100) * sortedEdges.length)
  );
  const visibleEdgeIds = new Set(sortedEdges.slice(0, visibleCount).map((e) => e.id));

  // Initialise / re-render Cytoscape when evidence changes
  useEffect(() => {
    if (!containerRef.current || !evidence) return;

    const cy = cytoscape({
      container: containerRef.current,
      elements: buildCyElements(evidence, visibleEdgeIds),
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            "border-width": 2,
            "border-color": "#1e293b",
            label: "data(label)",
            "font-size": "11px",
            color: "#f1f5f9",
            "text-valign": "bottom",
            "text-margin-y": 6,
            "text-outline-color": "#0a0f1e",
            "text-outline-width": 2,
            width: 40,
            height: 40,
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-color": "#38bdf8",
            "border-width": 3,
            "background-opacity": 0.9,
          },
        },
        {
          selector: "edge",
          style: {
            width: "data(weight)",
            "line-color": "#334155",
            "target-arrow-color": "#475569",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": "10px",
            color: "#94a3b8",
            "text-outline-color": "#0a0f1e",
            "text-outline-width": 2,
            "text-rotation": "autorotate",
          },
        },
      ],
      layout: {
        name: "breadthfirst",
        directed: true,
        padding: 24,
        spacingFactor: 1.5,
      },
      userZoomingEnabled: true,
      userPanningEnabled: true,
    });

    cy.on("tap", "node", (evt) => {
      onNodeClick?.(evt.target.id());
    });

    cyRef.current = cy;
    return () => { cy.destroy(); cyRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evidence]);

  // Update edges when slider changes without rebuilding
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !evidence) return;
    cy.remove("edge");
    const newEdges = evidence.edges
      .filter((e) => visibleEdgeIds.has(e.id))
      .map((e) => ({
        data: {
          id: e.id,
          source: e.src,
          target: e.dst,
          label: `$${(e.amount / 1000).toFixed(1)}k`,
          weight: Math.max(1, Math.min(8, e.amount / 15000)),
        },
      }));
    cy.add(newEdges);
    cy.style().update();
  }, [visibleEdgeIds, evidence]);

  // Highlight node when narrative link is clicked
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !highlightedNodeId) return;
    cy.nodes().unselect();
    const node = cy.getElementById(highlightedNodeId);
    if (node.length) {
      node.select();
      cy.animate({ center: { eles: node }, zoom: 1.4 }, { duration: 400 });
    }
  }, [highlightedNodeId]);

  const zoom = (dir: "in" | "out") => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.zoom({ level: cy.zoom() * (dir === "in" ? 1.2 : 0.83), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
  };

  const fit = () => cyRef.current?.fit(undefined, 24);

  return (
    <div className="card flex flex-col h-full overflow-hidden">
      <div className="card-header justify-between">
        <div className="flex items-center gap-2">
          <GitBranch size={16} className="text-violet-400" />
          <span className="text-sm font-semibold text-slate-200">Evidence Graph</span>
        </div>
        <div className="flex items-center gap-1">
          <button className="btn-ghost px-2 py-1.5 text-xs" onClick={() => zoom("in")}><ZoomIn size={14} /></button>
          <button className="btn-ghost px-2 py-1.5 text-xs" onClick={() => zoom("out")}><ZoomOut size={14} /></button>
          <button className="btn-ghost px-2 py-1.5 text-xs" onClick={fit}><Maximize2 size={14} /></button>
        </div>
      </div>

      {/* Role legend */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-slate-800 flex-wrap">
        {Object.entries(ROLE_COLORS).map(([role, color]) => (
          <div key={role} className="flex items-center gap-1.5 text-xs text-slate-400">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: color }} />
            <span className="capitalize">{role}</span>
          </div>
        ))}
      </div>

      {/* Cytoscape container */}
      <div className="relative flex-1 min-h-0">
        {!evidence ? (
          <div className="absolute inset-0 flex items-center justify-center text-slate-600 text-sm">
            Select an alert to view its evidence graph
          </div>
        ) : (
          <div ref={containerRef} className="absolute inset-0" />
        )}
      </div>

      {/* Timeline scrubber */}
      {evidence && sortedEdges.length > 1 && (
        <div className="px-4 py-3 border-t border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-500 mb-1.5">
            <span>Timeline scrubber</span>
            <span className="font-mono text-slate-400">
              {visibleCount}/{sortedEdges.length} tx
              {sliderValue < 100 && sortedEdges[visibleCount - 1] &&
                ` · ${new Date(sortedEdges[visibleCount - 1].timestamp).toLocaleTimeString()}`}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={sliderValue}
            onChange={(e) => setSliderValue(Number(e.target.value))}
            className="w-full accent-violet-500 cursor-pointer"
          />
        </div>
      )}
    </div>
  );
}
