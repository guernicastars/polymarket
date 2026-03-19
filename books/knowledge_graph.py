"""
Book Knowledge Graph — Math Textbook Edition

A knowledge graph that captures concepts, theorems, definitions, and their
relationships across math textbooks. Designed to be manually populated from
textbook photos.

Usage:
    from books.knowledge_graph import BookKnowledgeGraph
    kg = BookKnowledgeGraph()
    kg.load()  # loads from data/books.json
    kg.summary()
    kg.prereqs_for("eigenvalues")
    kg.concepts_in_book("Linear Algebra Done Right")
    kg.shortest_path("derivatives", "stokes_theorem")
    kg.export_viz()  # generates interactive HTML
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
GRAPH_FILE = DATA_DIR / "books.json"
VIZ_FILE = Path(__file__).parent / "viz" / "knowledge-graph.html"


@dataclass
class Book:
    id: str
    title: str
    authors: list[str]
    subject: str
    edition: str = ""
    year: int = 0
    chapters: list[str] = field(default_factory=list)


@dataclass
class Concept:
    id: str
    name: str
    subject: str
    book_id: str
    chapter: str = ""
    kind: str = "concept"  # concept | theorem | definition | formula | axiom | lemma | corollary
    statement: str = ""
    formula: str = ""
    page: int = 0
    importance: int = 1  # 1-5, how fundamental


@dataclass
class Relation:
    source: str  # concept id
    target: str  # concept id
    kind: str  # prerequisite | generalizes | applies | equivalent | example_of | proves | related
    note: str = ""


class BookKnowledgeGraph:
    def __init__(self):
        self.books: dict[str, Book] = {}
        self.concepts: dict[str, Concept] = {}
        self.relations: list[Relation] = []

    # ── Persistence ──────────────────────────────────────────────

    def save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "books": {k: asdict(v) for k, v in self.books.items()},
            "concepts": {k: asdict(v) for k, v in self.concepts.items()},
            "relations": [asdict(r) for r in self.relations],
        }
        GRAPH_FILE.write_text(json.dumps(data, indent=2))

    def load(self):
        if not GRAPH_FILE.exists():
            return
        data = json.loads(GRAPH_FILE.read_text())
        self.books = {k: Book(**v) for k, v in data.get("books", {}).items()}
        self.concepts = {k: Concept(**v) for k, v in data.get("concepts", {}).items()}
        self.relations = [Relation(**r) for r in data.get("relations", [])]

    # ── Add entities ─────────────────────────────────────────────

    def add_book(self, id: str, title: str, authors: list[str], subject: str, **kw) -> Book:
        b = Book(id=id, title=title, authors=authors, subject=subject, **kw)
        self.books[id] = b
        return b

    def add_concept(self, id: str, name: str, subject: str, book_id: str, **kw) -> Concept:
        c = Concept(id=id, name=name, subject=subject, book_id=book_id, **kw)
        self.concepts[id] = c
        return c

    def add_relation(self, source: str, target: str, kind: str, note: str = "") -> Relation:
        r = Relation(source=source, target=target, kind=kind, note=note)
        self.relations.append(r)
        return r

    # ── Queries ──────────────────────────────────────────────────

    def concepts_in_book(self, book_id: str) -> list[Concept]:
        return [c for c in self.concepts.values() if c.book_id == book_id]

    def concepts_by_subject(self, subject: str) -> list[Concept]:
        return [c for c in self.concepts.values() if c.subject.lower() == subject.lower()]

    def prereqs_for(self, concept_id: str) -> list[Concept]:
        """Direct prerequisites for a concept."""
        ids = [r.source for r in self.relations if r.target == concept_id and r.kind == "prerequisite"]
        return [self.concepts[i] for i in ids if i in self.concepts]

    def depends_on(self, concept_id: str) -> list[Concept]:
        """Concepts that depend on this one."""
        ids = [r.target for r in self.relations if r.source == concept_id and r.kind == "prerequisite"]
        return [self.concepts[i] for i in ids if i in self.concepts]

    def all_prereqs(self, concept_id: str) -> list[Concept]:
        """Transitive closure of prerequisites (BFS)."""
        visited = set()
        queue = [concept_id]
        while queue:
            cid = queue.pop(0)
            for r in self.relations:
                if r.target == cid and r.kind == "prerequisite" and r.source not in visited:
                    visited.add(r.source)
                    queue.append(r.source)
        return [self.concepts[i] for i in visited if i in self.concepts]

    def shortest_path(self, source: str, target: str) -> Optional[list[str]]:
        """BFS shortest path between two concepts (ignoring direction)."""
        adj: dict[str, list[str]] = {}
        for r in self.relations:
            adj.setdefault(r.source, []).append(r.target)
            adj.setdefault(r.target, []).append(r.source)
        visited = {source}
        queue = [(source, [source])]
        while queue:
            node, path = queue.pop(0)
            if node == target:
                return path
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None

    def cross_book_links(self) -> list[Relation]:
        """Relations connecting concepts from different books."""
        result = []
        for r in self.relations:
            s, t = self.concepts.get(r.source), self.concepts.get(r.target)
            if s and t and s.book_id != t.book_id:
                result.append(r)
        return result

    # ── Stats ────────────────────────────────────────────────────

    def summary(self) -> dict:
        subjects = set(c.subject for c in self.concepts.values())
        kinds = {}
        for c in self.concepts.values():
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        rel_kinds = {}
        for r in self.relations:
            rel_kinds[r.kind] = rel_kinds.get(r.kind, 0) + 1
        return {
            "books": len(self.books),
            "concepts": len(self.concepts),
            "relations": len(self.relations),
            "subjects": sorted(subjects),
            "concept_kinds": kinds,
            "relation_kinds": rel_kinds,
            "cross_book_links": len(self.cross_book_links()),
        }

    # ── Visualization ────────────────────────────────────────────

    def export_viz(self, output: Optional[str] = None) -> str:
        """Generate interactive D3.js force-directed knowledge graph."""
        output = output or str(VIZ_FILE)
        os.makedirs(os.path.dirname(output), exist_ok=True)

        nodes = []
        for c in self.concepts.values():
            book = self.books.get(c.book_id)
            nodes.append({
                "id": c.id,
                "name": c.name,
                "subject": c.subject,
                "kind": c.kind,
                "book": book.title if book else c.book_id,
                "importance": c.importance,
                "statement": c.statement,
                "formula": c.formula,
            })

        links = []
        for r in self.relations:
            if r.source in self.concepts and r.target in self.concepts:
                links.append({
                    "source": r.source,
                    "target": r.target,
                    "kind": r.kind,
                    "note": r.note,
                })

        subjects = sorted(set(c.subject for c in self.concepts.values()))
        books_list = [{"id": b.id, "title": b.title, "subject": b.subject} for b in self.books.values()]

        html = _VIZ_TEMPLATE.replace("__NODES__", json.dumps(nodes))
        html = html.replace("__LINKS__", json.dumps(links))
        html = html.replace("__SUBJECTS__", json.dumps(subjects))
        html = html.replace("__BOOKS__", json.dumps(books_list))

        Path(output).write_text(html)
        return output

    def print_tree(self, concept_id: str, depth: int = 3, _indent: int = 0):
        """Print prerequisite tree."""
        c = self.concepts.get(concept_id)
        if not c:
            return
        prefix = "  " * _indent + ("└─ " if _indent > 0 else "")
        print(f"{prefix}{c.name} [{c.kind}] ({c.subject})")
        if _indent < depth:
            for p in self.prereqs_for(concept_id):
                self.print_tree(p.id, depth, _indent + 1)


_VIZ_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Book Knowledge Graph</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, sans-serif; background: #0a0a0f; color: #e0e0e0; overflow: hidden; }
#graph { width: 100vw; height: 100vh; }
.controls { position: fixed; top: 16px; left: 16px; z-index: 10; display: flex; flex-direction: column; gap: 8px; }
.controls select, .controls input { background: #1a1a2e; color: #e0e0e0; border: 1px solid #333; padding: 8px 12px; border-radius: 6px; font-size: 13px; }
.tooltip { position: fixed; background: #1a1a2e; border: 1px solid #444; padding: 12px 16px; border-radius: 8px; font-size: 13px; pointer-events: none; max-width: 350px; z-index: 100; display: none; }
.tooltip h3 { color: #7c8cf8; margin-bottom: 6px; font-size: 14px; }
.tooltip .kind { color: #888; font-size: 11px; text-transform: uppercase; margin-bottom: 4px; }
.tooltip .stmt { color: #bbb; font-style: italic; margin-top: 4px; }
.tooltip .formula { color: #f0c040; font-family: monospace; margin-top: 4px; }
.legend { position: fixed; bottom: 16px; left: 16px; background: #1a1a2e; border: 1px solid #333; padding: 12px; border-radius: 8px; font-size: 12px; z-index: 10; }
.legend div { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
.legend span.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.stats { position: fixed; top: 16px; right: 16px; background: #1a1a2e; border: 1px solid #333; padding: 12px 16px; border-radius: 8px; font-size: 12px; z-index: 10; }
.stats div { margin: 2px 0; }
.stats .num { color: #7c8cf8; font-weight: bold; }
</style>
</head>
<body>
<div class="controls">
  <select id="subjectFilter"><option value="all">All Subjects</option></select>
  <select id="bookFilter"><option value="all">All Books</option></select>
  <select id="kindFilter">
    <option value="all">All Types</option>
    <option value="concept">Concepts</option>
    <option value="theorem">Theorems</option>
    <option value="definition">Definitions</option>
    <option value="formula">Formulas</option>
    <option value="axiom">Axioms</option>
    <option value="lemma">Lemmas</option>
  </select>
  <input id="search" type="text" placeholder="Search concepts...">
</div>
<div class="stats" id="stats"></div>
<div class="tooltip" id="tooltip"></div>
<div class="legend" id="legend"></div>
<svg id="graph"></svg>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const allNodes = __NODES__;
const allLinks = __LINKS__;
const subjects = __SUBJECTS__;
const books = __BOOKS__;

const subjectColors = {};
const palette = ["#7c8cf8","#f07878","#50c878","#f0c040","#c084fc","#ff8c42","#42d4f4","#ff6eb4","#a0e060","#ff5050"];
subjects.forEach((s, i) => subjectColors[s] = palette[i % palette.length]);

const kindShapes = { concept: "circle", theorem: "diamond", definition: "square", formula: "triangle", axiom: "star", lemma: "circle", corollary: "circle" };
const relColors = { prerequisite: "#666", generalizes: "#7c8cf8", applies: "#50c878", equivalent: "#f0c040", example_of: "#c084fc", proves: "#f07878", related: "#555" };

// Populate filters
const sf = d3.select("#subjectFilter");
subjects.forEach(s => sf.append("option").attr("value", s).text(s));
const bf = d3.select("#bookFilter");
books.forEach(b => bf.append("option").attr("value", b.id).text(b.title));

// Legend
const leg = d3.select("#legend");
subjects.forEach(s => {
  leg.append("div").html(`<span class="dot" style="background:${subjectColors[s]}"></span>${s}`);
});

const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select("#graph").attr("width", width).attr("height", height);
const g = svg.append("g");

svg.call(d3.zoom().scaleExtent([0.1, 8]).on("zoom", e => g.attr("transform", e.transform)));

let simulation, linkSel, nodeSel;

function render(nodes, links) {
  g.selectAll("*").remove();

  const nodeMap = new Set(nodes.map(n => n.id));
  const filteredLinks = links.filter(l => nodeMap.has(l.source.id || l.source) && nodeMap.has(l.target.id || l.target));

  simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(filteredLinks).id(d => d.id).distance(80))
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(d => 8 + d.importance * 3));

  linkSel = g.append("g").selectAll("line").data(filteredLinks).join("line")
    .attr("stroke", d => relColors[d.kind] || "#444")
    .attr("stroke-width", 1.2)
    .attr("stroke-opacity", 0.5)
    .attr("stroke-dasharray", d => d.kind === "related" ? "4,3" : null);

  nodeSel = g.append("g").selectAll("g").data(nodes).join("g")
    .call(d3.drag().on("start", dragStart).on("drag", dragging).on("end", dragEnd));

  nodeSel.append("circle")
    .attr("r", d => 5 + d.importance * 2.5)
    .attr("fill", d => subjectColors[d.subject] || "#888")
    .attr("stroke", "#222")
    .attr("stroke-width", 1.5);

  nodeSel.append("text")
    .text(d => d.name)
    .attr("dx", d => 8 + d.importance * 2)
    .attr("dy", 4)
    .attr("font-size", d => 10 + d.importance)
    .attr("fill", "#ccc");

  nodeSel.on("mouseover", (e, d) => {
    const tip = d3.select("#tooltip");
    let html = `<div class="kind">${d.kind} · ${d.subject}</div><h3>${d.name}</h3><div>Book: ${d.book}</div>`;
    if (d.statement) html += `<div class="stmt">${d.statement}</div>`;
    if (d.formula) html += `<div class="formula">${d.formula}</div>`;
    tip.html(html).style("display", "block").style("left", (e.clientX + 16) + "px").style("top", (e.clientY - 10) + "px");
  }).on("mouseout", () => d3.select("#tooltip").style("display", "none"));

  simulation.on("tick", () => {
    linkSel.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    nodeSel.attr("transform", d => `translate(${d.x},${d.y})`);
  });

  d3.select("#stats").html(
    `<div><span class="num">${nodes.length}</span> concepts</div>` +
    `<div><span class="num">${filteredLinks.length}</span> relations</div>` +
    `<div><span class="num">${new Set(nodes.map(n => n.book)).size}</span> books</div>`
  );
}

function filter() {
  const sub = d3.select("#subjectFilter").node().value;
  const book = d3.select("#bookFilter").node().value;
  const kind = d3.select("#kindFilter").node().value;
  const q = d3.select("#search").node().value.toLowerCase();

  let nodes = allNodes.filter(n => {
    if (sub !== "all" && n.subject !== sub) return false;
    if (book !== "all" && n.book_id !== book) return false;
    if (kind !== "all" && n.kind !== kind) return false;
    if (q && !n.name.toLowerCase().includes(q) && !n.id.toLowerCase().includes(q)) return false;
    return true;
  });
  render(nodes, allLinks);
}

d3.selectAll("#subjectFilter, #bookFilter, #kindFilter").on("change", filter);
d3.select("#search").on("input", filter);

function dragStart(e, d) { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
function dragging(e, d) { d.fx = e.x; d.fy = e.y; }
function dragEnd(e, d) { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }

render(allNodes, allLinks);
</script>
</body>
</html>"""
