"""In-memory knowledge graph backed by NetworkX for fast graph traversal."""

from __future__ import annotations

from typing import Any

import networkx as nx

from percos.stores.tables import CommittedFactRow, RelationRow


class KnowledgeGraph:
    """Lightweight in-memory graph for relationship traversal and reasoning.

    Nodes are entity IDs (str), edges carry relation metadata.
    This graph is rebuilt from the DB on startup and kept in sync during runtime.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    # ── Build / sync ────────────────────────────────────
    def load_from_facts_and_relations(
        self,
        facts: list[CommittedFactRow],
        relations: list[RelationRow],
    ) -> None:
        """Rebuild graph from DB rows."""
        self._graph.clear()
        for f in facts:
            self._graph.add_node(f.id, entity_type=f.entity_type, data=f.entity_data)
        for r in relations:
            self._graph.add_edge(
                r.source_id,
                r.target_id,
                relation_type=r.relation_type,
                weight=r.weight,
                id=r.id,
            )

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self._graph.add_node(node_id, **attrs)

    def add_edge(self, src: str, tgt: str, **attrs: Any) -> None:
        self._graph.add_edge(src, tgt, **attrs)

    # ── Query ───────────────────────────────────────────
    def neighbors(self, node_id: str, depth: int = 1) -> dict[str, Any]:
        """Return neighborhood subgraph up to given depth."""
        if node_id not in self._graph:
            return {"nodes": [], "edges": []}
        visited: set[str] = set()
        frontier = {node_id}
        nodes = []
        edges = []
        for _ in range(depth):
            next_frontier: set[str] = set()
            for n in frontier:
                if n in visited:
                    continue
                visited.add(n)
                node_data = dict(self._graph.nodes.get(n, {}))
                nodes.append({"id": n, **node_data})
                for _, neighbor, edge_data in self._graph.edges(n, data=True):
                    edges.append({"source": n, "target": neighbor, **edge_data})
                    next_frontier.add(neighbor)
                for predecessor in self._graph.predecessors(n):
                    edge_data = self._graph.edges[predecessor, n]
                    edges.append({"source": predecessor, "target": n, **edge_data})
                    next_frontier.add(predecessor)
            frontier = next_frontier - visited
        return {"nodes": nodes, "edges": edges}

    def find_by_type(self, entity_type: str) -> list[dict[str, Any]]:
        """Return all nodes of a given entity type."""
        return [
            {"id": n, **data}
            for n, data in self._graph.nodes(data=True)
            if data.get("entity_type") == entity_type
        ]

    def shortest_path(self, src: str, tgt: str) -> list[str]:
        try:
            return nx.shortest_path(self._graph, src, tgt)  # type: ignore[return-value]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()
