"""
Knowledge Graph Engine using NetworkX and PyVis.

Implements `IGraphStore` interface for maintaining spatial/semantic entity graphs,
performing topological subgraph slicing, and exporting interactive HTML visualizations.
"""

import os
from typing import List, Dict, Any, Optional
import networkx as nx

from hacktronix.domain.interfaces import IGraphStore
from hacktronix.domain.entities import Entity, Relationship
from hacktronix.domain.value_objects import EntityCategory, RelationType


class NetworkXGraphStore(IGraphStore):
    """
    Graph store implementation powered by NetworkX and PyVis.
    """

    # Category color mapping for PyVis rendering
    CATEGORY_COLORS = {
        EntityCategory.ROOM: "#1E88E5",      # Blue
        EntityCategory.OBJECT: "#4CAF50",    # Green
        EntityCategory.PERSON: "#E53935",    # Red
        EntityCategory.FACE: "#FB8C00",      # Orange
        EntityCategory.DOOR: "#FDD835",      # Yellow
        EntityCategory.INVENTORY: "#8E24AA", # Purple
        EntityCategory.UNKNOWN: "#757575",   # Grey
    }

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def update_graph(self, entities: List[Entity], relationships: List[Relationship]) -> None:
        """Update graph nodes and edges with entities and relationships."""
        for ent in entities:
            cat = ent.category if isinstance(ent.category, EntityCategory) else EntityCategory(str(ent.category))
            color = self.CATEGORY_COLORS.get(cat, "#757575")
            label = f"{ent.name}\n({cat.value})"
            
            # Format states preview
            states_str = ", ".join([f"{k}:{v.value}" for k, v in ent.states.items()])
            
            self.graph.add_node(
                ent.id,
                label=label,
                name=ent.name,
                category=cat.value,
                room_id=ent.room_id,
                confidence=ent.confidence.value,
                states=states_str,
                color=color,
                title=f"ID: {ent.id}<br>Category: {cat.value}<br>States: {states_str}"
            )

        for rel in relationships:
            rel_type = rel.relation_type.value if isinstance(rel.relation_type, RelationType) else str(rel.relation_type)
            self.graph.add_edge(
                rel.source_id,
                rel.target_id,
                id=rel.id,
                relation_type=rel_type,
                label=rel_type,
                confidence=rel.confidence.value,
            )

    def get_subgraph_nodes(self, center_node_id: str, radius: int = 1) -> List[str]:
        """
        Retrieves node IDs within N-hop distance from the center node ID.
        """
        if center_node_id not in self.graph:
            return []
        
        # Undirected traversal to include incoming and outgoing edges
        undirected = self.graph.to_undirected()
        try:
            lengths = nx.single_source_shortest_path_length(undirected, center_node_id, cutoff=radius)
            return list(lengths.keys())
        except Exception:
            return [center_node_id]

    def export_pyvis_html(self, output_path: str = "data/knowledge_graph.html") -> str:
        """
        Exports the NetworkX graph to an interactive HTML visualizer using PyVis.
        Returns the output file path.
        """
        try:
            from pyvis.network import Network
            net = Network(height="600px", width="100%", directed=True, notebook=False, bgcolor="#111827", font_color="#FFFFFF")
            net.from_nx(self.graph)
            net.toggle_physics(True)
            
            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                
            net.write_html(output_path)
            return output_path
        except Exception as e:
            # Fallback plain HTML if PyVis has issue
            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            html_content = f"<html><body style='background:#111827;color:#fff;'><h3>Knowledge Graph ({len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges)</h3></body></html>"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return output_path

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self.graph.clear()
