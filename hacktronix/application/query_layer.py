"""
Query Layer & Context Slicer.

Retrieves only the most relevant subset of the World Model for the AI Agent.
Combines semantic FAISS search + NetworkX graph neighborhood traversal.
"""

from typing import List, Optional

from hacktronix.domain.entities import Entity, Relationship, WorldSlice
from hacktronix.domain.interfaces import IWorldRepository, IGraphStore, IVectorStore
from hacktronix.domain.value_objects import EntityCategory


class QueryLayer:
    """
    Provides the Retrieve-and-Slice operation for agent context.

    Strategy:
    1. Semantic FAISS top-K entity retrieval by objective.
    2. Graph 1-hop neighborhood expansion around retrieved entity IDs.
    3. Assemble WorldSlice from result set.
    4. Deduplicate and cap total entities to avoid context bloat.
    """

    MAX_ENTITIES_IN_SLICE: int = 12

    def __init__(
        self,
        repository: IWorldRepository,
        graph_store: IGraphStore,
        vector_store: IVectorStore,
    ) -> None:
        self.repository = repository
        self.graph_store = graph_store
        self.vector_store = vector_store

    def retrieve_slice(
        self,
        objective: str,
        current_room_id: Optional[str] = None,
        top_k: int = 5,
    ) -> WorldSlice:
        """
        Retrieve a compact, relevant WorldSlice for the given objective.

        Args:
            objective: Natural-language description of agent goal.
            current_room_id: Current room entity ID (ensures room always included).
            top_k: Maximum number of entities to retrieve from vector search.

        Returns:
            WorldSlice ready for agent consumption.
        """
        # Step 1: Semantic search
        semantic_ids = self.vector_store.search_relevant_entities(objective, top_k=top_k)

        # Step 2: Graph neighborhood expansion
        graph_ids = set(semantic_ids)
        for entity_id in list(semantic_ids):
            neighbors = self.graph_store.get_subgraph_nodes(entity_id, radius=1)
            graph_ids.update(neighbors)

        # Always include current room
        if current_room_id:
            graph_ids.add(current_room_id)
            room_neighbors = self.graph_store.get_subgraph_nodes(current_room_id, radius=1)
            graph_ids.update(room_neighbors)

        # Step 3: Fetch entities from repository
        candidate_ids = list(graph_ids)[: self.MAX_ENTITIES_IN_SLICE]
        candidate_entities: List[Entity] = []
        for eid in candidate_ids:
            ent = self.repository.get_entity(eid)
            if ent:
                candidate_entities.append(ent)

        # Step 4: Build WorldSlice
        current_room: Optional[Entity] = None
        visible_entities: List[Entity] = []
        inventory: List[Entity] = self.repository.get_inventory()

        inventory_ids = {i.id for i in inventory}

        for ent in candidate_entities:
            if ent.category == EntityCategory.ROOM and (
                not current_room_id or ent.id == current_room_id
            ):
                current_room = ent
            elif ent.id not in inventory_ids:
                visible_entities.append(ent)

        # Collect relevant relationships between slice entities
        slice_ids = {e.id for e in candidate_entities}
        all_rels = self.repository.get_all_relationships()
        relevant_rels: List[Relationship] = [
            r for r in all_rels
            if r.source_id in slice_ids and r.target_id in slice_ids
        ]

        # Generate summary
        summary_parts = []
        if current_room:
            exits = [f"{k}: {current_room.get_state(f'exit_{k}', '?')}" for k in ["north", "south", "east", "west"] if current_room.get_state(f"exit_{k}")]
            if exits:
                summary_parts.append("Exits: " + ", ".join(exits))
        summary = " | ".join(summary_parts)

        return WorldSlice(
            objective=objective,
            current_room=current_room,
            visible_entities=visible_entities,
            relationships=relevant_rels,
            inventory=inventory,
            graph_context_summary=summary,
        )

    def query_by_entity_name(self, name: str) -> Optional[Entity]:
        """Direct lookup of an entity by partial name match."""
        all_ents = self.repository.get_all_entities()
        name_lower = name.lower()
        for ent in all_ents:
            if name_lower in ent.name.lower():
                return ent
        return None
