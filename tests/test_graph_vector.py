"""
Unit Tests for Module 3: Knowledge Graph & Vector Store.
"""

import os
import pytest
from hacktronix.domain.entities import Entity, Relationship
from hacktronix.domain.value_objects import EntityCategory, RelationType
from hacktronix.infrastructure.graph_store import NetworkXGraphStore
from hacktronix.infrastructure.vector_store import FAISSVectorStore


def test_networkx_graph_store():
    store = NetworkXGraphStore()
    r1 = Entity(id="room_1", name="Armory", category=EntityCategory.ROOM)
    r2 = Entity(id="room_2", name="Courtyard", category=EntityCategory.ROOM)
    obj = Entity(id="sword", name="Golden Sword", category=EntityCategory.OBJECT, room_id="room_1")
    rel1 = Relationship(id="rel_1", source_id="room_1", relation_type=RelationType.CONNECTED_TO, target_id="room_2")
    rel2 = Relationship(id="rel_2", source_id="sword", relation_type=RelationType.LOCATED_IN, target_id="room_1")

    store.update_graph([r1, r2, obj], [rel1, rel2])

    subgraph = store.get_subgraph_nodes("room_1", radius=1)
    assert "room_1" in subgraph
    assert "room_2" in subgraph
    assert "sword" in subgraph

    html_path = store.export_pyvis_html("data/test_graph.html")
    assert os.path.exists(html_path)
    if os.path.exists("data/test_graph.html"):
        os.remove("data/test_graph.html")


def test_faiss_vector_store():
    vector_store = FAISSVectorStore()

    e1 = Entity(id="key_1", name="Golden Key", category=EntityCategory.OBJECT)
    e1.set_state("material", "gold")
    e2 = Entity(id="chest_1", name="Wooden Chest", category=EntityCategory.OBJECT)
    e2.set_state("status", "locked")

    vector_store.index_entity(e1)
    vector_store.index_entity(e2)

    results = vector_store.search_relevant_entities("Find a key made of gold", top_k=1)
    assert len(results) == 1
    assert results[0] == "key_1"
