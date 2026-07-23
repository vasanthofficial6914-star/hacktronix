"""
Unit tests for Query Layer.
"""

import pytest
from hacktronix.domain.entities import Entity, Relationship
from hacktronix.domain.value_objects import EntityCategory, RelationType, Confidence
from hacktronix.infrastructure.db.database import DatabaseManager
from hacktronix.infrastructure.db.repositories import SQLiteWorldRepository
from hacktronix.infrastructure.graph_store import NetworkXGraphStore
from hacktronix.infrastructure.vector_store import FAISSVectorStore
from hacktronix.application.query_layer import QueryLayer


@pytest.fixture
def query_stack(tmp_path):
    db_mgr = DatabaseManager(str(tmp_path / "ql_test.db"))
    repo = SQLiteWorldRepository(db_mgr)
    gs = NetworkXGraphStore()
    vs = FAISSVectorStore()
    ql = QueryLayer(repo, gs, vs)
    return repo, gs, vs, ql


def test_empty_world_returns_empty_slice(query_stack):
    repo, gs, vs, ql = query_stack
    ws = ql.retrieve_slice("Find the key")
    assert ws.current_room is None
    assert ws.visible_entities == []


def test_slice_contains_relevant_entity(query_stack):
    repo, gs, vs, ql = query_stack

    room = Entity(id="hall", name="Main Hall", category=EntityCategory.ROOM)
    key = Entity(id="brass_key", name="Brass Key", category=EntityCategory.OBJECT, room_id="hall")
    repo.save_entity(room)
    repo.save_entity(key)
    vs.index_entity(room)
    vs.index_entity(key)

    rel = Relationship(id="r1", source_id="brass_key", relation_type=RelationType.LOCATED_IN, target_id="hall")
    repo.save_relationship(rel)
    gs.update_graph([room, key], [rel])

    ws = ql.retrieve_slice("Find the Brass Key", current_room_id="hall")
    entity_ids = [e.id for e in ws.visible_entities] + ([ws.current_room.id] if ws.current_room else [])
    assert "brass_key" in entity_ids


def test_slice_formatted_text_is_not_empty(query_stack):
    repo, gs, vs, ql = query_stack
    room = Entity(id="r1", name="Library", category=EntityCategory.ROOM)
    repo.save_entity(room)
    vs.index_entity(room)
    gs.update_graph([room], [])

    ws = ql.retrieve_slice("Explore", current_room_id="r1")
    text = ws.format_as_text_slice()
    assert "Library" in text
