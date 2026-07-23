"""
Unit Tests for Module 2: Database & Persistence Layer.
"""

import os
import pytest
from hacktronix.domain.value_objects import EntityCategory, RelationType, Confidence
from hacktronix.domain.entities import Entity, Relationship
from hacktronix.infrastructure.db.database import DatabaseManager
from hacktronix.infrastructure.db.repositories import SQLiteWorldRepository


@pytest.fixture
def temp_repo(tmp_path):
    db_file = str(tmp_path / "test_world.db")
    db_mgr = DatabaseManager(db_file)
    return SQLiteWorldRepository(db_mgr)


def test_entity_persistence(temp_repo):
    room = Entity(id="room_kitchen", name="Kitchen", category=EntityCategory.ROOM)
    room.set_state("clean", "true")
    temp_repo.save_entity(room)

    fetched = temp_repo.get_entity("room_kitchen")
    assert fetched is not None
    assert fetched.name == "Kitchen"
    assert fetched.category == EntityCategory.ROOM
    assert fetched.get_state("clean") == "true"


def test_relationship_persistence(temp_repo):
    r1 = Entity(id="room_1", name="Room 1", category=EntityCategory.ROOM)
    r2 = Entity(id="room_2", name="Room 2", category=EntityCategory.ROOM)
    temp_repo.save_entity(r1)
    temp_repo.save_entity(r2)

    rel = Relationship(id="rel_1", source_id="room_1", relation_type=RelationType.CONNECTED_TO, target_id="room_2")
    temp_repo.save_relationship(rel)

    rels = temp_repo.get_relationships_for_entity("room_1")
    assert len(rels) == 1
    assert rels[0].relation_type == RelationType.CONNECTED_TO


def test_inventory_management(temp_repo):
    item = Entity(id="item_sword", name="Iron Sword", category=EntityCategory.OBJECT)
    temp_repo.save_entity(item)
    temp_repo.add_to_inventory("item_sword")

    inv = temp_repo.get_inventory()
    assert len(inv) == 1
    assert inv[0].id == "item_sword"

    temp_repo.remove_from_inventory("item_sword")
    assert len(temp_repo.get_inventory()) == 0


def test_version_history_and_timeline(temp_repo):
    version_id = temp_repo.add_state_history("ADD_ENTITY", "Added sword", '{"id": "item_sword"}', "item_sword")
    assert version_id > 0

    history = temp_repo.get_state_history(limit=5)
    assert len(history) == 1
    assert history[0].description == "Added sword"

    t_entry = temp_repo.add_timeline_event("TEXT", "You enter the kitchen", '{"room": "kitchen"}')
    assert t_entry.id is not None
    timeline = temp_repo.get_timeline(limit=5)
    assert len(timeline) == 1
    assert timeline[0].raw_observation == "You enter the kitchen"
