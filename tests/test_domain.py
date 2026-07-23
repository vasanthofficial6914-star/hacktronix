"""
Unit Tests for Module 1: Core Domain Models & Value Objects.
"""

import pytest
from hacktronix.domain.value_objects import (
    Confidence,
    BoundingBox,
    StateAttribute,
    EntityCategory,
    RelationType,
)
from hacktronix.domain.entities import (
    Entity,
    Relationship,
    Observation,
    WorldSlice,
)


def test_confidence_clamping_and_decay():
    conf_over = Confidence(1.5)
    assert conf_over.value == 1.0

    conf_under = Confidence(-0.5)
    assert conf_under.value == 0.0

    conf = Confidence(1.0)
    decayed = conf.decay(elapsed_seconds=10.0, decay_rate=0.1)
    assert decayed.value < 1.0
    assert decayed.value > 0.0

    stale_conf = Confidence(0.1)
    assert stale_conf.is_stale() is True


def test_confidence_bayes_merge():
    c1 = Confidence(0.8)
    c2 = Confidence(0.8)
    merged = c1.merge_bayes(c2)
    assert merged.value > 0.8  # Two agreeing observations increase confidence


def test_bounding_box_properties():
    bbox = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=90, label="person")
    assert bbox.area == 3200.0
    assert bbox.center == (30.0, 50.0)
    bbox_dict = bbox.to_dict()
    assert bbox_dict["label"] == "person"


def test_entity_creation_and_serialization():
    ent = Entity(
        id="room_dungeon",
        name="Dungeon Cell",
        category=EntityCategory.ROOM,
    )
    ent.set_state("door_locked", "true", confidence_score=0.9)
    assert ent.get_state("door_locked") == "true"

    data = ent.to_dict()
    assert data["id"] == "room_dungeon"
    assert data["category"] == "room"
    assert data["states"]["door_locked"]["value"] == "true"

    reconstructed = Entity.from_dict(data)
    assert reconstructed.id == ent.id
    assert reconstructed.category == EntityCategory.ROOM
    assert reconstructed.get_state("door_locked") == "true"


def test_world_slice_text_formatting():
    room = Entity(id="room_lab", name="Research Lab", category=EntityCategory.ROOM)
    obj = Entity(id="obj_key", name="Brass Key", category=EntityCategory.OBJECT)
    rel = Relationship(id="r1", source_id="obj_key", relation_type=RelationType.LOCATED_IN, target_id="room_lab")

    slice_obj = WorldSlice(
        objective="Find key",
        current_room=room,
        visible_entities=[obj],
        relationships=[rel],
        inventory=[],
    )

    formatted = slice_obj.format_as_text_slice()
    assert "CURRENT LOCATION: Research Lab" in formatted
    assert "Brass Key" in formatted
    assert "obj_key --(located_in)--> room_lab" in formatted
