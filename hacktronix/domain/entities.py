"""
Domain Entities for HackModel-AI.

Contains definitions for Entity, Relationship, Observation, WorldSlice,
StateHistoryEntry, and TimelineEntry.
Pure domain models with rich attributes and conversion utilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import uuid

from hacktronix.domain.value_objects import (
    EntityCategory,
    RelationType,
    ObservationType,
    Confidence,
    BoundingBox,
    StateAttribute,
)


@dataclass
class Entity:
    """
    Primary entity in the World Model (Room, Door, Object, Person, Face, Item).
    """
    id: str
    name: str
    category: EntityCategory
    room_id: Optional[str] = None
    confidence: Confidence = field(default_factory=Confidence)
    states: Dict[str, StateAttribute] = field(default_factory=dict)
    bounding_box: Optional[BoundingBox] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def set_state(self, key: str, value: str, confidence_score: float = 1.0) -> None:
        """Add or update a state attribute."""
        self.states[key] = StateAttribute(
            key=key,
            value=value,
            confidence=Confidence(confidence_score),
            updated_at=time.time(),
        )
        self.updated_at = time.time()

    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve state attribute value by key."""
        attr = self.states.get(key)
        return attr.value if attr else default

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity into structured dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value if isinstance(self.category, EntityCategory) else str(self.category),
            "room_id": self.room_id,
            "confidence": self.confidence.value,
            "states": {k: v.to_dict() for k, v in self.states.items()},
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Deserialize entity from dictionary representation."""
        bbox_data = data.get("bounding_box")
        bbox = BoundingBox(**bbox_data) if bbox_data else None

        states_dict = {}
        for k, v in data.get("states", {}).items():
            conf_val = v.get("confidence", 1.0) if isinstance(v, dict) else 1.0
            val = v.get("value", str(v)) if isinstance(v, dict) else str(v)
            states_dict[k] = StateAttribute(
                key=k,
                value=val,
                confidence=Confidence(conf_val),
                updated_at=v.get("updated_at", time.time()) if isinstance(v, dict) else time.time(),
            )

        return cls(
            id=data["id"],
            name=data["name"],
            category=EntityCategory(data["category"]),
            room_id=data.get("room_id"),
            confidence=Confidence(data.get("confidence", 1.0)),
            states=states_dict,
            bounding_box=bbox,
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


@dataclass
class Relationship:
    """
    Edge connecting two entities in the World Model Graph.
    """
    id: str
    source_id: str
    relation_type: RelationType
    target_id: str
    confidence: Confidence = field(default_factory=Confidence)
    last_observed: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "relation_type": self.relation_type.value if isinstance(self.relation_type, RelationType) else str(self.relation_type),
            "target_id": self.target_id,
            "confidence": self.confidence.value,
            "last_observed": self.last_observed,
        }


@dataclass
class Observation:
    """
    Raw or semi-structured observation incoming from text environment or vision detector.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_type: ObservationType = ObservationType.TEXT
    raw_text: str = ""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    current_room_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type.value,
            "raw_text": self.raw_text,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "current_room_id": self.current_room_id,
            "timestamp": self.timestamp,
        }


@dataclass
class WorldSlice:
    """
    Filtered, relevant slice of the World Model provided to the AI Agent.
    Contains ONLY what the agent needs for its current objective.
    """
    objective: str
    current_room: Optional[Entity] = None
    visible_entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    inventory: List[Entity] = field(default_factory=list)
    graph_context_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "current_room": self.current_room.to_dict() if self.current_room else None,
            "visible_entities": [e.to_dict() for e in self.visible_entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "inventory": [i.to_dict() for i in self.inventory],
            "graph_context_summary": self.graph_context_summary,
        }

    def format_as_text_slice(self) -> str:
        """Formats slice into clean markdown text for zero-history LLM prompt."""
        lines = []
        if self.current_room:
            lines.append(f"CURRENT LOCATION: {self.current_room.name} (ID: {self.current_room.id})")
            for k, v in self.current_room.states.items():
                lines.append(f"  - Room State [{k}]: {v.value}")
        else:
            lines.append("CURRENT LOCATION: Unknown")

        lines.append("\nVISIBLE ENTITIES:")
        if not self.visible_entities:
            lines.append("  - None")
        else:
            for ent in self.visible_entities:
                state_str = ", ".join([f"{k}={v.value}" for k, v in ent.states.items()])
                state_info = f" ({state_str})" if state_str else ""
                lines.append(f"  - [{ent.category.value.upper()}] {ent.name} (ID: {ent.id}){state_info}")

        lines.append("\nRELATIONSHIPS:")
        if not self.relationships:
            lines.append("  - None")
        else:
            for rel in self.relationships:
                lines.append(f"  - {rel.source_id} --({rel.relation_type.value})--> {rel.target_id}")

        lines.append("\nAGENT INVENTORY:")
        if not self.inventory:
            lines.append("  - Empty")
        else:
            for item in self.inventory:
                lines.append(f"  - {item.name} (ID: {item.id})")

        if self.graph_context_summary:
            lines.append(f"\nCONTEXT SUMMARY: {self.graph_context_summary}")

        return "\n".join(lines)


@dataclass
class StateHistoryEntry:
    """Immutable state snapshot for version history and audit log."""
    version_id: int
    event_type: str
    entity_id: Optional[str]
    description: str
    snapshot_json: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TimelineEntry:
    """Timeline entry logging observation events."""
    id: str
    source_type: str
    raw_observation: str
    parsed_json: str
    timestamp: float = field(default_factory=time.time)
