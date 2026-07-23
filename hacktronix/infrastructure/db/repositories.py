"""
SQLite Repository Implementation for World Model.

Implements `IWorldRepository` interface for storing entities, relationships, inventory,
version snapshots, and observation timelines using SQLite.
"""

import json
import time
from typing import List, Optional, Dict, Any

from hacktronix.domain.interfaces import IWorldRepository
from hacktronix.domain.entities import (
    Entity,
    Relationship,
    StateHistoryEntry,
    TimelineEntry,
)
from hacktronix.domain.value_objects import (
    EntityCategory,
    RelationType,
    Confidence,
    BoundingBox,
    StateAttribute,
)
from hacktronix.infrastructure.db.database import DatabaseManager


class SQLiteWorldRepository(IWorldRepository):
    """
    Concrete SQLite implementation of IWorldRepository.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def save_entity(self, entity: Entity) -> None:
        """Upsert entity record in database."""
        states_data = {k: v.to_dict() for k, v in entity.states.items()}
        states_json = json.dumps(states_data)
        bbox_json = json.dumps(entity.bounding_box.to_dict()) if entity.bounding_box else None

        sql = """
        INSERT INTO entities (id, name, category, room_id, confidence, states_json, bounding_box_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            category = excluded.category,
            room_id = excluded.room_id,
            confidence = excluded.confidence,
            states_json = excluded.states_json,
            bounding_box_json = excluded.bounding_box_json,
            updated_at = excluded.updated_at;
        """
        cat_str = entity.category.value if isinstance(entity.category, EntityCategory) else str(entity.category)
        with self.db.get_connection() as conn:
            conn.execute(sql, (
                entity.id,
                entity.name,
                cat_str,
                entity.room_id,
                entity.confidence.value,
                states_json,
                bbox_json,
                entity.created_at,
                entity.updated_at,
            ))
            conn.commit()

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Fetch entity by primary key ID."""
        sql = "SELECT * FROM entities WHERE id = ?;"
        with self.db.get_connection() as conn:
            row = conn.execute(sql, (entity_id,)).fetchone()
            if not row:
                return None
            return self._row_to_entity(row)

    def get_entities_by_category(self, category: EntityCategory) -> List[Entity]:
        """Fetch entities matching given category enum."""
        cat_str = category.value if isinstance(category, EntityCategory) else str(category)
        sql = "SELECT * FROM entities WHERE category = ?;"
        with self.db.get_connection() as conn:
            rows = conn.execute(sql, (cat_str,)).fetchall()
            return [self._row_to_entity(r) for r in rows]

    def get_all_entities(self) -> List[Entity]:
        """Fetch all entities."""
        sql = "SELECT * FROM entities;"
        with self.db.get_connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [self._row_to_entity(r) for r in rows]

    def delete_entity(self, entity_id: str) -> None:
        """Delete entity and associated relationships/inventory."""
        sql = "DELETE FROM entities WHERE id = ?;"
        with self.db.get_connection() as conn:
            conn.execute(sql, (entity_id,))
            conn.commit()

    def save_relationship(self, relationship: Relationship) -> None:
        """Upsert relationship record."""
        rel_str = relationship.relation_type.value if isinstance(relationship.relation_type, RelationType) else str(relationship.relation_type)
        sql = """
        INSERT INTO relationships (id, source_id, relation_type, target_id, confidence, last_observed)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            relation_type = excluded.relation_type,
            confidence = excluded.confidence,
            last_observed = excluded.last_observed;
        """
        with self.db.get_connection() as conn:
            conn.execute(sql, (
                relationship.id,
                relationship.source_id,
                rel_str,
                relationship.target_id,
                relationship.confidence.value,
                relationship.last_observed,
            ))
            conn.commit()

    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Fetch all relationships where entity is source or target."""
        sql = "SELECT * FROM relationships WHERE source_id = ? OR target_id = ?;"
        with self.db.get_connection() as conn:
            rows = conn.execute(sql, (entity_id, entity_id)).fetchall()
            return [self._row_to_relationship(r) for r in rows]

    def get_all_relationships(self) -> List[Relationship]:
        """Fetch all relationships."""
        sql = "SELECT * FROM relationships;"
        with self.db.get_connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [self._row_to_relationship(r) for r in rows]

    def add_to_inventory(self, entity_id: str) -> None:
        """Add entity to inventory table."""
        sql = "INSERT INTO inventory (entity_id, acquired_at) VALUES (?, ?) ON CONFLICT(entity_id) DO NOTHING;"
        with self.db.get_connection() as conn:
            conn.execute(sql, (entity_id, time.time()))
            conn.commit()

    def remove_from_inventory(self, entity_id: str) -> None:
        """Remove entity from inventory table."""
        sql = "DELETE FROM inventory WHERE entity_id = ?;"
        with self.db.get_connection() as conn:
            conn.execute(sql, (entity_id,))
            conn.commit()

    def get_inventory(self) -> List[Entity]:
        """Retrieve full entity objects in agent inventory."""
        sql = "SELECT e.* FROM entities e JOIN inventory i ON e.id = i.entity_id;"
        with self.db.get_connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [self._row_to_entity(r) for r in rows]

    def add_state_history(self, event_type: str, description: str, snapshot_json: str, entity_id: Optional[str] = None) -> int:
        """Log state change snapshot to audit history."""
        sql = """
        INSERT INTO state_history (event_type, entity_id, description, snapshot_json, timestamp)
        VALUES (?, ?, ?, ?, ?);
        """
        ts = time.time()
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (event_type, entity_id, description, snapshot_json, ts))
            conn.commit()
            return cursor.lastrowid or 0

    def get_state_history(self, limit: int = 50) -> List[StateHistoryEntry]:
        """Fetch recent history snapshots."""
        sql = "SELECT * FROM state_history ORDER BY version_id DESC LIMIT ?;"
        with self.db.get_connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
            return [
                StateHistoryEntry(
                    version_id=r["version_id"],
                    event_type=r["event_type"],
                    entity_id=r["entity_id"],
                    description=r["description"],
                    snapshot_json=r["snapshot_json"],
                    timestamp=r["timestamp"],
                )
                for r in rows
            ]

    def add_timeline_event(self, source_type: str, raw_obs: str, parsed_json: str) -> TimelineEntry:
        """Log observation into timeline."""
        import uuid
        event_id = str(uuid.uuid4())
        ts = time.time()
        sql = """
        INSERT INTO timeline (id, source_type, raw_observation, parsed_json, timestamp)
        VALUES (?, ?, ?, ?, ?);
        """
        with self.db.get_connection() as conn:
            conn.execute(sql, (event_id, source_type, raw_obs, parsed_json, ts))
            conn.commit()
        return TimelineEntry(
            id=event_id,
            source_type=source_type,
            raw_observation=raw_obs,
            parsed_json=parsed_json,
            timestamp=ts,
        )

    def get_timeline(self, limit: int = 50) -> List[TimelineEntry]:
        """Fetch timeline events."""
        sql = "SELECT * FROM timeline ORDER BY timestamp DESC LIMIT ?;"
        with self.db.get_connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
            return [
                TimelineEntry(
                    id=r["id"],
                    source_type=r["source_type"],
                    raw_observation=r["raw_observation"],
                    parsed_json=r["parsed_json"],
                    timestamp=r["timestamp"],
                )
                for r in rows
            ]

    def _row_to_entity(self, row: Any) -> Entity:
        """Map SQLite row to Entity domain object."""
        states_dict = {}
        if row["states_json"]:
            raw_states = json.loads(row["states_json"])
            for k, v in raw_states.items():
                states_dict[k] = StateAttribute(
                    key=k,
                    value=v["value"],
                    confidence=Confidence(v.get("confidence", 1.0)),
                    updated_at=v.get("updated_at", time.time()),
                )

        bbox = None
        if row["bounding_box_json"]:
            bbox_raw = json.loads(row["bounding_box_json"])
            bbox = BoundingBox(**bbox_raw)

        return Entity(
            id=row["id"],
            name=row["name"],
            category=EntityCategory(row["category"]),
            room_id=row["room_id"],
            confidence=Confidence(row["confidence"]),
            states=states_dict,
            bounding_box=bbox,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_relationship(self, row: Any) -> Relationship:
        """Map SQLite row to Relationship domain object."""
        return Relationship(
            id=row["id"],
            source_id=row["source_id"],
            relation_type=RelationType(row["relation_type"]),
            target_id=row["target_id"],
            confidence=Confidence(row["confidence"]),
            last_observed=row["last_observed"],
        )
