"""
Self-Correcting World Model Updater.

Merges incoming Observations into the World Model using confidence-weighted
conflict resolution, entity deduplication, and state decay.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from hacktronix.domain.entities import Entity, Observation, Relationship
from hacktronix.domain.interfaces import IWorldRepository, IGraphStore, IVectorStore
from hacktronix.domain.value_objects import Confidence, EntityCategory, RelationType


class UpdaterEngine:
    """
    Resolves conflicts, merges entities, and updates the full World Model stack
    (SQLite repository + NetworkX graph + FAISS vector index) on every observation.
    """

    SIMILARITY_DEDUPE_THRESHOLD: float = 0.85  # Name similarity to trigger dedup
    CONFLICT_REPLACE_THRESHOLD: float = 0.1    # Confidence delta to trigger override

    def __init__(
        self,
        repository: IWorldRepository,
        graph_store: IGraphStore,
        vector_store: IVectorStore,
    ) -> None:
        self.repository = repository
        self.graph_store = graph_store
        self.vector_store = vector_store

    def process_observation(self, observation: Observation) -> Dict[str, Any]:
        """
        Main entry point: merges a structured Observation into the World Model.

        Returns a summary of what was added, updated, or resolved.
        """
        added, updated, conflicts_resolved = [], [], []

        # Log to timeline
        self.repository.add_timeline_event(
            source_type=observation.source_type.value,
            raw_obs=observation.raw_text[:500],
            parsed_json=json.dumps(observation.to_dict(), default=str),
        )

        for entity in observation.entities:
            result = self._merge_entity(entity)
            if result == "added":
                added.append(entity.id)
            elif result == "updated":
                updated.append(entity.id)
            elif result == "conflict_resolved":
                conflicts_resolved.append(entity.id)

        for rel in observation.relationships:
            self._merge_relationship(rel)

        # Sync graph and vector store
        all_entities = self.repository.get_all_entities()
        all_rels = self.repository.get_all_relationships()
        self.graph_store.update_graph(all_entities, all_rels)
        for ent in observation.entities:
            self.vector_store.index_entity(ent)

        # Log version snapshot
        snapshot = {
            "observation_id": observation.id,
            "added": added,
            "updated": updated,
            "conflicts_resolved": conflicts_resolved,
            "timestamp": time.time(),
        }
        self.repository.add_state_history(
            event_type="OBSERVATION_PROCESSED",
            description=f"Processed {len(observation.entities)} entities from {observation.source_type.value}",
            snapshot_json=json.dumps(snapshot),
        )

        return snapshot

    def _merge_entity(self, new_entity: Entity) -> str:
        """
        Merge a new entity into the repository.
        Returns: 'added', 'updated', or 'conflict_resolved'
        """
        existing = self.repository.get_entity(new_entity.id)

        if existing is None:
            # New entity – check for near-duplicate by name
            dedup_id = self._find_duplicate(new_entity)
            if dedup_id and dedup_id != new_entity.id:
                # Merge into existing duplicate
                existing = self.repository.get_entity(dedup_id)
                if existing:
                    merged = self._merge_states(existing, new_entity)
                    self.repository.save_entity(merged)
                    return "updated"

            self.repository.save_entity(new_entity)
            return "added"

        # Entity exists – conflict-weighted merge
        merged, did_conflict = self._resolve_entity_conflicts(existing, new_entity)
        self.repository.save_entity(merged)
        return "conflict_resolved" if did_conflict else "updated"

    def _merge_relationship(self, rel: Relationship) -> None:
        """Upsert relationship, ensuring both endpoint entities are saved."""
        # Ensure orphan endpoints don't break FK constraints
        for endpoint_id in [rel.source_id, rel.target_id]:
            if not self.repository.get_entity(endpoint_id):
                placeholder = Entity(
                    id=endpoint_id,
                    name=endpoint_id.replace("_", " ").title(),
                    category=EntityCategory.UNKNOWN,
                    confidence=Confidence(0.5),
                )
                self.repository.save_entity(placeholder)
        self.repository.save_relationship(rel)

    def _resolve_entity_conflicts(
        self, old: Entity, new: Entity
    ) -> Tuple[Entity, bool]:
        """
        Compare existing vs new entity states and resolve conflicts by confidence.

        Returns (merged_entity, conflict_detected).
        """
        did_conflict = False

        # Update room_id if confidence allows
        if new.room_id and new.room_id != old.room_id:
            if new.confidence.value > old.confidence.value + self.CONFLICT_REPLACE_THRESHOLD:
                old.room_id = new.room_id
                did_conflict = True

        # Merge confidence (Bayesian combination)
        old.confidence = old.confidence.merge_bayes(new.confidence)
        old.updated_at = time.time()

        # Merge state attributes
        for key, new_attr in new.states.items():
            old_attr = old.states.get(key)
            if old_attr is None:
                old.states[key] = new_attr
            elif new_attr.value != old_attr.value:
                # Conflict: pick higher-confidence value
                if new_attr.confidence.value > old_attr.confidence.value + self.CONFLICT_REPLACE_THRESHOLD:
                    old.states[key] = new_attr
                    did_conflict = True
                elif abs(new_attr.confidence.value - old_attr.confidence.value) <= self.CONFLICT_REPLACE_THRESHOLD:
                    # Uncertain – keep old but log
                    did_conflict = True

        # Update bounding box if new one has higher confidence
        if new.bounding_box and (
            old.bounding_box is None
            or new.bounding_box.confidence > old.bounding_box.confidence
        ):
            old.bounding_box = new.bounding_box

        return old, did_conflict

    def _merge_states(self, target: Entity, source: Entity) -> Entity:
        """Copy source states into target entity (lower-confidence merge)."""
        for k, v in source.states.items():
            if k not in target.states:
                target.states[k] = v
        target.updated_at = time.time()
        return target

    def _find_duplicate(self, entity: Entity) -> Optional[str]:
        """
        Checks if a near-duplicate entity exists in the same room using name similarity.
        Returns the ID of the duplicate if found.
        """
        same_room_entities = []
        if entity.room_id:
            all_ents = self.repository.get_all_entities()
            same_room_entities = [
                e for e in all_ents if e.room_id == entity.room_id and e.id != entity.id
            ]

        new_name = entity.name.lower()
        for existing in same_room_entities:
            existing_name = existing.name.lower()
            sim = self._name_similarity(new_name, existing_name)
            if sim >= self.SIMILARITY_DEDUPE_THRESHOLD:
                return existing.id

        return None

    @staticmethod
    def _name_similarity(a: str, b: str) -> float:
        """Jaccard token similarity between two strings."""
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    def apply_confidence_decay(self, decay_rate: float = 0.02) -> None:
        """
        Apply temporal confidence decay to all dynamic entities in the repository.
        Static entities (ROOM category) are not decayed.
        """
        now = time.time()
        for entity in self.repository.get_all_entities():
            if entity.category == EntityCategory.ROOM:
                continue  # Rooms don't decay
            elapsed = now - entity.updated_at
            entity.confidence = entity.confidence.decay(elapsed, decay_rate)
            if entity.confidence.is_stale():
                # Mark stale in states but don't delete (preserve history)
                entity.set_state("stale", "true", confidence_score=entity.confidence.value)
            self.repository.save_entity(entity)
