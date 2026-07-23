"""
Structured Observation Extractor.

Converts raw text descriptions (from text env) and raw vision detections (from vision env)
into structured Observation domain objects for the World Model Updater.
"""

import json
import uuid
import time
from typing import Any, Dict, List, Optional

from hacktronix.domain.entities import Entity, Relationship, Observation
from hacktronix.domain.value_objects import (
    EntityCategory,
    RelationType,
    ObservationType,
    Confidence,
    BoundingBox,
)


class ObservationExtractor:
    """
    Converts raw text adventure observations and vision detection results
    into structured Observation domain objects.
    """

    # ------------------------------------------------------------------ #
    # Track 1: Text Environment Extraction                                  #
    # ------------------------------------------------------------------ #

    def extract_from_text_obs(self, raw_obs: Dict[str, Any]) -> Observation:
        """
        Parses a text adventure observation dict into an Observation domain object.

        Args:
            raw_obs: dict returned by TextAdventureEnv.observe()

        Returns:
            Structured Observation with entities and relationships.
        """
        entities: List[Entity] = []
        relationships: List[Relationship] = []

        current_room_data = raw_obs.get("current_room", {})
        room_id = current_room_data.get("id", "unknown_room")
        room_name = current_room_data.get("name", "Unknown Room")
        room_desc = current_room_data.get("description", "")

        # Room entity
        room_entity = Entity(
            id=room_id,
            name=room_name,
            category=EntityCategory.ROOM,
        )
        room_entity.set_state("description", room_desc[:120])
        exits = current_room_data.get("exits", {})
        for direction, exit_room_id in exits.items():
            room_entity.set_state(f"exit_{direction}", exit_room_id)
        entities.append(room_entity)

        # Rooms connected via exits
        for direction, exit_room_id in exits.items():
            rel = Relationship(
                id=f"rel_exit_{room_id}_{exit_room_id}",
                source_id=room_id,
                relation_type=RelationType.CONNECTED_TO,
                target_id=exit_room_id,
                confidence=Confidence(1.0),
                last_observed=time.time(),
            )
            relationships.append(rel)

        # Objects in room
        for obj_data in raw_obs.get("objects_in_room", []):
            obj_id = obj_data.get("id", str(uuid.uuid4()))
            obj_entity = Entity(
                id=obj_id,
                name=obj_data.get("name", "Unknown Object"),
                category=EntityCategory.OBJECT,
                room_id=room_id,
            )
            for k, v in obj_data.get("states", {}).items():
                obj_entity.set_state(k, str(v))
            obj_entity.set_state("takeable", str(obj_data.get("takeable", True)))
            if obj_data.get("description"):
                obj_entity.set_state("description", obj_data["description"][:100])
            entities.append(obj_entity)

            # Object located in room
            relationships.append(Relationship(
                id=f"rel_located_{obj_id}_{room_id}",
                source_id=obj_id,
                relation_type=RelationType.LOCATED_IN,
                target_id=room_id,
                confidence=Confidence(1.0),
                last_observed=time.time(),
            ))

        # Inventory objects
        for inv_data in raw_obs.get("inventory", []):
            inv_id = inv_data.get("id", str(uuid.uuid4()))
            inv_entity = Entity(
                id=inv_id,
                name=inv_data.get("name", "Item"),
                category=EntityCategory.INVENTORY,
            )
            entities.append(inv_entity)

        return Observation(
            id=str(uuid.uuid4()),
            source_type=ObservationType.TEXT,
            raw_text=json.dumps(raw_obs, indent=2),
            entities=entities,
            relationships=relationships,
            current_room_id=room_id,
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------ #
    # Track 2: Vision Detection Extraction                                  #
    # ------------------------------------------------------------------ #

    def extract_from_vision_obs(
        self, vision_result: Dict[str, Any], scene_room_id: str = "camera_scene"
    ) -> Observation:
        """
        Converts vision detector output into a structured Observation.

        Args:
            vision_result: dict from OpenCVVisionDetector.detect_objects_and_faces()
            scene_room_id: logical ID of the visual scene (e.g. 'camera_scene').

        Returns:
            Structured Observation with detected entities and spatial relationships.
        """
        entities: List[Entity] = []
        relationships: List[Relationship] = []

        # Visual scene as a virtual "room"
        scene_entity = Entity(
            id=scene_room_id,
            name="Camera Scene",
            category=EntityCategory.ROOM,
        )
        scene_entity.set_state("source", "vision_stream")
        frame_size = vision_result.get("frame_size", {})
        scene_entity.set_state("frame_width", str(frame_size.get("width", 0)))
        scene_entity.set_state("frame_height", str(frame_size.get("height", 0)))
        entities.append(scene_entity)

        all_detections = (
            vision_result.get("objects", [])
            + vision_result.get("people", [])
            + vision_result.get("faces", [])
        )

        for idx, det in enumerate(all_detections):
            raw_name = det.get("name", f"entity_{idx}")
            cat_str = det.get("category", "object")
            try:
                cat = EntityCategory(cat_str)
            except ValueError:
                cat = EntityCategory.OBJECT

            # Stable ID from position to allow entity tracking across frames
            bbox_dict = det.get("bounding_box", {})
            bbox_id_str = f"{int(bbox_dict.get('xmin', 0))}_{int(bbox_dict.get('ymin', 0))}"
            entity_id = f"{cat_str}_{raw_name.lower().replace(' ', '_')}_{bbox_id_str}"

            conf = Confidence(float(det.get("confidence", 0.9)))
            bbox = None
            if bbox_dict:
                bbox = BoundingBox(
                    xmin=float(bbox_dict.get("xmin", 0)),
                    ymin=float(bbox_dict.get("ymin", 0)),
                    xmax=float(bbox_dict.get("xmax", 0)),
                    ymax=float(bbox_dict.get("ymax", 0)),
                    label=raw_name,
                    confidence=conf.value,
                )

            detected_entity = Entity(
                id=entity_id,
                name=raw_name,
                category=cat,
                room_id=scene_room_id,
                confidence=conf,
                bounding_box=bbox,
            )
            detected_entity.set_state("confidence", str(round(conf.value, 3)))
            entities.append(detected_entity)

            relationships.append(Relationship(
                id=f"rel_detected_{entity_id}_{scene_room_id}",
                source_id=entity_id,
                relation_type=RelationType.LOCATED_IN,
                target_id=scene_room_id,
                confidence=conf,
                last_observed=time.time(),
            ))

        return Observation(
            id=str(uuid.uuid4()),
            source_type=ObservationType.VISION,
            raw_text=json.dumps({k: v for k, v in vision_result.items() if k != "frame"}, indent=2),
            entities=entities,
            relationships=relationships,
            current_room_id=scene_room_id,
            timestamp=time.time(),
        )
