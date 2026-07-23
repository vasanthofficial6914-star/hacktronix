"""
FastAPI Routes for HackModel-AI REST API.

Endpoints:
  POST /detect    - Vision detection on image
  POST /update    - Update world model with observation
  POST /query     - Retrieve relevant world slice
  GET  /world     - Full world state snapshot
  POST /chat      - Single agent reasoning step
  POST /run       - Full agent run (multi-step loop)
  GET  /history   - State history timeline
  GET  /state     - System health and statistics
"""

import json
import base64
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from hacktronix.api.schemas import (
    ObservationRequest,
    QueryRequest,
    AgentStepRequest,
    AgentRunRequest,
    DetectionRequest,
    EntityResponse,
    RelationshipResponse,
    WorldSliceResponse,
    AgentStepResponse,
    HistoryItem,
    SystemState,
    UpdateResult,
)
from hacktronix.application.world_model_service import build_world_model_stack

# Build the full DI stack once at module level (singleton)
_stack = build_world_model_stack()
repository = _stack["repository"]
graph_store = _stack["graph_store"]
vector_store = _stack["vector_store"]
extractor = _stack["extractor"]
updater = _stack["updater"]
query_layer = _stack["query_layer"]
text_env = _stack["text_env"]
video_env = _stack["video_env"]
agent = _stack["agent"]

router = APIRouter()


# ─────────────────────────────────────────────────────── #
#  POST /detect                                           #
# ─────────────────────────────────────────────────────── #

@router.post("/detect")
async def detect_objects(req: DetectionRequest) -> Dict[str, Any]:
    """
    Run object/face detection on a provided base64 image or synthetic frame.
    """
    try:
        detector = video_env.stream.__class__  # reuse VideoStreamManager
        from hacktronix.infrastructure.vision.detector import OpenCVVisionDetector
        det = OpenCVVisionDetector()

        if req.image_base64:
            img_bytes = base64.b64decode(req.image_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            import cv2
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            result = det.detect_objects_and_faces(frame)
        else:
            # Synthetic frame
            result = det.detect_objects_and_faces(None)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────── #
#  POST /update                                           #
# ─────────────────────────────────────────────────────── #

@router.post("/update", response_model=UpdateResult)
async def update_world(req: ObservationRequest) -> UpdateResult:
    """Update the World Model with a raw text observation."""
    try:
        if req.source_type == "vision":
            obs = extractor.extract_from_vision_obs(
                {"objects": [], "people": [], "faces": [], "frame_size": {"width": 0, "height": 0}},
                scene_room_id=req.current_room_id or "camera_scene"
            )
        else:
            # Parse as text-env observation (simple key: description format)
            raw_env_obs = {
                "current_room": {
                    "id": req.current_room_id or "unknown",
                    "name": req.current_room_id or "Unknown Room",
                    "description": req.raw_text,
                    "exits": {},
                },
                "objects_in_room": [],
                "inventory": [],
            }
            obs = extractor.extract_from_text_obs(raw_env_obs)

        summary = updater.process_observation(obs)
        return UpdateResult(
            added=summary.get("added", []),
            updated=summary.get("updated", []),
            conflicts_resolved=summary.get("conflicts_resolved", []),
            timestamp=summary.get("timestamp", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────── #
#  POST /query                                            #
# ─────────────────────────────────────────────────────── #

@router.post("/query", response_model=WorldSliceResponse)
async def query_world(req: QueryRequest) -> WorldSliceResponse:
    """Retrieve relevant World Slice for a given objective."""
    try:
        world_slice = query_layer.retrieve_slice(
            objective=req.objective,
            current_room_id=req.current_room_id,
            top_k=req.top_k,
        )
        return _slice_to_response(world_slice, req.objective)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────── #
#  GET /world                                             #
# ─────────────────────────────────────────────────────── #

@router.get("/world")
async def get_full_world() -> Dict[str, Any]:
    """Returns full World Model state."""
    try:
        entities = repository.get_all_entities()
        relationships = repository.get_all_relationships()
        inventory = repository.get_inventory()
        return {
            "entities": [e.to_dict() for e in entities],
            "relationships": [r.to_dict() for r in relationships],
            "inventory": [i.to_dict() for i in inventory],
            "total_entities": len(entities),
            "total_relationships": len(relationships),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────── #
#  POST /chat  (single agent step)                       #
# ─────────────────────────────────────────────────────── #

@router.post("/chat", response_model=AgentStepResponse)
async def chat_agent_step(req: AgentStepRequest) -> AgentStepResponse:
    """Execute one agent reasoning step and return the result."""
    try:
        result = agent.step_once(req.objective)
        return AgentStepResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────── #
#  POST /run  (full loop)                                 #
# ─────────────────────────────────────────────────────── #

@router.post("/run")
async def run_agent(req: AgentRunRequest) -> Dict[str, Any]:
    """Run the agent autonomously for up to max_steps."""
    try:
        agent.max_steps = req.max_steps
        result = agent.run(req.objective)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────── #
#  GET /history                                           #
# ─────────────────────────────────────────────────────── #

@router.get("/history", response_model=List[HistoryItem])
async def get_history(limit: int = 20) -> List[HistoryItem]:
    """Returns world state version history."""
    try:
        entries = repository.get_state_history(limit=limit)
        return [
            HistoryItem(
                version_id=e.version_id,
                event_type=e.event_type,
                entity_id=e.entity_id,
                description=e.description,
                timestamp=e.timestamp,
            )
            for e in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────── #
#  GET /state                                             #
# ─────────────────────────────────────────────────────── #

@router.get("/state", response_model=SystemState)
async def get_system_state() -> SystemState:
    """Returns overall system health and entity statistics."""
    try:
        entities = repository.get_all_entities()
        rels = repository.get_all_relationships()
        inv = repository.get_inventory()
        timeline = repository.get_timeline(limit=999)
        return SystemState(
            total_entities=len(entities),
            total_relationships=len(rels),
            inventory_count=len(inv),
            timeline_events=len(timeline),
            graph_nodes=graph_store.graph.number_of_nodes(),
            graph_edges=graph_store.graph.number_of_edges(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────── #
#  Helpers                                               #
# ─────────────────────────────────────────────────────── #

def _entity_to_response(ent) -> EntityResponse:
    d = ent.to_dict()
    return EntityResponse(
        id=d["id"],
        name=d["name"],
        category=d["category"],
        room_id=d.get("room_id"),
        confidence=d["confidence"],
        states={k: v["value"] if isinstance(v, dict) else v for k, v in d.get("states", {}).items()},
    )


def _slice_to_response(world_slice, objective: str) -> WorldSliceResponse:
    return WorldSliceResponse(
        objective=objective,
        current_room=_entity_to_response(world_slice.current_room) if world_slice.current_room else None,
        visible_entities=[_entity_to_response(e) for e in world_slice.visible_entities],
        relationships=[
            RelationshipResponse(
                id=r.id,
                source_id=r.source_id,
                relation_type=r.relation_type.value if hasattr(r.relation_type, "value") else str(r.relation_type),
                target_id=r.target_id,
                confidence=r.confidence.value,
            )
            for r in world_slice.relationships
        ],
        inventory=[_entity_to_response(i) for i in world_slice.inventory],
        graph_context_summary=world_slice.graph_context_summary,
        formatted_text=world_slice.format_as_text_slice(),
    )
