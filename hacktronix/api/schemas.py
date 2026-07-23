"""
Pydantic Schemas for FastAPI REST API.

Request and response models with full type validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ObservationRequest(BaseModel):
    source_type: str = "text"
    raw_text: str
    current_room_id: Optional[str] = None


class QueryRequest(BaseModel):
    objective: str
    current_room_id: Optional[str] = None
    top_k: int = 5


class AgentStepRequest(BaseModel):
    objective: str
    use_ollama: bool = False


class AgentRunRequest(BaseModel):
    objective: str
    max_steps: int = 15
    use_ollama: bool = False


class DetectionRequest(BaseModel):
    image_base64: Optional[str] = None


class EntityResponse(BaseModel):
    id: str
    name: str
    category: str
    room_id: Optional[str]
    confidence: float
    states: Dict[str, Any]


class RelationshipResponse(BaseModel):
    id: str
    source_id: str
    relation_type: str
    target_id: str
    confidence: float


class WorldSliceResponse(BaseModel):
    objective: str
    current_room: Optional[EntityResponse]
    visible_entities: List[EntityResponse]
    relationships: List[RelationshipResponse]
    inventory: List[EntityResponse]
    graph_context_summary: str
    formatted_text: str


class AgentStepResponse(BaseModel):
    step: int
    room: str
    reasoning: str
    action: str
    result: str
    goal_achieved: bool
    world_slice: str


class HistoryItem(BaseModel):
    version_id: int
    event_type: str
    entity_id: Optional[str]
    description: str
    timestamp: float


class SystemState(BaseModel):
    total_entities: int
    total_relationships: int
    inventory_count: int
    timeline_events: int
    graph_nodes: int
    graph_edges: int


class UpdateResult(BaseModel):
    added: List[str]
    updated: List[str]
    conflicts_resolved: List[str]
    timestamp: float
