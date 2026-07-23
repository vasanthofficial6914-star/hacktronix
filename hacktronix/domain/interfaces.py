"""
Abstract Domain Interfaces (Contracts).

Defines Repository & Service contracts using ABCs to ensure clean architecture
and dependency inversion.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from hacktronix.domain.entities import (
    Entity,
    Relationship,
    Observation,
    WorldSlice,
    StateHistoryEntry,
    TimelineEntry,
)
from hacktronix.domain.value_objects import EntityCategory


class IWorldRepository(ABC):
    """Abstract Repository for World Model Persistence."""

    @abstractmethod
    def save_entity(self, entity: Entity) -> None:
        """Save or update an entity."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve entity by ID."""
        pass

    @abstractmethod
    def get_entities_by_category(self, category: EntityCategory) -> List[Entity]:
        """Get all entities matching a category."""
        pass

    @abstractmethod
    def get_all_entities(self) -> List[Entity]:
        """Retrieve all active entities."""
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity by ID."""
        pass

    @abstractmethod
    def save_relationship(self, relationship: Relationship) -> None:
        """Save or update a relationship between entities."""
        pass

    @abstractmethod
    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Retrieve relationships where entity is source or target."""
        pass

    @abstractmethod
    def get_all_relationships(self) -> List[Relationship]:
        """Retrieve all active relationships."""
        pass

    @abstractmethod
    def add_to_inventory(self, entity_id: str) -> None:
        """Add item to agent inventory."""
        pass

    @abstractmethod
    def remove_from_inventory(self, entity_id: str) -> None:
        """Remove item from agent inventory."""
        pass

    @abstractmethod
    def get_inventory(self) -> List[Entity]:
        """Retrieve all items in agent inventory."""
        pass

    @abstractmethod
    def add_state_history(self, event_type: str, description: str, snapshot_json: str, entity_id: Optional[str] = None) -> int:
        """Log state change into version history."""
        pass

    @abstractmethod
    def get_state_history(self, limit: int = 50) -> List[StateHistoryEntry]:
        """Retrieve history snapshots."""
        pass

    @abstractmethod
    def add_timeline_event(self, source_type: str, raw_obs: str, parsed_json: str) -> TimelineEntry:
        """Log observation event in timeline."""
        pass

    @abstractmethod
    def get_timeline(self, limit: int = 50) -> List[TimelineEntry]:
        """Retrieve observation timeline events."""
        pass


class IGraphStore(ABC):
    """Abstract Knowledge Graph Engine (NetworkX wrapper)."""

    @abstractmethod
    def update_graph(self, entities: List[Entity], relationships: List[Relationship]) -> None:
        """Update knowledge graph nodes and edges."""
        pass

    @abstractmethod
    def get_subgraph_nodes(self, center_node_id: str, radius: int = 1) -> List[str]:
        """Get entity IDs within N hops from center node."""
        pass

    @abstractmethod
    def export_pyvis_html(self, output_path: str) -> str:
        """Render and export interactive HTML string/filepath of the graph."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear graph nodes and edges."""
        pass


class IVectorStore(ABC):
    """Abstract Vector Store for Semantic Search (FAISS wrapper)."""

    @abstractmethod
    def index_entity(self, entity: Entity) -> None:
        """Generate embedding and index entity."""
        pass

    @abstractmethod
    def search_relevant_entities(self, query: str, top_k: int = 5) -> List[str]:
        """Perform semantic search and return relevant entity IDs."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear vector index."""
        pass


class IVisionDetector(ABC):
    """Abstract Computer Vision Engine (YOLOv11 + MediaPipe + OpenCV wrapper)."""

    @abstractmethod
    def detect_objects_and_faces(self, image_input: Any) -> Dict[str, Any]:
        """Detect objects, people, and faces from image array/file."""
        pass


class ILLMProvider(ABC):
    """Abstract LLM Provider (Ollama / Local Models / Mock)."""

    @abstractmethod
    def generate_action(self, objective: str, world_slice_text: str) -> Dict[str, Any]:
        """Generate structured next action from objective and world slice."""
        pass

    @abstractmethod
    def extract_structured_observation(self, text_description: str) -> Dict[str, Any]:
        """Convert unstructured text into structured room/object JSON."""
        pass
