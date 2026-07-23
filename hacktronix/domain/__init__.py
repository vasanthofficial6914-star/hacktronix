"""
Domain Layer: Entities, Value Objects, and Abstract Interfaces.
Contains pure Python domain logic with zero external framework dependencies.
"""

from hacktronix.domain.value_objects import (
    EntityCategory,
    RelationType,
    ObservationType,
    Confidence,
    BoundingBox,
    StateAttribute,
)
from hacktronix.domain.entities import (
    Entity,
    Relationship,
    Observation,
    WorldSlice,
    StateHistoryEntry,
    TimelineEntry,
)
from hacktronix.domain.interfaces import (
    IWorldRepository,
    IGraphStore,
    IVectorStore,
    IVisionDetector,
    ILLMProvider,
)

__all__ = [
    "EntityCategory",
    "RelationType",
    "ObservationType",
    "Confidence",
    "BoundingBox",
    "StateAttribute",
    "Entity",
    "Relationship",
    "Observation",
    "WorldSlice",
    "StateHistoryEntry",
    "TimelineEntry",
    "IWorldRepository",
    "IGraphStore",
    "IVectorStore",
    "IVisionDetector",
    "ILLMProvider",
]
