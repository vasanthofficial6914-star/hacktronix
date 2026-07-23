"""
Value Objects for HackModel-AI Domain.

Defines immutable value objects, enums, confidence bounds, bounding boxes, and states.
Follows DDD principles (immutable, self-validating).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import time


class EntityCategory(str, Enum):
    """Supported entity categories in the World Model."""
    ROOM = "room"
    OBJECT = "object"
    PERSON = "person"
    FACE = "face"
    DOOR = "door"
    INVENTORY = "inventory"
    UNKNOWN = "unknown"


class RelationType(str, Enum):
    """Semantic relationship types between entities in the World Model."""
    CONNECTED_TO = "connected_to"
    LOCATED_IN = "located_in"
    CONTAINS = "contains"
    CARRIED_BY = "carried_by"
    ON_TOP_OF = "on_top_of"
    NEXT_TO = "next_to"
    NEAR = "near"
    HAS_STATE = "has_state"
    FACES = "faces"


class ObservationType(str, Enum):
    """Origin source of observation data."""
    TEXT = "text"
    VISION = "vision"
    SIMULATED = "simulated"


@dataclass(frozen=True)
class Confidence:
    """
    Confidence score bounded in range [0.0, 1.0].
    Includes exponential decay capability for temporal uncertainty tracking.
    """
    value: float = 1.0

    def __post_init__(self) -> None:
        clamped = max(0.0, min(1.0, float(self.value)))
        object.__setattr__(self, "value", round(clamped, 4))

    def decay(self, elapsed_seconds: float, decay_rate: float = 0.01) -> "Confidence":
        """
        Applies exponential decay over elapsed time: C(t) = C_0 * e^(-lambda * dt)
        """
        import math
        new_val = self.value * math.exp(-decay_rate * elapsed_seconds)
        return Confidence(new_val)

    def is_stale(self, threshold: float = 0.15) -> bool:
        """Returns True if confidence drops below threshold."""
        return self.value < threshold

    def merge_bayes(self, other: "Confidence") -> "Confidence":
        """Combines two independent confidence observations using Bayesian update."""
        p1 = self.value
        p2 = other.value
        num = p1 * p2
        den = num + ((1.0 - p1) * (1.0 - p2))
        return Confidence(num / den if den > 0 else 0.5)


@dataclass(frozen=True)
class BoundingBox:
    """
    Normalized or absolute 2D Bounding Box for Computer Vision detections.
    Coordinates: [xmin, ymin, xmax, ymax]
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = "object"
    confidence: float = 1.0

    @property
    def area(self) -> float:
        """Returns the bounding box area."""
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)

    @property
    def center(self) -> tuple[float, float]:
        """Returns the center (cx, cy) of the box."""
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert bounding box to dictionary format."""
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "label": self.label,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class StateAttribute:
    """
    Represents a dynamic state attribute of an entity (e.g. key="status", value="open").
    """
    key: str
    value: str
    confidence: Confidence = Confidence(1.0)
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if self.updated_at == 0.0:
            object.__setattr__(self, "updated_at", time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence.value,
            "updated_at": self.updated_at,
        }
