"""
Deterministic Mock LLM Provider.

Used in tests and offline demos. Provides rule-based, predictable actions
without requiring any model downloads or running servers.
"""

import re
from typing import Any, Dict

from hacktronix.domain.interfaces import ILLMProvider


class MockLLMProvider(ILLMProvider):
    """
    Rule-based LLM mock that parses the World Slice to produce deterministic actions.
    Enables full test coverage and offline hackathon demonstrations.
    """

    def generate_action(self, objective: str, world_slice_text: str) -> Dict[str, Any]:
        """
        Deterministic action generator using keyword matching on world slice text.
        Priority order: take goal > use key > open locked > take key > explore.
        """
        objective_lower = objective.lower()
        slice_lower = world_slice_text.lower()

        # Priority 1: If goal item visible, take it
        goal_words = [w for w in objective_lower.split() if len(w) > 3]
        for word in goal_words:
            if word in slice_lower and "take" not in objective_lower:
                visible_items = self._extract_visible_entities(world_slice_text, category="object")
                for item_name in visible_items:
                    if word in item_name.lower():
                        return {
                            "reasoning": f"Goal item '{word}' is visible in the current scene. Taking it immediately.",
                            "action": "take",
                            "target": item_name,
                        }

        # Priority 2: If holding a key and there's a locked item, open it
        if "key" in slice_lower and "locked" in slice_lower and "inventory" in slice_lower:
            locked_items = re.findall(r'\[OBJECT\] ([\w\s]+)\s*\(.*locked=true', world_slice_text)
            if locked_items:
                return {
                    "reasoning": "Holding a key and a locked object is present. Opening it.",
                    "action": "open",
                    "target": locked_items[0].strip(),
                }

        # Priority 3: Take visible key if not in inventory
        if "key" in slice_lower and "inventory" not in slice_lower.split("key")[0]:
            key_items = self._extract_visible_entities(world_slice_text, category="object")
            for item in key_items:
                if "key" in item.lower():
                    return {
                        "reasoning": "A key is visible. Taking it as it will likely be useful.",
                        "action": "take",
                        "target": item,
                    }

        # Priority 4: Examine unexamined interesting items
        for keyword in ["chest", "note", "orb", "gem"]:
            if keyword in slice_lower:
                return {
                    "reasoning": f"Interesting item '{keyword}' is visible. Examining it for clues.",
                    "action": "examine",
                    "target": keyword,
                }

        # Priority 5: Navigate toward unexplored exits
        exits = re.findall(r'exit_(\w+)=(\w+_\w+)', world_slice_text)
        if exits:
            direction, target_room = exits[0]
            return {
                "reasoning": f"No immediate goal items visible. Exploring {direction} toward {target_room}.",
                "action": "go",
                "target": direction,
            }

        return {
            "reasoning": "No clear action identified. Exploring to gather more information.",
            "action": "explore",
            "target": "room",
        }

    def extract_structured_observation(self, text_description: str) -> Dict[str, Any]:
        """Returns a minimal structured observation from text (used in testing)."""
        return {
            "room": {"name": "Unknown", "description": text_description[:100]},
            "objects": [],
            "relationships": [],
        }

    @staticmethod
    def _extract_visible_entities(world_slice_text: str, category: str = "object") -> list:
        """Extract entity names of a given category from formatted world slice text."""
        pattern = rf'\[{category.upper()}\]\s+([\w\s]+)\s*\(ID:'
        return re.findall(pattern, world_slice_text, re.IGNORECASE)
