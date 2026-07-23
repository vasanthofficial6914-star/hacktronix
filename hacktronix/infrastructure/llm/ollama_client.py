"""
Local LLM Provider – Ollama Client.

Connects to a running Ollama instance (gemma:2b / qwen2.5:3b / llama3.2)
and issues structured prompts with JSON output parsing.
"""

import json
import re
from typing import Any, Dict

import httpx

from hacktronix.domain.interfaces import ILLMProvider


SYSTEM_PROMPT = """You are an autonomous World Model Agent. You make decisions STRICTLY based on the provided Current World Slice.
Do NOT assume any facts outside of this slice. NEVER reference past conversations.
Always respond in valid JSON format only."""

ACTION_PROMPT_TEMPLATE = """
CURRENT OBJECTIVE:
{objective}

RELEVANT WORLD SLICE:
{world_slice}

INSTRUCTIONS:
Analyze the objective and the visible entities, relationships, and inventory in the World Slice above.
Select the single BEST next action. Valid actions are:
  - go <direction>        (explore a connected room)
  - take <object_name>    (pick up an object)
  - examine <object_name> (inspect an object for clues)
  - open <object_name>    (open a container or door)
  - use <object_name>     (use an item from inventory)
  - drop <object_name>    (drop an inventory item)

Think step by step (1-2 sentences), then output your action.

OUTPUT FORMAT (JSON only, no markdown):
{{
  "reasoning": "<concise 1-2 sentence explanation>",
  "action": "<action_verb>",
  "target": "<entity_name_or_direction>"
}}
"""

EXTRACT_PROMPT_TEMPLATE = """
Convert the following text description of a room into structured JSON.
Extract rooms, objects, their states, and relationships.

TEXT:
{text}

OUTPUT FORMAT (JSON only):
{{
  "room": {{"name": "", "description": ""}},
  "objects": [{{"name": "", "states": {{}}, "description": ""}}],
  "relationships": [{{"source": "", "type": "", "target": ""}}]
}}
"""


class OllamaLLMProvider(ILLMProvider):
    """
    Connects to a local Ollama server for LLM inference.
    Falls back to deterministic mock responses if Ollama is unreachable.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def _call_ollama(self, prompt: str) -> str:
        """Low-level call to Ollama /api/generate endpoint."""
        try:
            payload = {
                "model": self.model,
                "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 512},
            }
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()
        except Exception as e:
            return json.dumps({
                "reasoning": f"Ollama unavailable ({str(e)[:60]}). Using heuristic fallback.",
                "action": "explore",
                "target": "room"
            })

    def generate_action(self, objective: str, world_slice_text: str) -> Dict[str, Any]:
        """Generate the agent's next structured action."""
        prompt = ACTION_PROMPT_TEMPLATE.format(
            objective=objective,
            world_slice=world_slice_text,
        )
        raw = self._call_ollama(prompt)
        return self._parse_json_response(raw, default_action="explore", default_target="room")

    def extract_structured_observation(self, text_description: str) -> Dict[str, Any]:
        """Extract structured JSON from unstructured text."""
        prompt = EXTRACT_PROMPT_TEMPLATE.format(text=text_description)
        raw = self._call_ollama(prompt)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {"room": {}, "objects": [], "relationships": []}

    @staticmethod
    def _parse_json_response(raw: str, default_action: str, default_target: str) -> Dict[str, Any]:
        """Extract JSON dict from raw LLM output (handles markdown code blocks)."""
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(clean)
            return {
                "reasoning": str(data.get("reasoning", "No reasoning provided.")),
                "action": str(data.get("action", default_action)).lower(),
                "target": str(data.get("target", default_target)),
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "reasoning": f"Raw LLM response: {raw[:200]}",
                "action": default_action,
                "target": default_target,
            }
